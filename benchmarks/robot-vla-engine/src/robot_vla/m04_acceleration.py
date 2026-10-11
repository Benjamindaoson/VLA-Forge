"""Opt-in static CUDA graph for the frozen SmolVLA sampler; FP32 unchanged."""

import time

import torch


def validate_tensor_inputs(reference, incoming):
    if len(reference) != len(incoming):
        raise ValueError("CUDA graph input count changed")
    for old, new in zip(reference, incoming, strict=True):
        if (old.shape, old.dtype, old.device) != (new.shape, new.dtype, new.device):
            raise ValueError("CUDA graph input metadata changed; recapture required")


class CudaGraphSampler:
    """Capture pure sample_actions with explicit fresh noise and static shapes.

    Input values are copied on every call. The returned clone owns its storage.
    Initialization preserves RNG state and is reported separately as cold cost.
    Unsupported RTC arguments and metadata changes fail closed.
    """

    def __init__(self, model):
        self.model = model
        self.original = model.sample_actions
        self.embed_prefix = model.embed_prefix
        self.embed_suffix = model.embed_suffix
        self.graph = None
        self.cold_seconds = None

    @torch.inference_mode()
    def __call__(self, images, masks, tokens, lang_masks, state, noise=None, **kwargs):
        if kwargs and any(value is not None for value in kwargs.values()):
            raise ValueError("RTC/dynamic sampler arguments are unsupported")
        if noise is None:
            noise = self.model.sample_noise(
                (state.shape[0], self.model.config.chunk_size, self.model.config.max_action_dim),
                state.device,
            )
        prefix = self.embed_prefix(images, masks, tokens, lang_masks, state=state)
        incoming = [*prefix, noise]
        if self.graph is None:
            if not all(x.is_cuda for x in incoming):
                raise ValueError("CUDA graph requires CUDA inputs")
            started = time.perf_counter()
            self.static = [x.clone() for x in incoming]

            def invoke():
                # Vision preprocessing uses dynamic boolean indexing and stays eager.
                # The graph captures only prefix attention + all denoising steps.
                previous = self.model.embed_prefix
                previous_suffix = self.model.embed_suffix
                self.model.embed_prefix = lambda *a, **k: tuple(self.static[:3])
                self.model.embed_suffix = self._capture_suffix
                try:
                    return self.original(
                        images, masks, tokens, lang_masks, state, noise=self.static[-1]
                    )
                finally:
                    self.model.embed_prefix = previous
                    self.model.embed_suffix = previous_suffix

            # Warmup on a side stream before capture, without consuming caller RNG.
            with torch.random.fork_rng(devices=[state.device]):
                stream = torch.cuda.Stream(device=state.device)
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        invoke()
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    self.output = invoke()
                torch.cuda.synchronize()
            self.cold_seconds = time.perf_counter() - started
        validate_tensor_inputs(self.static, incoming)
        for old, new in zip(self.static, incoming, strict=True):
            old.copy_(new)
        self.graph.replay()
        return self.output.clone()

    def _capture_suffix(self, noisy_actions, timestep):
        # Algebra identical to pinned LeRobot embed_suffix. Its constant all-one
        # attention mask is created directly on GPU rather than copied from CPU.
        from lerobot.policies.smolvla.modeling_smolvla import create_sinusoidal_pos_embedding

        action_emb = self.model.action_in_proj(noisy_actions)
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.model.vlm_with_expert.expert_hidden_size,
            self.model.config.min_period,
            self.model.config.max_period,
            device=action_emb.device,
        ).to(dtype=action_emb.dtype)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        value = torch.cat([action_emb, time_emb], dim=2)
        value = self.model.action_time_mlp_out(
            torch.nn.functional.silu(self.model.action_time_mlp_in(value))
        )
        padding = torch.ones(value.shape[:2], dtype=torch.bool, device=value.device)
        attention = torch.ones(value.shape[:2], dtype=value.dtype, device=value.device)
        return value, padding, attention
