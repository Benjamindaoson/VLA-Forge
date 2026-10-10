"""Thin integration of pinned official policies, with source-specific processors."""

from pathlib import Path

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors

from .preflight import verified_normalizer
from .protocol import load_protocol


def load_smol_checkpoint(config, path):
    from accelerate import init_empty_weights
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from safetensors.torch import load_file

    with init_empty_weights():
        policy = SmolVLAPolicy(config)
    state = load_file(str(Path(path) / "model.safetensors"))
    policy.load_state_dict(state, strict=True, assign=True)
    del state
    if any(p.is_meta for p in policy.parameters()):
        raise ValueError("unloaded meta parameters")
    return policy.float().to(config.device)


def build_policy(name, root, device="cpu", tiny=False):
    root = Path(root)
    size = 64 if tiny else 256
    inputs = {
        "observation.state": PolicyFeature(
            FeatureType.STATE, (48 if name in {"act", "diffusion"} else 8,)
        ),
        **{
            f"observation.images.{key}": PolicyFeature(FeatureType.VISUAL, (3, size, size))
            for key in ["image", "image2"]
        },
    }
    output = {"action": PolicyFeature(FeatureType.ACTION, (7,))}
    common = dict(input_features=inputs, output_features=output, device=device)
    if name == "act":
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy

        extra = (
            dict(
                chunk_size=4,
                n_action_steps=2,
                dim_model=64,
                n_heads=4,
                dim_feedforward=128,
                n_encoder_layers=1,
                n_decoder_layers=1,
                n_vae_encoder_layers=1,
                latent_dim=8,
            )
            if tiny
            else dict(chunk_size=50, n_action_steps=10)
        )
        return ACTPolicy(ACTConfig(**common, pretrained_backbone_weights=None, **extra))
    if name == "diffusion":
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

        extra = (
            dict(
                horizon=8,
                n_action_steps=2,
                down_dims=(32, 64),
                diffusion_step_embed_dim=32,
                num_inference_steps=2,
                spatial_softmax_num_keypoints=8,
            )
            if tiny
            else dict(horizon=64, n_action_steps=10, num_inference_steps=10)
        )
        return DiffusionPolicy(DiffusionConfig(**common, pretrained_backbone_weights=None, **extra))
    if name != "smolvla":
        raise ValueError("unknown policy")
    cfg = PreTrainedConfig.from_pretrained(root / "data/models/smolvla_base")
    cfg.input_features = inputs
    cfg.output_features = output
    cfg.device = device
    cfg.load_vlm_weights = False
    cfg.vlm_model_name = str((root / "data/models/smolvlm").resolve())
    cfg.n_action_steps = 10
    # Reduced spatial resolution/denoising is engineering-only, never a formal optimization claim.
    if tiny:
        cfg.resize_imgs_with_padding = (64, 64)
        cfg.num_steps = 2
    return load_smol_checkpoint(cfg, root / "data/models/smolvla_base")


def processors(policy, root, condition):
    protocol = load_protocol(Path(root) / "configs/data_protocol_v1.json")
    raw = verified_normalizer(root, protocol, condition)
    stats = {
        k: {name: torch.tensor(v) for name, v in values.items() if name != "count"}
        for k, values in raw["stats"].items()
    }
    if policy.config.type in {"act", "diffusion"}:
        for name in ["mean", "std", "min", "max"]:
            constant = 1.0 if name in {"std", "max"} else 0.0
            stats["observation.state"][name] = torch.cat(
                [stats["observation.state"][name], torch.full((40,), constant)]
            )
    for key in policy.config.image_features:
        # Fixed constants, not source/global image statistics, avoid held-out information.
        stats[key] = {"mean": torch.full((3, 1, 1), 0.5), "std": torch.full((3, 1, 1), 0.5)}
    return make_pre_post_processors(policy.config, dataset_stats=stats)


def make_dataset(policy, root, protocol, condition, tiny=False):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from torchvision.transforms import Resize

    ids = [int(i) for i in protocol["conditions"][condition]["episode_ids"]]
    delta = {"action": [i / 10 for i in policy.config.action_delta_indices]}
    if policy.config.observation_delta_indices is not None:
        for key in policy.config.input_features:
            delta[key] = [i / 10 for i in policy.config.observation_delta_indices]
    return LeRobotDataset(
        "lerobot/libero",
        root=Path(root) / "data/libero",
        episodes=ids,
        revision=protocol["revision"],
        video_backend="pyav",
        delta_timestamps=delta,
        image_transforms=Resize((64, 64)) if tiny else None,
    )
