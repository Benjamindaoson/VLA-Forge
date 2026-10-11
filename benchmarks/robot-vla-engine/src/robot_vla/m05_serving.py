"""Versioned, simulation-only continuous-action API and inference engine."""

import asyncio
import base64
import io
import time
from pathlib import Path

import numpy as np

MAX_AGE_S = 10.0


def validate_request(request, version):
    if request.get("scope") != "simulation":
        raise ValueError("Only authorized simulation observations are accepted")
    try:
        observed_at = float(request["observed_at"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Missing or invalid observation time") from exc
    age = time.time() - observed_at
    if not 0 <= age <= MAX_AGE_S:
        raise ValueError("Expired or future observation")
    if request.get("version") != version:
        raise ValueError("Model version mismatch")
    try:
        state = np.asarray(request.get("state"), dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Expected finite canonical 8D state") from exc
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("Expected finite canonical 8D state")
    if not isinstance(request.get("task"), str) or not 0 < len(request["task"]) <= 1024:
        raise ValueError("Missing task instruction")
    if not isinstance(request.get("observation_id"), str) or not 0 < len(request["observation_id"]) <= 128:
        raise ValueError("Missing observation identity")
    if any(not isinstance(request.get(key), str) or not request[key] for key in ("image1", "image2")):
        raise ValueError("Missing dual-camera observation")


def validate_response(response, observation_id, version):
    if response["observation_id"] != observation_id or response["version"] != version:
        raise ValueError("Stale observation or model revision")
    if not 0 <= time.time() - response["observed_at"] <= MAX_AGE_S:
        raise ValueError("Expired actions must not reach simulator")
    actions = np.asarray(response["actions"])
    if (
        actions.shape != (50, 7)
        or not np.isfinite(actions).all()
        or np.abs(actions).max() > 1.00001
    ):
        raise ValueError("Invalid action chunk")


def camera_batch(image1, image2, state, task):
    """Convert raw LIBERO RGB cameras into the frozen model orientation."""
    import torch

    images = []
    for value in (image1, image2):
        rgb = np.asarray(value)
        if rgb.shape != (256, 256, 3) or rgb.dtype != np.uint8:
            raise ValueError("Pinned cameras require 256x256 uint8 RGB")
        image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
        images.append(torch.flip(image, dims=(-2, -1)).unsqueeze(0))
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("Expected finite canonical 8D state")
    return {
        "observation.images.image": images[0],
        "observation.images.image2": images[1],
        "observation.state": torch.from_numpy(state.copy()).unsqueeze(0),
        "task": [task],
    }


class PolicyEngine:
    def __init__(self, version, adapter=None, backend="eager"):
        import torch

        from robot_vla.m05_training import BASE, load_policy, sha256

        self.version = version
        self.backend = backend
        self.adapter = adapter
        start = time.perf_counter()
        self.policy, self.pre, self.post = load_policy(adapter, train=False, lora=False)
        self.base = self.policy.get_base_model() if adapter else self.policy
        self.graph = None
        if backend == "cuda_graph":
            from robot_vla.m04_acceleration import CudaGraphSampler

            self.graph = CudaGraphSampler(self.base.model)
            self.base.model.sample_actions = self.graph
        elif backend not in ("eager", "bf16", "fp16", "tf32", "compile"):
            raise ValueError("Unknown backend")
        if backend == "compile":
            self.base.model.sample_actions = torch.compile(
                self.base.model.sample_actions, backend="inductor", mode="reduce-overhead"
            )
        torch.backends.cuda.matmul.allow_tf32 = backend == "tf32"
        torch.backends.cudnn.allow_tf32 = backend == "tf32"
        self.model_hash = sha256(
            Path(adapter) / "adapter_model.safetensors" if adapter else BASE / "model.safetensors"
        )
        self.processor_hash = sha256(BASE / "policy_preprocessor.json")
        self.load_s = time.perf_counter() - start
        self.calls = 0

    def predict(self, image1, image2, state, task, seed=0, noise=None):
        import torch

        from robot_vla.m05_training import seed_all

        torch.set_num_threads(1)
        seed_all(seed)
        t0 = time.perf_counter()
        if tuple(np.asarray(state).shape) != (8,):
            raise ValueError("8D state required")
        batch = camera_batch(image1, image2, state, task)
        t = time.perf_counter()
        batch = self.pre(batch)
        torch.cuda.synchronize()
        prep = time.perf_counter() - t
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(self.backend)
        t = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            chunk = self.policy.predict_action_chunk(batch, noise=noise)
        torch.cuda.synchronize()
        inference = time.perf_counter() - t
        t = time.perf_counter()
        actions = self.post(chunk)[0].numpy()
        if actions.shape != (50, 7) or not np.isfinite(actions).all():
            raise ValueError("Nonfinite action output")
        clipped = np.clip(actions, -1, 1)
        self.calls += 1
        return {
            "actions": clipped.tolist(),
            "raw_actions": actions.tolist(),
            "processor_s": prep,
            "chunk_s": inference,
            "postprocess_s": time.perf_counter() - t,
            "predict_s": time.perf_counter() - t0,
            "clipped_values": int(np.count_nonzero(actions != clipped)),
            "model_sha256": self.model_hash,
            "processor_sha256": self.processor_hash,
            "version": self.version,
            "backend": self.backend,
            "gpu_allocated_peak": torch.cuda.max_memory_allocated(),
            "gpu_reserved_peak": torch.cuda.max_memory_reserved(),
        }


def decode_image(value):
    from PIL import Image

    if not isinstance(value, str) or len(value) > 2_000_000:
        raise ValueError("Invalid image payload size")
    data = base64.b64decode(value, validate=True)
    try:
        im = Image.open(io.BytesIO(data))
    except (OSError, ValueError) as exc:
        raise ValueError("Invalid encoded image") from exc
    if im.size != (256, 256):
        raise ValueError("Image shape mismatch")
    return np.asarray(im.convert("RGB"))


def create_app(engine, versions, log_path, max_pending=8, activate=None):
    from fastapi import FastAPI, HTTPException

    from robot_vla.m05_training import append_event

    app = FastAPI(title="VLA-Forge simulation action policy")
    app.state.engine = engine
    app.state.pending = 0
    app.state.errors = 0
    lock = asyncio.Lock()

    @app.get("/health")
    def health():
        return {"status": "ok", "scope": "simulation_only"}

    @app.get("/ready")
    def ready():
        e = app.state.engine
        return {
            "ready": e is not None,
            "version": e.version if e else None,
            "backend": e.backend if e else None,
            "pending": app.state.pending,
            "errors": app.state.errors,
            "model_sha256": e.model_hash if e else None,
        }

    @app.post("/v1/action-chunks")
    async def predict(request: dict):
        received = time.perf_counter()
        if app.state.pending >= max_pending:
            app.state.errors += 1
            raise HTTPException(429, "Queue capacity reached")
        app.state.pending += 1
        try:
            validate_request(request, app.state.engine.version)
            timeout = min(30, max(0.05, float(request.get("timeout_s", 10))))
            await asyncio.wait_for(lock.acquire(), timeout)
            try:
                validate_request(request, app.state.engine.version)
                queue = time.perf_counter() - received
                images = [decode_image(request[k]) for k in ("image1", "image2")]
                # The lock is held through inference: no timed-out thread can race a model switch.
                result = await asyncio.to_thread(
                    app.state.engine.predict,
                    *images,
                    request["state"],
                    request["task"],
                    int(request.get("seed", 0)),
                )
                result.update(
                    observation_id=request["observation_id"],
                    observed_at=request["observed_at"],
                    queue_s=queue,
                    request_s=time.perf_counter() - received,
                    observation_age_s=time.time() - request["observed_at"],
                )
                if result["request_s"] > timeout:
                    raise TimeoutError("Inference deadline exceeded; result discarded")
                validate_response(result, request["observation_id"], request["version"])
                append_event(
                    log_path,
                    {
                        "event": "request_completed",
                        **{k: v for k, v in result.items() if k not in ("actions", "raw_actions")},
                    },
                )
                return result
            finally:
                lock.release()
        except (ValueError, TimeoutError, asyncio.TimeoutError) as exc:
            app.state.errors += 1
            append_event(
                log_path,
                {
                    "event": "request_rejected",
                    "error": str(exc),
                    "request_s": time.perf_counter() - received,
                },
            )
            raise HTTPException(422 if isinstance(exc, ValueError) else 504, str(exc)) from exc
        finally:
            app.state.pending -= 1

    @app.post("/v1/development-version")
    async def switch(request: dict):
        version = request.get("version")
        backend = request.get("backend", "eager")
        rollback = request.get("rollback", False)
        if type(rollback) is not bool:
            raise HTTPException(422, "rollback must be boolean")
        if version not in versions:
            raise HTTPException(404, "Unregistered development version")
        async with lock:
            start = time.perf_counter()
            old = app.state.engine
            try:
                fresh = await asyncio.to_thread(PolicyEngine, version, versions[version], backend)
                activation = activate(version, rollback) if activate is not None else None
                app.state.engine = fresh
                del old
                import gc

                import torch

                gc.collect()
                torch.cuda.empty_cache()
            except Exception as exc:
                app.state.errors += 1
                raise HTTPException(500, "Model load failed; prior version retained") from exc
            r = {
                "version": version,
                "backend": backend,
                "switch_s": time.perf_counter() - start,
                "model_sha256": fresh.model_hash,
                "activation": activation,
                "rollback": rollback,
            }
            append_event(log_path, {"event": "development_version_changed", **r})
            return r

    return app
