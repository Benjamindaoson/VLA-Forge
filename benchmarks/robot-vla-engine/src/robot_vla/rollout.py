"""Observable closed-loop execution; simulator errors never count as failed policies."""

import json
import time

import numpy as np

from .metrics import summarize_latency
from .schema import content_hash
from .telemetry import Timer


def record_episode(
    env,
    policy,
    run,
    *,
    suite,
    task_id,
    episode_id,
    seed,
    instruction,
    max_steps,
    action_dim=7,
    preprocess=lambda x: x,
    postprocess=lambda x: x,
    state_extract=lambda x: x.get("state", []),
    observation_sink=None,
    synchronize=None,
    environment_config=None,
    policy_metadata=None,
):
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    identity = dict(suite=suite, task_id=str(task_id), episode_id=str(episode_id), seed=seed)
    folder = run.directory / ("rollout-" + content_hash(identity)[:16])
    folder.mkdir(exist_ok=False)
    policy.reset()
    obs, _ = env.reset(seed=seed)
    actions, states = [], [np.asarray(state_extract(obs)).copy()]
    latencies = {
        name: []
        for name in (
            "preprocess",
            "forward",
            "postprocess",
            "policy",
            "environment",
            "control_cycle",
        )
    }
    start = time.perf_counter()
    success, reason = False, "timeout"
    try:
        for step in range(max_steps):
            cycle_start = time.perf_counter()
            if observation_sink:
                observation_sink(obs, step)
            with Timer(synchronize) as policy_timer:
                with Timer(synchronize) as pre_timer:
                    batch = preprocess(obs)
                with Timer(synchronize) as forward_timer:
                    raw_action = policy.select_action(batch)
                with Timer(synchronize) as post_timer:
                    action = np.asarray(postprocess(raw_action), dtype=np.float32)
            if action.shape != (action_dim,):
                raise ValueError(f"action shape {action.shape}, expected {(action_dim,)}")
            if not np.isfinite(action).all():
                raise ValueError("nonfinite policy action")
            with Timer() as env_timer:
                obs, _, terminated, truncated, info = env.step(action)
            if "is_success" not in info or not isinstance(info["is_success"], (bool, np.bool_)):
                raise ValueError("environment must expose a measured boolean success signal")
            success = bool(info["is_success"])
            actions.append(action.copy())
            states.append(np.asarray(state_extract(obs)).copy())
            for key, timer in [
                ("preprocess", pre_timer),
                ("forward", forward_timer),
                ("postprocess", post_timer),
                ("policy", policy_timer),
                ("environment", env_timer),
            ]:
                latencies[key].append(timer.elapsed_ms)
            latencies["control_cycle"].append((time.perf_counter() - cycle_start) * 1000)
            if success or terminated or truncated:
                reason = (
                    "success"
                    if success
                    else ("environment_truncated" if truncated else "environment_terminated")
                )
                break
    except BaseException as exc:
        np.savez_compressed(
            folder / "partial_trajectory.npz",
            actions=np.asarray(actions),
            states=np.asarray(states),
        )
        run.artifact(folder / "partial_trajectory.npz", "partial_rollout")
        run.event("rollout_error", dict(**identity, error=f"{type(exc).__name__}: {exc}"))
        raise
    elapsed = time.perf_counter() - start
    trajectory = folder / "trajectory.npz"
    np.savez_compressed(
        trajectory,
        actions=np.asarray(actions),
        states=np.asarray(states),
        **{f"latency_{k}_ms": np.asarray(v) for k, v in latencies.items()},
    )
    result = dict(
        **identity,
        run_id=run.run_id,
        evaluation_run_id=run.run_id,
        instruction=instruction,
        success=success,
        episode_length=len(actions),
        termination_reason=reason,
        wall_seconds=elapsed,
        achieved_serial_control_hz=len(actions) / elapsed,
        environment_configuration=environment_config or {},
        policy=policy_metadata or {},
        latency={k: summarize_latency(v) for k, v in latencies.items()},
        forward_scope="select_action including action queue lookup; chunk calls must be profiled separately",
        trajectory_path=str(trajectory),
        metadata_path=str(folder / "rollout.json"),
    )
    chunk_calls = getattr(policy, "chunk_calls", None)
    if chunk_calls is not None:
        result["chunk_calls"] = list(chunk_calls)
        result["chunk_forward_latency"] = summarize_latency([v["latency_ms"] for v in chunk_calls])
        result["policy_inference_intervals_steps"] = np.diff(
            [v["control_step"] for v in chunk_calls]
        ).tolist()
        result["forward_scope"] = (
            "select_action including queue; chunk_forward_latency measures actual chunk generation separately"
        )
    (folder / "rollout.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    run.artifact(trajectory, "rollout_trajectory")
    run.artifact(folder / "rollout.json", "rollout_metadata")
    run.event("rollout", result)
    return result
