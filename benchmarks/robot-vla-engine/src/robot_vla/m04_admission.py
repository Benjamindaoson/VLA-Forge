"""Fail-closed source identity and legal continuous-action supervision checks."""

import numpy as np


def source_group(suite, task_id, init_state_id):
    suite = suite.removeprefix("libero_")
    suite = "long" if suite == "10" else suite
    if suite not in {"spatial", "object", "goal", "long"}:
        raise ValueError("unknown suite")
    return f"{suite}:{int(task_id)}:{int(init_state_id)}"


def validate_supervision(actions, states):
    actions, states = np.asarray(actions), np.asarray(states)
    if (
        actions.ndim != 2
        or actions.shape[1] != 7
        or states.shape != (len(actions), 8)
        or not len(actions)
    ):
        raise ValueError("action/observation alignment")
    if not np.isfinite(actions).all() or not np.isfinite(states).all():
        raise ValueError("non-finite supervision")
    if np.max(np.abs(actions)) > 1.000001:
        raise ValueError("action bounds exceed relative controller contract")
    return {"transitions": len(actions), "maximum_absolute_action": float(np.max(np.abs(actions)))}


def validate_identity(row, source, suite, task_id, init_state_id):
    if row.get("success") is not True or source.get("success") is not False:
        raise ValueError("expected successful correction of a recorded failure")
    if row.get("source_m02_episode") != source.get("episode_id"):
        raise ValueError("source episode mismatch")
    if source_group(row["suite"], row["task_id"], row["init_state_id"]) != source_group(
        suite, task_id, init_state_id
    ):
        raise ValueError("source task/init-state mismatch")
