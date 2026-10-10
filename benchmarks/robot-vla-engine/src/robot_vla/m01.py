"""Strict evidence accounting for the independent M01 LIBERO baseline."""

M01_SUITES = {
    "spatial": ("libero_spatial", 0),
    "object": ("libero_object", 0),
    "goal": ("libero_goal", 0),
    "long": ("libero_10", 0),
}


def fit_libero_state_to_checkpoint(state, *, expected_dim, normalizer_dim=None):
    source_dim = state.shape[-1]
    if normalizer_dim is None:
        normalizer_dim = expected_dim
    if source_dim == normalizer_dim:
        if source_dim == 8 and expected_dim == 6:
            # The pinned checkpoint's fitted normalizer stores 8-D LIBERO state
            # statistics although its model feature schema declares six values.
            # SmolVLAPolicy.prepare_state pads the normalized input to max_state_dim.
            semantics = "full LIBERO pose and gripper state; official normalizer width 8; policy feature schema width 6"
        else:
            semantics = "checkpoint_native"
        return state, semantics
    raise ValueError(
        f"unsupported LIBERO state width {source_dim}; policy schema {expected_dim}, normalizer {normalizer_dim}"
    )


def episode_success_metrics(rows):
    completed = [row for row in rows if row.get("status") == "complete"]
    if any(type(row.get("success")) is not bool for row in completed):
        raise ValueError("completed episode must have a measured boolean success")
    successes = sum(row["success"] for row in completed)
    failures = len(completed) - successes
    return {
        "completed_episodes": len(completed),
        "successes": successes,
        "failures": failures,
        "success_rate": successes / len(completed) if completed else None,
    }


def validate_smoke_coverage(rows):
    smoke = [row for row in rows if row.get("phase") == "smoke"]
    expected = {("spatial", "0", str(i)) for i in range(3)}
    actual = [(row.get("suite"), str(row.get("task_id")), str(row.get("episode_id"))) for row in smoke]
    if len(actual) != len(set(actual)):
        raise ValueError("duplicate smoke episode identity")
    if set(actual) != expected or len(smoke) != len(expected):
        raise ValueError("smoke coverage must contain exactly three spatial task 0 episodes")
    if any(row.get("status") != "complete" or type(row.get("success")) is not bool for row in smoke):
        raise ValueError("smoke coverage contains incomplete episodes")
    return len(smoke)


def validate_m01_coverage(rows):
    formal = [row for row in rows if row.get("phase") == "formal"]
    expected = {
        (suite, "0", str(episode))
        for suite in M01_SUITES
        for episode in range(10)
    }
    actual = [
        (row.get("suite"), str(row.get("task_id")), str(row.get("episode_id")))
        for row in formal
    ]
    if len(actual) != len(set(actual)):
        raise ValueError("duplicate formal episode identity")
    if set(actual) != expected or len(formal) != len(expected):
        raise ValueError("formal coverage must contain exactly ten episodes for each fixed suite task")
    if any(row.get("status") != "complete" or type(row.get("success")) is not bool for row in formal):
        raise ValueError("formal coverage contains incomplete episodes")
    return {suite: 10 for suite in M01_SUITES}
