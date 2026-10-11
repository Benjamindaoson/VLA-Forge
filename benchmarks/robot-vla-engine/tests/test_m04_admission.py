import numpy as np
import pytest

from robot_vla.m04_admission import source_group, validate_identity, validate_supervision


def test_aliases_form_same_source_group():
    assert source_group("libero_10", 4, 7) == source_group("long", 4, 7)
    assert source_group("libero_goal", 6, 6) == "goal:6:6"


def test_supervision_rejects_invalid_actions():
    actions = np.zeros((3, 7), dtype=np.float32)
    validate_supervision(actions, np.zeros((3, 8)))
    actions[0, 2] = 1.01
    with pytest.raises(ValueError, match="bounds"):
        validate_supervision(actions, np.zeros((3, 8)))
    actions[0, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_supervision(actions, np.zeros((3, 8)))


def test_supervision_rejects_misalignment():
    with pytest.raises(ValueError, match="alignment"):
        validate_supervision(np.zeros((3, 7)), np.zeros((2, 8)))


def test_identity_rejects_different_task_or_successful_source():
    source = {"episode_id": "goal_task04_episode04_attempt1", "success": False}
    row = {
        "source_m02_episode": source["episode_id"],
        "success": True,
        "suite": "goal",
        "task_id": 4,
        "init_state_id": 4,
    }
    validate_identity(row, source, "goal", 4, 4)
    with pytest.raises(ValueError):
        validate_identity(row, source, "goal", 6, 4)
    with pytest.raises(ValueError):
        validate_identity(row, {**source, "success": True}, "goal", 4, 4)
