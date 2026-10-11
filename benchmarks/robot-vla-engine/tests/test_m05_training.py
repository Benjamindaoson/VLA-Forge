import importlib.util

import numpy as np
import pytest


def module():
    assert importlib.util.find_spec("robot_vla.m05_training"), (
        "M05 data and checkpoint contract missing"
    )
    from robot_vla import m05_training

    return m05_training


def test_action_window_never_crosses_episode_boundary():
    indices, padding = module().action_window(8, 10, 5)
    assert indices.tolist() == [8, 9, 9, 9, 9]
    assert padding.tolist() == [False, False, True, True, True]


def test_rejects_noncanonical_state_and_illegal_actions():
    m = module()
    m.validate_sample(np.zeros(8), np.zeros((50, 7)))
    with pytest.raises(ValueError):
        m.validate_sample(np.zeros(6), np.zeros((50, 7)))
    with pytest.raises(ValueError):
        m.validate_sample(np.zeros(8), np.ones((50, 7)) * 2)


def test_sampling_resume_restores_next_indices():
    m = module()
    a = m.IndexSampler(17, 45)
    a.take(8)
    saved = a.state_dict()
    expected = a.take(8)
    b = m.IndexSampler(17, 99)
    b.load_state_dict(saved)
    assert b.take(8).tolist() == expected.tolist()


@pytest.mark.skipif(
    __import__("os").environ.get("M05_GPU_TEST") != "1", reason="requires frozen cloud GPU assets"
)
def test_real_official_flow_matching_updates_and_reloads(tmp_path):
    m = module()
    assert hasattr(m, "train_job"), "real training entry point missing"
    r = m.train_job(
        {
            "run_id": "gpu-smoke",
            "output": str(tmp_path),
            "steps": 2,
            "microbatch": 1,
            "accumulation": 1,
            "precision": "fp32",
            "dataset": "corrective",
            "seed": 5001,
            "lr": 0.0001,
        }
    )
    assert r["optimizer_updates"] == 2
    assert r["parameter_delta_linf"] > 0
    assert r["reload_linf"] == 0
    assert r["finite_gradients"]


def test_replay_collation_normalizes_camera_names_before_stacking():
    import torch

    m = module()
    common = {
        "observation.state": torch.zeros(8),
        "action": torch.zeros(50, 7),
        "action_is_pad": torch.zeros(50, dtype=torch.bool),
        "task": "same task",
    }
    correction = {
        **common,
        "observation.images.agentview_image": torch.zeros(3, 16, 16),
        "observation.images.robot0_eye_in_hand_image": torch.ones(3, 16, 16),
        "next.done": True,
    }
    expert = {
        **common,
        "observation.images.image": torch.zeros(3, 16, 16),
        "observation.images.image2": torch.ones(3, 16, 16),
    }
    batch = m.training_batch([correction, expert], [0, 1])
    assert batch["observation.images.image2"].shape == (2, 3, 16, 16)
    assert batch["observation.images.image2"].sum() == 1536
    assert batch["action"].shape == (2, 50, 7)
