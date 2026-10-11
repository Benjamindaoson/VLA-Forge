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


def test_raw_m04_images_are_rotated_but_official_expert_images_are_not():
    import torch

    m = module()
    pixels = torch.arange(3 * 16 * 16).reshape(3, 16, 16).float()
    common = {
        "observation.state": torch.zeros(8),
        "action": torch.zeros(50, 7),
        "action_is_pad": torch.zeros(50, dtype=torch.bool),
        "task": "same",
    }
    raw = {
        **common,
        "observation.images.agentview_image": pixels,
        "observation.images.robot0_eye_in_hand_image": pixels,
    }
    official = {**common, "observation.images.image": pixels, "observation.images.image2": pixels}
    b = m.training_batch([raw, official], [0, 1])
    assert b["observation.images.image"][0, 0, 0, 0].item() == 255
    assert b["observation.images.image"][1, 0, 0, 0].item() == 0


def test_replay_data_identity_binds_corrective_and_expert_sources(tmp_path, monkeypatch):
    m = module()
    correction = tmp_path / "correction"
    expert = tmp_path / "expert"
    correction.mkdir()
    (expert / "data").mkdir(parents=True)
    (correction / "provenance.json").write_text('{"revision":1}')
    (expert / "data/selection.json").write_text('{"episodes":[1]}')
    monkeypatch.setattr(m, "DATA", correction)
    monkeypatch.setattr(m, "EVIDENCE", expert)
    first = m.dataset_fingerprint("replay")
    standard = m.dataset_fingerprint("standard")
    (correction / "provenance.json").write_text('{"revision":2}')
    assert m.dataset_fingerprint("replay") != first
    assert m.dataset_fingerprint("standard") == standard
