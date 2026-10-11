import importlib.util
import time

import pytest


def module():
    assert importlib.util.find_spec("robot_vla.m05_serving"), "Simulator policy API missing"
    from robot_vla import m05_serving

    return m05_serving


def test_rejects_hardware_and_stale_observation():
    m = module()
    with pytest.raises(ValueError):
        m.validate_request({"scope": "hardware", "observed_at": time.time()}, "v1")
    with pytest.raises(ValueError):
        m.validate_request({"scope": "simulation", "observed_at": 0, "version": "v1"}, "v1")


def test_rejects_version_and_state_mismatch():
    m = module()
    r = {
        "scope": "simulation",
        "version": "v1",
        "observed_at": time.time(),
        "state": [0] * 8,
        "task": "place bowl",
        "observation_id": "step10",
    }
    m.validate_request(r, "v1")
    with pytest.raises(ValueError):
        m.validate_request({**r, "version": "old"}, "v1")
    with pytest.raises(ValueError):
        m.validate_request({**r, "state": [0] * 6}, "v1")


def test_age_checked_again_before_physics_execution():
    m = module()
    r = {
        "observation_id": "step10",
        "version": "v1",
        "observed_at": time.time(),
        "actions": [[0] * 7] * 50,
    }
    m.validate_response(r, "step10", "v1")
    with pytest.raises(ValueError):
        m.validate_response(r, "step11", "v1")
    with pytest.raises(ValueError):
        m.validate_response({**r, "observed_at": 0}, "step10", "v1")


def test_raw_simulator_cameras_follow_frozen_libero_orientation():
    import numpy as np

    m = module()
    raw = np.arange(256 * 256, dtype=np.uint8).reshape(256, 256)
    rgb = np.repeat(raw[:, :, None], 3, axis=2)
    batch = m.camera_batch(rgb, rgb, np.zeros(8), "place bowl")
    assert batch["observation.images.image"].shape == (1, 3, 256, 256)
    assert batch["observation.images.image"][0, 0, 0, 0].item() == rgb[-1, -1, 0] / 255
    assert batch["observation.images.image2"][0, 0, -1, -1].item() == rgb[0, 0, 0] / 255


def test_modelops_activation_is_atomic_with_version_switch(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    m = module()

    class FakeEngine:
        def __init__(self, version, adapter=None, backend="eager"):
            self.version = version
            self.backend = backend
            self.model_hash = version

    monkeypatch.setattr(m, "PolicyEngine", FakeEngine)
    calls = []

    def activate(version, rollback):
        calls.append((version, rollback))
        if version == "rejected":
            raise ValueError("development gate failed")
        return {"revision": len(calls)}

    app = m.create_app(
        FakeEngine("base"),
        {"base": None, "candidate": "/private", "rejected": "/private"},
        tmp_path / "events.jsonl",
        activate=activate,
    )
    client = TestClient(app)
    rejected = client.post("/v1/development-version", json={"version": "rejected"})
    assert rejected.status_code == 500
    assert client.get("/ready").json()["version"] == "base"
    switched = client.post(
        "/v1/development-version", json={"version": "candidate", "rollback": False}
    )
    assert switched.status_code == 200
    assert client.get("/ready").json()["version"] == "candidate"
    restored = client.post("/v1/development-version", json={"version": "base", "rollback": True})
    assert restored.status_code == 200
    assert client.get("/ready").json()["version"] == "base"
    assert calls == [("rejected", False), ("candidate", False), ("base", True)]
