import numpy as np
import pytest

from robot_vla.m03 import action_delta, compare_sim_state, transform_observation


def test_state_comparison_distinguishes_exact_and_tolerant():
    a = {"qpos": [0, 1], "qvel": [0], "time": 2}
    b = {"qpos": [0, 1 + 1e-9], "qvel": [0], "time": 2}
    error = compare_sim_state(a, b, atol=1e-8)
    assert not error.exact and error.within_tolerance
    assert error.qpos_linf == pytest.approx(1e-9)


def test_shape_mismatch_cannot_pass_as_visual_or_physical_match():
    with pytest.raises(ValueError, match="shape differs"):
        compare_sim_state(
            {"qpos": [1], "qvel": [0], "time": 0}, {"qpos": [1, 2], "qvel": [0], "time": 0}
        )


def test_sham_is_byte_equal_but_does_not_mutate_source():
    obs = {
        "pixels": {
            "image": np.full((32, 32, 3), 80, np.uint8),
            "image2": np.full((32, 32, 3), 100, np.uint8),
        },
        "state": np.array([1.0, 2.0]),
    }
    sham = transform_observation(obs, "sham")
    assert np.array_equal(sham["pixels"]["image"], obs["pixels"]["image"])
    assert sham["pixels"]["image"] is not obs["pixels"]["image"]
    assert np.array_equal(obs["pixels"]["image"], np.full((32, 32, 3), 80, np.uint8))


def test_visual_treatments_change_only_declared_pixels():
    obs = {
        "pixels": {
            "image": np.full((32, 32, 3), 80, np.uint8),
            "image2": np.full((32, 32, 3), 100, np.uint8),
        }
    }
    bright = transform_observation(obs, "brightness")
    assert np.all(bright["pixels"]["image"] == 92)
    occluded = transform_observation(obs, "occlusion", camera="image2")
    assert np.array_equal(occluded["pixels"]["image"], obs["pixels"]["image"])
    assert np.all(occluded["pixels"]["image2"][14:18, 14:18] == 0)
    assert np.all(occluded["pixels"]["image2"][:8, :8] == 100)


def test_action_delta_reports_paired_behavior_difference():
    result = action_delta(np.array([[0.0, 1.0]]), np.array([[1.0, 1.5]]))
    assert result == {"mean_abs": 0.75, "linf": 1.0}
