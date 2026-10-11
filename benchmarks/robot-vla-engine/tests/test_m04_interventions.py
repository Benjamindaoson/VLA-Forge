import numpy as np
import pytest

from robot_vla.m04_interventions import (
    benchmark_suite_name,
    intervention_is_active,
    m02_source_episode_name,
    scale_environment_action,
)


def test_unit_scale_preserves_policy_command():
    action = np.array([0.2, -0.4, 0.8, 1.0], dtype=np.float32)
    result = scale_environment_action(action, 1.0)
    np.testing.assert_array_equal(result, action)


def test_scaled_command_is_clipped_only_after_scale():
    action = np.array([0.2, -0.4, 0.8, 1.0], dtype=np.float32)
    result = scale_environment_action(action, 1.5)
    np.testing.assert_allclose(result, [0.3, -0.6, 1.0, 1.0])


@pytest.mark.parametrize("scale", [0, -0.1, 2.01, float("nan")])
def test_rejects_unsafe_scale(scale):
    with pytest.raises(ValueError):
        scale_environment_action(np.zeros(4), scale)


def test_observation_intervention_window_uses_half_open_step_interval():
    assert intervention_is_active(50, 50, 50)
    assert intervention_is_active(99, 50, 50)
    assert not intervention_is_active(49, 50, 50)
    assert not intervention_is_active(100, 50, 50)


def test_m02_source_keys_map_to_official_benchmark_names():
    assert benchmark_suite_name("long") == "libero_10"
    assert m02_source_episode_name("goal", 6, 7) == "goal_task06_episode07_attempt1"
    with pytest.raises(ValueError):
        benchmark_suite_name("unknown")
