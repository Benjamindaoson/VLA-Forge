import pytest

from vla_forge.interventions import RolloutResult, run_paired_intervention
from vla_forge.models import InterventionFactor, InterventionSpec
from vla_forge.runtime import ActionChunkScheduler, ActionRejected, SchedulerConfig


class FakeSimulator:
    """A contract-only double; not LIBERO or evidence of trained performance."""

    def __init__(self, *, unstable=False):
        self.state = None
        self.restores = 0
        self.changed = False
        self.unstable = unstable

    def restore(self, snapshot_ref):
        self.state = snapshot_ref
        self.restores += 1
        self.changed = False

    def state_fingerprint(self):
        return self.state + (str(self.restores) if self.unstable else "")

    def apply_intervention(self, intervention):
        self.changed = True

    def rollout(self, *, checkpoint, seed):
        return RolloutResult(succeeded=self.changed, steps=10, evidence_uri="synthetic://trace")


def spec():
    return InterventionSpec(
        intervention_id="i1", factor=InterventionFactor.ACTION_HORIZON,
        value="5", kind="task_preserving", task_goal_preserved=True,
        environment_version="libero-v1", snapshot_ref="snap",
    )


def test_paired_snapshot_restored_twice():
    fake = FakeSimulator()
    result = run_paired_intervention(
        fake, snapshot_ref="snap", checkpoint="base", seed=123, intervention=spec()
    )
    assert result.success_delta == 1
    assert fake.restores == 2


def test_restore_fingerprint_mismatch_fails():
    with pytest.raises(RuntimeError, match="Replay mismatch"):
        run_paired_intervention(
            FakeSimulator(unstable=True),
            snapshot_ref="snap", checkpoint="base", seed=1, intervention=spec(),
        )


def test_wrong_snapshot_is_rejected():
    with pytest.raises(ValueError, match="snapshot"):
        run_paired_intervention(
            FakeSimulator(), snapshot_ref="other",
            checkpoint="base", seed=1, intervention=spec(),
        )


def test_runtime_chunk_queues_actions():
    scheduler = ActionChunkScheduler(SchedulerConfig(action_dimensions=2))
    scheduler.submit([[0.1, 0.2], [0.3, 0.4]], observation_timestamp=1, now_seconds=1.01)
    assert scheduler.next_action(now_seconds=1.02) == (0.1, 0.2)
    assert scheduler.next_action(now_seconds=1.03) == (0.3, 0.4)
    with pytest.raises(ActionRejected, match="exhausted"):
        scheduler.next_action(now_seconds=1.04)


def test_runtime_expires_old_observation():
    scheduler = ActionChunkScheduler(SchedulerConfig(action_dimensions=2))
    scheduler.submit([[0.0, 0.0]], observation_timestamp=1, now_seconds=1.01)
    with pytest.raises(ActionRejected, match="expired"):
        scheduler.next_action(now_seconds=1.5)
    assert scheduler.remaining == 0


def test_invalid_chunk_does_not_replace_previous_valid_chunk():
    scheduler = ActionChunkScheduler(SchedulerConfig(action_dimensions=2))
    scheduler.submit([[0.0, 0.0]], observation_timestamp=1, now_seconds=1.01)
    with pytest.raises(ActionRejected, match="Non-finite"):
        scheduler.submit([[float("nan"), 0]], observation_timestamp=1, now_seconds=1.02)
    assert scheduler.remaining == 1


def test_dimension_and_magnitude_check():
    scheduler = ActionChunkScheduler(SchedulerConfig(action_dimensions=2))
    with pytest.raises(ActionRejected, match="dimension"):
        scheduler.submit([[0.0]], observation_timestamp=1, now_seconds=1)
    with pytest.raises(ActionRejected, match="out-of-bounds"):
        scheduler.submit([[2.0, 0.0]], observation_timestamp=1, now_seconds=1)
