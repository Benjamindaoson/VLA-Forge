import json

import pytest

from vla_forge.adapters.lerobot_commands import (
    LiberoEvalConfig, SmolVLATrainConfig, libero_eval_command, smolvla_sft_command,
)
from vla_forge.cli import main


def test_eval_command_pins_protocol():
    command = libero_eval_command(
        LiberoEvalConfig(
            policy_path="lerobot/smolvla_base", output_dir="outputs",
            episodes_per_task=10, seed=123, action_steps=10,
        )
    )
    assert "--env.hard_reset=true" in command
    assert "--env.init_states=true" in command
    assert "--eval.batch_size=10" in command
    assert "--policy.n_action_steps=10" in command


def test_bad_suite_rejected():
    with pytest.raises(ValueError, match="suite"):
        libero_eval_command(
            LiberoEvalConfig(policy_path="checkpoint", output_dir="out", suite="unknown")
        )


def test_normal_sft_command_does_not_claim_weighted_training():
    cmd = smolvla_sft_command(
        SmolVLATrainConfig(
            dataset_repo_id="org/verified", checkpoint="org/base",
            output_dir="output", steps=100, batch_size=2,
        )
    )
    assert "--policy.path=org/base" in cmd
    assert "--policy.push_to_hub=false" in cmd
    assert not any("weighted" in arg for arg in cmd)


def test_cli_example_explicitly_synthetic(capsys):
    assert main(["demo-planner"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["data_kind"] == "synthetic_illustration"
    assert result["recommendation"]["mode"] in {"diagnose", "repair"}


def test_cli_eval_command(capsys):
    assert main(["eval-command", "--policy", "base", "--output-dir", "out"]) == 0
    assert "lerobot-eval" in capsys.readouterr().out


def test_weighted_flow_matching_loss_has_gradient():
    torch = pytest.importorskip("torch")
    from vla_forge.training import weighted_flow_matching_loss

    p = torch.tensor([[[1.0]], [[3.0]]], requires_grad=True)
    value = weighted_flow_matching_loss(p, torch.zeros_like(p), torch.tensor([1.0, 3.0]))
    assert value.item() == pytest.approx(7.0)
    value.backward()
    assert p.grad is not None


def test_weighted_flow_matching_ignores_padding():
    torch = pytest.importorskip("torch")
    from vla_forge.training import weighted_flow_matching_loss

    p = torch.tensor([[[2.0], [100.0]]])
    value = weighted_flow_matching_loss(
        p, torch.zeros_like(p), torch.ones(1),
        valid_action_mask=torch.tensor([[True, False]]),
    )
    assert value.item() == pytest.approx(4.0)


def test_weighted_flow_matching_disallows_negative_weights():
    torch = pytest.importorskip("torch")
    from vla_forge.training import weighted_flow_matching_loss

    with pytest.raises(ValueError, match="nonnegative"):
        weighted_flow_matching_loss(
            torch.zeros(2, 2, 1), torch.ones(2, 2, 1), torch.tensor([1.0, -1.0])
        )
