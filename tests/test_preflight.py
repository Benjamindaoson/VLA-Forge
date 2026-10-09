from vla_forge.preflight import collect_report, main


def test_preflight_does_not_claim_real_robot_results(tmp_path):
    report = collect_report(tmp_path)
    assert report["report_kind"] == "read_only_environment_preflight"
    assert report["real_libero_executed"] is False
    assert report["policy_trained"] is False
    assert report["free_disk_gb"] >= 0
    assert "cuda_available" in report


def test_preflight_can_emit_without_hardware(capsys, tmp_path):
    code = main(["--data-root", str(tmp_path)])
    result = capsys.readouterr().out
    assert code == 0
    assert '"packages"' in result
