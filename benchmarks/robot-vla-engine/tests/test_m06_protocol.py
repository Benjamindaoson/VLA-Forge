"""M06 must fail closed before any held-out rollout."""

import copy

import pytest

from robot_vla.m06_protocol import EXCLUDED_GROUPS, validate_freeze_structure


def frozen_rows():
    rows = []
    for suite in ("spatial", "object", "goal", "long"):
        for task in range(10):
            for init in range(40, 50):
                rows.append({"suite": suite, "task_id": task, "init_state_id": init,
                             "source_group": f"{suite}:{task}:{init}", "seed": 95000 + len(rows)})
    return {"status": "FROZEN_NOT_EXECUTED", "rows": rows,
            "development_group_overlap": [], "old_M02_state_bytes_overlap": []}


def test_exact_400_case_contract():
    summary = validate_freeze_structure(frozen_rows())
    assert summary["rows"] == 400
    assert summary["per_suite"] == {name: 100 for name in ("spatial", "object", "goal", "long")}
    assert len(EXCLUDED_GROUPS) == 6


@pytest.mark.parametrize("mutate", [
    lambda d: d["rows"].pop(),
    lambda d: d["rows"].__setitem__(0, copy.deepcopy(d["rows"][1])),
    lambda d: d["rows"][0].__setitem__("init_state_id", 7),
    lambda d: d.__setitem__("development_group_overlap", ["goal:6:40"]),
    lambda d: d["rows"][0].__setitem__("source_group", "goal:6:6"),
])
def test_fail_closed_freeze(mutate):
    d = frozen_rows()
    mutate(d)
    with pytest.raises(ValueError):
        validate_freeze_structure(d)
