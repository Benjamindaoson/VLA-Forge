import pytest
from fastapi.testclient import TestClient

from vla_forge.api import create_app
from vla_forge.models import FailureCase, RepairCase
from vla_forge.storage import CaseConflictError, CaseRepository


def sample_case():
    return RepairCase(
        failure=FailureCase(
            case_id="c1", task_id="task-1", policy_version="v0",
            source_episode_id="ep1", split_group_id="group1",
            environment_version="libero-pinned-v1",
        )
    )


def test_case_is_idempotent_but_immutable(tmp_path):
    repo = CaseRepository(tmp_path / "database.sqlite")
    before = sample_case()
    digest = repo.insert_case(before)
    assert repo.insert_case(before) == digest
    assert repo.get_case("c1").failure.task_id == "task-1"
    after = sample_case()
    after.failure.task_id = "changed"
    with pytest.raises(CaseConflictError, match="overwritten"):
        repo.insert_case(after)


def test_audit_events_cannot_be_rewritten(tmp_path):
    repo = CaseRepository(tmp_path / "database.sqlite")
    repo.insert_case(sample_case())
    digest = repo.append_event("e1", "c1", "diagnostic", {"result": "pass"})
    assert digest == repo.append_event("e1", "c1", "diagnostic", {"result": "pass"})
    with pytest.raises(CaseConflictError, match="immutable"):
        repo.append_event("e1", "c1", "diagnostic", {"result": "fail"})
    assert repo.list_events("c1")[0]["payload"] == {"result": "pass"}


def test_api_case_ingestion_and_event_listing(tmp_path):
    client = TestClient(create_app(tmp_path / "api.sqlite"))
    assert client.get("/health").json()["status"] == "ok"
    payload = sample_case().model_dump(mode="json")
    response = client.post("/v1/cases", json=payload)
    assert response.status_code == 201
    assert len(response.json()["sha256"]) == 64
    assert client.get("/v1/cases/c1").json()["failure"]["task_id"] == "task-1"
    assert len(client.get("/v1/cases").json()) == 1
    assert client.get("/v1/cases/missing").status_code == 404
    assert client.post(
        "/v1/cases/c1/events",
        json={"event_id": "e1", "kind": "probe", "payload": {"outcome": "pass"}},
    ).status_code == 201
    assert client.get("/v1/cases/c1/events").json()[0]["payload"] == {"outcome": "pass"}


def test_api_conflict_code(tmp_path):
    client = TestClient(create_app(tmp_path / "api.sqlite"))
    first = sample_case().model_dump(mode="json")
    assert client.post("/v1/cases", json=first).status_code == 201
    first["failure"]["task_id"] = "changed"
    assert client.post("/v1/cases", json=first).status_code == 409


def test_api_router_has_no_hidden_training_action(tmp_path):
    client = TestClient(create_app(tmp_path / "api.sqlite"))
    request = {
        "belief": {"probabilities": {"timing": 1.0}},
        "options": [
            {
                "option_id": "adjust_horizon", "kind": "runtime_fix", "cost": 0,
                "safety_risk": 0, "regression_risk": 0,
                "estimated_gain_by_hypothesis": {"timing": 0.25},
            }
        ],
        "settings": {"budget": 1},
    }
    answer = client.post("/v1/diagnosis/recommend", json=request)
    assert answer.status_code == 200
    assert answer.json()["choice_id"] == "adjust_horizon"
