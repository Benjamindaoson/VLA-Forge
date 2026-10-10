"""Bounded correction execution. Contract-test producers are not scientific evidence."""

import json
import os
import time
from pathlib import Path

import numpy as np

from .schema import file_hash

FIELDS = ("sim", "observation", "success", "pixels_image", "pixels_image2")
SEEDS = (42, 43, 44)


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _command(value):
    value = np.asarray(value)
    if (
        value.shape != (7,)
        or value.dtype.kind not in "fiu"
        or not np.isfinite(value).all()
        or np.any((value < -1) | (value > 1))
    ):
        raise ValueError("invalid command: expected finite seven-vector within [-1,1]")
    return value.copy()


def _snapshot(value):
    if set(value) != set(FIELDS):
        raise ValueError("snapshot fields differ")
    if type(value["success"]) not in (bool, np.bool_):
        raise ValueError("success must be an observed boolean predicate")
    result = {k: np.asarray(v).copy() for k, v in value.items()}
    for k, v in result.items():
        if not v.size or v.dtype.kind not in "buif" or not np.isfinite(v).all():
            raise ValueError("invalid snapshot: " + k)
    if result["observation"].shape != (8,) or result["sim"].ndim != 1:
        raise ValueError("state shape mismatch")
    for k in ("pixels_image", "pixels_image2"):
        v = result[k]
        if v.dtype != np.uint8 or v.ndim != 3 or v.shape[-1] != 3:
            raise ValueError("camera must be uint8 RGB")
    return result


def _same(a, b):
    return set(a) == set(b) and all(
        np.asarray(a[k]).dtype == np.asarray(b[k]).dtype and np.array_equal(a[k], b[k]) for k in a
    )


def _arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k].copy() for k in archive.files}


class BudgetExpired(Exception):
    pass


def _check_time(deadline):
    if time.monotonic() >= deadline:
        raise BudgetExpired("wall-time budget exhausted")


def _execute(case, factory, producer, directory, deadline, *, phase, seed=None, replay=None):
    """One full reset. Partial files are retained even if observation capture fails."""
    directory.mkdir()
    start = time.monotonic()
    row = dict(
        status="RUNNING",
        phase=phase,
        seed=seed,
        steps=0,
        success=None,
        relative_directory=directory.name,
        actuator_feedback=None,
    )
    _json(directory / "record.json", row)
    scene, observations, commands, raw_predictions = None, [], [], []
    prefix_observations = []
    log = (directory / "events.jsonl").open("x", encoding="utf-8")

    def event(event, **data):
        log.write(
            json.dumps(dict(event=event, elapsed=time.monotonic() - start, **data), allow_nan=False)
            + "\n"
        )
        log.flush()
        os.fsync(log.fileno())

    def step(command, step_phase):
        _check_time(deadline)
        command = _command(command)
        event("command_dispatched", phase=step_phase, command=command.tolist())
        observed, terminal, truncated = scene.step(command)
        if type(terminal) not in (bool, np.bool_) or type(truncated) not in (bool, np.bool_):
            raise ValueError("terminal flags must be boolean")
        observed = _snapshot(observed)
        if any(
            observed[k].shape != prefix_observations[0][k].shape
            or observed[k].dtype != prefix_observations[0][k].dtype
            for k in FIELDS
        ):
            raise ValueError("snapshot shape or dtype changed within episode")
        return observed, bool(terminal or truncated)

    try:
        _check_time(deadline)
        scene = factory()
        current = _snapshot(scene.reset(case["environment_seed"]))
        prefix_observations.append(current)
        _check_time(deadline)
        for command in case["prefix_commands"]:
            if bool(current["success"]):
                raise ValueError("success before end of prefix")
            current, done = step(command, "prefix")
            prefix_observations.append(current)
            _check_time(deadline)
            if done:
                raise ValueError("terminal prefix cannot be extended")
        observations.append(current)
        prefix_values = {k: np.stack([s[k] for s in prefix_observations]) for k in FIELDS}
        prefix_matches = "prefix_reference" not in case or _same(
            prefix_values, case["prefix_reference"]
        )
        if not _same(current, case["reference"]) or not prefix_matches:
            row["status"] = "RECONSTRUCTION_MISMATCH"
        elif bool(current["success"]):
            row.update(status="NOT_APPLICABLE", success=True)
        elif phase == "qualification":
            row.update(status="QUALIFIED", success=False)
        else:
            if replay is None:
                producer.reset(seed)
            remaining = case["horizon"] - len(case["prefix_commands"])
            limit = remaining if replay is None else len(replay)
            if not 0 < limit <= remaining:
                raise ValueError("invalid correction horizon")
            for index in range(limit):
                _check_time(deadline)
                if replay is None:
                    raw, command = producer.predict(current)
                    _check_time(deadline)
                    raw = np.asarray(raw)
                    if (
                        raw.shape != (7,)
                        or raw.dtype.kind not in "fiu"
                        or not np.isfinite(raw).all()
                    ):
                        raise ValueError("invalid raw prediction")
                else:
                    raw, command = None, replay[index]
                command = _command(command)
                # Record dispatched command even when env.step/render fails afterward.
                commands.append(command)
                if raw is not None:
                    raw_predictions.append(raw.copy())
                current, done = step(command, "correction")
                observations.append(current)
                row["steps"] += 1
                event("step_observed", index=index, success=bool(current["success"]))
                _check_time(deadline)
                if bool(current["success"]):
                    row.update(status="SUCCESS", success=True)
                    break
                if done:
                    row.update(status="TERMINATED", success=False)
                    break
            else:
                row.update(status="HORIZON", success=False)
    except BudgetExpired as exc:
        row.update(status="BUDGET", success=None, error=str(exc))
    except Exception as exc:
        row.update(status="ERROR", success=None, error=f"{type(exc).__name__}: {exc}")
    finally:
        if scene is not None:
            try:
                scene.close()
            except Exception as exc:
                row.update(status="ERROR", success=None, close_error=str(exc))
        row["seconds"] = time.monotonic() - start
        if time.monotonic() >= deadline and row["status"] != "ERROR":
            row.update(status="BUDGET", success=None, error="wall-time budget exhausted")
        data = dict(
            commands=np.asarray(commands).reshape(-1, 7),
            raw_predictions=np.asarray(raw_predictions).reshape(-1, 7),
        )
        for label, samples in (("", observations), ("prefix_", prefix_observations)):
            if samples:
                data.update({label + k: np.stack([s[k] for s in samples]) for k in FIELDS})
        np.savez_compressed(directory / "trajectory.npz", **data)
        event("execution_finished", status=row["status"], steps=row["steps"])
        log.close()
        _json(directory / "record.json", row)
    return row


def collect_case(case, scene_factory, producer, directory, *, deadline):
    """Acquire at most one correction. Exactness is local and scoped to saved fields."""
    directory = Path(directory)
    commands = np.asarray(case["prefix_commands"])
    if commands.ndim != 2 or commands.shape[1:] != (7,) or not len(commands):
        raise ValueError("invalid prefix commands")
    for command in commands:
        _command(command)
    if type(case["horizon"]) is not int or case["horizon"] <= len(commands):
        raise ValueError("no remaining horizon")
    if type(case["environment_seed"]) is not int:
        raise ValueError("invalid environment seed")
    reference = _snapshot(case["reference"])
    case = dict(case, reference=reference)
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory / "input.npz", prefix_commands=commands, **reference)
    result = dict(
        schema_version=1,
        case_id=case["case_id"],
        status="RUNNING",
        horizon=case["horizon"],
        prefix_steps=len(commands),
        environment_seed=case["environment_seed"],
        verified_correction=False,
        success_rate=None,
        actuator_feedback=None,
        qualifications=[],
        attempts=[],
        confirmations=[],
        scope="development case; exact saved states and cameras",
    )
    _json(directory / "summary.json", result)

    def execute(name, **kwargs):
        return _execute(case, scene_factory, producer, directory / name, deadline, **kwargs)

    for repeat in range(2):
        if time.monotonic() >= deadline:
            result["status"] = "INCOMPLETE_BUDGET"
            break
        row = execute(f"qualification_{repeat}", phase="qualification")
        result["qualifications"].append(row)
        if row["status"] == "NOT_APPLICABLE":
            result["status"] = "NOT_APPLICABLE"
            break
        if row["status"] != "QUALIFIED":
            result["status"] = (
                "INCOMPLETE_BUDGET" if row["status"] == "BUDGET" else "BLOCKED_RECONSTRUCTION"
            )
            break
        if repeat == 0:
            archive = _arrays(directory / "qualification_0/trajectory.npz")
            case["prefix_reference"] = {k: archive["prefix_" + k] for k in FIELDS}
    else:
        result["status"] = "NO_VERIFIED_CORRECTION"
        for seed in SEEDS:
            if time.monotonic() >= deadline:
                result["status"] = "INCOMPLETE_BUDGET"
                break
            row = execute(f"attempt_{seed}", phase="attempt", seed=seed)
            result["attempts"].append(row)
            if row["status"] == "BUDGET":
                result["status"] = "INCOMPLETE_BUDGET"
                break
            if row["status"] not in ("SUCCESS", "HORIZON", "TERMINATED"):
                result["status"] = "INCOMPLETE_ERROR"
                break
            if row["status"] == "SUCCESS":
                original = _arrays(directory / f"attempt_{seed}/trajectory.npz")
                result["status"] = "UNVERIFIED_CONFIRMATION"
                for repeat in range(2):
                    if time.monotonic() >= deadline:
                        result["status"] = "INCOMPLETE_BUDGET"
                        break
                    confirmation = execute(
                        f"confirmation_{repeat}", phase="confirmation", replay=original["commands"]
                    )
                    actual = _arrays(directory / f"confirmation_{repeat}/trajectory.npz")
                    keys = ["commands", *FIELDS, *["prefix_" + k for k in FIELDS]]
                    confirmation["exact"] = confirmation["status"] == "SUCCESS" and all(
                        k in actual
                        and np.array_equal(original[k], actual[k])
                        and original[k].dtype == actual[k].dtype
                        for k in keys
                    )
                    result["confirmations"].append(confirmation)
                    if confirmation["status"] == "BUDGET":
                        result["status"] = "INCOMPLETE_BUDGET"
                        break
                    if not confirmation["exact"]:
                        break
                if len(result["confirmations"]) == 2 and all(
                    r["exact"] for r in result["confirmations"]
                ):
                    if time.monotonic() >= deadline:
                        result["status"] = "INCOMPLETE_BUDGET"
                    else:
                        result.update(status="VERIFIED", verified_correction=True)
                break  # never hunt another seed after a non-reproducible success

    tried = {r["seed"] for r in result["attempts"]}
    reason = (
        "NOT_RUN_BUDGET"
        if result["status"] == "INCOMPLETE_BUDGET"
        else "NOT_RUN_AFTER_ERROR"
        if result["status"] == "INCOMPLETE_ERROR"
        else "NOT_RUN_FIRST_SUCCESS"
        if any(r["status"] == "SUCCESS" for r in result["attempts"])
        else "NOT_RUN_PRECONDITION"
    )
    for seed in SEEDS:
        if seed not in tried:
            result["attempts"].append(dict(seed=seed, status=reason, steps=0, success=None))
    valid = [r for r in result["attempts"] if r["status"] in ("SUCCESS", "HORIZON", "TERMINATED")]
    result["valid_attempts"] = len(valid)
    result["queue_stop"] = result["status"] in (
        "INCOMPLETE_BUDGET",
        "INCOMPLETE_ERROR",
        "BLOCKED_RECONSTRUCTION",
        "UNVERIFIED_CONFIRMATION",
    )
    result["attempt_counts"] = dict(
        planned=3,
        executed=len(tried),
        valid=len(valid),
        errors=sum(r["status"] == "ERROR" for r in result["attempts"]),
        not_run=3 - len(tried),
    )
    result["success_rate"] = (
        sum(r["success"] for r in valid) / len(valid)
        if valid and not result["queue_stop"]
        else None
    )
    result["files"] = {
        p.relative_to(directory).as_posix(): file_hash(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.name != "summary.json"
    }
    _json(directory / "summary.json", result)
    return result


def collect_cases(cases, scene_factory, producer_factory, directory, *, deadline):
    """Stop the whole queue after an execution error; retain explicit skipped cases."""
    cases = list(cases)
    if not 0 < len(cases) <= 4 or len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("expected one to four unique cases")
    stopped_by = None
    root = Path(directory).resolve()
    for case in cases:
        target = (root / case["case_id"]).resolve()
        if target.parent != root:
            raise ValueError("case identity must be one path component")
        if stopped_by is not None:
            target.mkdir(parents=True, exist_ok=False)
            result = dict(
                case_id=case["case_id"],
                status="NOT_RUN_AFTER_QUEUE_STOP",
                stopped_by=stopped_by,
                verified_correction=False,
                queue_stop=True,
                success_rate=None,
                actuator_feedback=None,
                attempts=[
                    dict(seed=s, status="NOT_RUN_AFTER_QUEUE_STOP", steps=0, success=None)
                    for s in SEEDS
                ],
            )
            _json(target / "summary.json", result)
        else:
            result = collect_case(
                case,
                lambda c=case: scene_factory(c),
                producer_factory(case),
                target,
                deadline=deadline,
            )
            if result["queue_stop"]:
                stopped_by = case["case_id"]
        yield case, result, target


def verify_record(directory):
    """Re-read bytes and replay equality; caller must also bind summary via Registry."""
    directory = Path(directory).resolve()
    result = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if result.get("status") != "VERIFIED" or result.get("verified_correction") is not True:
        raise ValueError("record is incomplete or unverified")
    files = result.get("files", {})
    if not files or "input.npz" not in files:
        raise ValueError("missing integrity manifest")
    for name, expected in files.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or file_hash(path) != expected:
            raise ValueError("record integrity mismatch: " + name)
    qualifications = result.get("qualifications", [])
    attempts = [r for r in result.get("attempts", []) if r["status"] == "SUCCESS"]
    confirmations = result.get("confirmations", [])
    if (
        len(qualifications) != 2
        or any(r["status"] != "QUALIFIED" for r in qualifications)
        or len(attempts) != 1
        or len(confirmations) != 2
    ):
        raise ValueError("incomplete qualification/confirmation")
    reference = _arrays(directory / "input.npz")
    execution_rows = [*qualifications, *attempts, *confirmations]
    identities = [r["relative_directory"] for r in execution_rows]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate execution identity")
    for phase, rows in [
        ("qualification", qualifications),
        ("attempt", attempts),
        ("confirmation", confirmations),
    ]:
        for index, row in enumerate(rows):
            expected_name = f"{phase}_{row['seed'] if phase == 'attempt' else index}"
            record_name = row["relative_directory"] + "/record.json"
            if (
                row["relative_directory"] != expected_name
                or row["phase"] != phase
                or record_name not in files
            ):
                raise ValueError("execution identity/phase mismatch")
            saved = json.loads((directory / record_name).read_text(encoding="utf-8"))
            if saved != {k: v for k, v in row.items() if k != "exact"}:
                raise ValueError("execution record differs from summary")

    def trajectory(row):
        name = row["relative_directory"] + "/trajectory.npz"
        if name not in files:
            raise ValueError("trajectory missing from integrity manifest")
        return _arrays(directory / name)

    qualified_prefix = {k: trajectory(qualifications[0])["prefix_" + k] for k in FIELDS}
    for row in execution_rows:
        values = trajectory(row)
        if not _same({k: values[k][0] for k in FIELDS}, {k: reference[k] for k in FIELDS}):
            raise ValueError("intervention reference differs")
        if not _same({k: values["prefix_" + k] for k in FIELDS}, qualified_prefix):
            raise ValueError("execution prefix trajectory differs")
    original = trajectory(attempts[0])
    n = len(original["commands"])
    if (
        not 0 < n <= result["horizon"] - result["prefix_steps"]
        or original["success"].shape != (n + 1,)
        or original["success"][0]
        or not original["success"][-1]
        or original["success"][1:-1].any()
        or original["raw_predictions"].shape != (n, 7)
    ):
        raise ValueError("invalid success sequence")
    for command in original["commands"]:
        _command(command)
    for row in confirmations:
        replay = trajectory(row)
        keys = ["commands", *FIELDS, *["prefix_" + k for k in FIELDS]]
        if row["status"] != "SUCCESS" or not _same(
            {k: original[k] for k in keys}, {k: replay[k] for k in keys}
        ):
            raise ValueError("confirmation differs")
    return result
