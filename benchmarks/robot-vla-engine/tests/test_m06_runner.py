"""Fail-closed physical pairing and durable receipt contracts."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_m06_independent.py'
spec = importlib.util.spec_from_file_location('run_m06_independent', SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

def sample(offset=0.0):
    return {'sim': np.array([1.0 + offset, 2.0]), 'qpos': np.array([3.0 + offset]),
            'qvel': np.array([0.0]), 'time': 0.5}

def test_physical_reference_is_write_once_and_rejects_mismatch(tmp_path):
    path = tmp_path / 'reference.npz'
    tolerance = {'sim_vector_max_abs': 1e-6, 'qpos_max_abs': 1e-6,
                 'qvel_max_abs': 1e-6, 'time_abs_s': 1e-9}
    first = runner.compare_reference(path, sample(), tolerance, create=True)
    assert first['matched'] and path.is_file()
    assert runner.compare_reference(path, sample(), tolerance, create=False)['max_abs_difference']['sim'] == 0
    with pytest.raises(ValueError, match='Unmatched initial physics'):
        runner.compare_reference(path, sample(0.01), tolerance, create=False)
    with pytest.raises(ValueError, match='Unmatched initial sim time'):
        runner.compare_reference(path, {**sample(), 'time': 0.6}, tolerance, create=False)
    with pytest.raises(ValueError, match='absent'):
        runner.compare_reference(tmp_path / 'missing.npz', sample(), tolerance, create=False)

def test_receipt_rejects_missing_or_changed_artifact(tmp_path):
    case = {'source_group': 'goal:6:40'}
    paths = {}
    for kind in ('video', 'trajectory', 'action_trace', 'sim_states'):
        p = tmp_path / kind
        p.write_bytes(kind.encode())
        paths[kind] = {'path': str(p), 'sha256': runner.sha256(p)}
    receipt = {'status': 'complete', 'condition': 'frozen', 'source_group': 'goal:6:40',
               'checkpoint_sha256': 'model', 'protocol_sha256': 'protocol',
               'floating_parameter_dtypes': ['torch.float32'], 'tf32_allowed': False,
               'success': False, 'initial_match': {'matched': True}, **paths}
    assert runner.valid_receipt(receipt, case, 'frozen', 'model', 'protocol')
    (tmp_path / 'video').write_bytes(b'changed')
    assert not runner.valid_receipt(receipt, case, 'frozen', 'model', 'protocol')
