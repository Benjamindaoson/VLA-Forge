"""The demo must cover unique matched real sources for a full technical film."""
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/build_m06_demo.py'
spec = importlib.util.spec_from_file_location('build_m06_demo', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def test_selection_never_pads_with_repeated_episode():
    cases = [{'suite': suite, 'task_id': task, 'init_state_id': init}
             for suite in module.SUITES for task in range(10) for init in range(40, 50)]
    receipts = {}
    for case in cases:
        for condition in module.CONDITIONS:
            key = (condition, case['suite'], case['task_id'], case['init_state_id'])
            receipts[key] = {'steps': 260, 'success': condition == 'frozen'}
    chosen = module.select_cases({'cases': cases}, receipts)
    keys = {(item['case']['suite'], item['case']['task_id'], item['case']['init_state_id']) for item in chosen}
    assert len(keys) == len(chosen)
    assert 180 <= sum(item['duration'] for item in chosen) <= 240
    assert sum(item['case']['suite'] == 'goal' and item['case']['task_id'] == 6 for item in chosen) >= 3

def test_selection_accepts_short_unique_real_episodes():
    cases = [{'suite': suite, 'task_id': task, 'init_state_id': init}
             for suite in module.SUITES for task in range(10) for init in range(40, 50)]
    receipts = {(condition, case['suite'], case['task_id'], case['init_state_id']):
                {'steps': 60, 'success': condition == 'frozen'}
                for case in cases for condition in module.CONDITIONS}
    chosen = module.select_cases({'cases': cases}, receipts)
    ids = {(row['case']['suite'], row['case']['task_id'], row['case']['init_state_id'])
           for row in chosen}
    assert len(ids) == len(chosen)
    assert len(chosen) >= 18
    assert 180 <= sum(row['duration'] for row in chosen) <= 240
