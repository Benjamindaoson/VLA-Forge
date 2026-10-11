"""Freeze unopened replacement M06 test states, model identities, and release rules."""
from __future__ import annotations

import json
import subprocess

from robot_vla.m06_protocol import (
    ADMISSION,
    ENGINE,
    EVIDENCE,
    FREEZE,
    M05,
    ROOT,
    atomic_json,
    frozen_rules,
    load_adapter_identities,
    sha256,
    validate_freeze_structure,
    validate_source_bytes,
)


def build() -> dict:
    source = json.loads(FREEZE.read_text())
    structure = validate_freeze_structure(source)
    bytes_check = validate_source_bytes(source['rows'])
    admission = json.loads(ADMISSION.read_text())
    recorded_overlap = set(admission['prior_test_contamination_groups']) & {r['source_group'] for r in source['rows']}
    if recorded_overlap:
        raise ValueError(f'Contaminated source groups in M06: {recorded_overlap}')
    lineage_path = M05 / 'data/lineage_gate.json'
    lineage = json.loads(lineage_path.read_text())
    if not lineage['passed_recorded_identity_checks']:
        raise ValueError('M05 recorded lineage failed')
    protocol_path = ENGINE / 'configs/data_protocol_v1.json'
    m05_path = ROOT / 'configs/m05/formal_protocol.json'
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    return {
        'schema': 'vla-forge.m06.frozen.v1', 'status': 'FROZEN_NOT_EXECUTED',
        'source_freeze': {'path': str(FREEZE), 'sha256': sha256(FREEZE), 'structure': structure, 'source_bytes': bytes_check},
        'source_admission': {'path': str(ADMISSION), 'sha256': sha256(ADMISSION), 'contamination_overlap': sorted(recorded_overlap)},
        'm05_lineage': {'path': str(lineage_path), 'sha256': sha256(lineage_path), 'expert_physical_mapping_known': lineage['expert_to_libero_init_mapping_known']},
        'data_protocol': {'path': str(protocol_path), 'sha256': sha256(protocol_path)},
        'm05_training_protocol': {'path': str(m05_path), 'sha256': sha256(m05_path)},
        'source_commit': commit, 'model_identities': load_adapter_identities(),
        'rules': frozen_rules(), 'cases': source['rows'],
    }

def main() -> None:
    content = build()
    path = EVIDENCE / 'protocol_frozen.json'
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != content:
            raise ValueError('Existing M06 frozen protocol differs; refusing overwrite')
        print(f'unchanged {path}')
        return
    atomic_json(path, content)
    print(json.dumps({'path': str(path), 'sha256': sha256(path), 'cases': len(content['cases'])}))

if __name__ == '__main__':
    main()
