"""Fail-closed M05 source-group and development init-state lineage check."""
from __future__ import annotations

import json
from pathlib import Path

from robot_vla.m05_training import EVIDENCE, atomic_json, sha256

PROTOCOL = Path('configs/m05/formal_protocol.json')
SELECTION = EVIDENCE / 'data/selection.json'


def main() -> None:
    protocol, selection = json.loads(PROTOCOL.read_text()), json.loads(SELECTION.read_text())
    dev = {(row['suite'], int(row['task']), int(row['init'])) for row in protocol['development']}
    forbidden = set(map(int, protocol['formal_test_forbidden_init_ids']))
    corrective = set(selection['corrective']['source_states'])
    corrective_inits = {int(row.split(':')[-1]) for row in corrective}
    expert = set(selection['standard']['source_groups'])
    replay = set(selection['replay']['source_groups'])
    checks = {
        'eight_unique_development_cases': len(dev) == 8,
        'development_init_disjoint_from_m06': not {row[2] for row in dev} & forbidden,
        'corrective_init_disjoint_from_development': not corrective_inits & {row[2] for row in dev},
        'corrective_init_disjoint_from_m06': not corrective_inits & forbidden,
        'standard_expert_groups_disjoint_from_replay_expert_groups': not expert & replay,
        'all_corrective_source_groups_goal6': all(row.startswith('goal:6:') for row in corrective),
    }
    row = {
        'scope': 'M05 data identity gate; no M06 formal evaluation was executed',
        'protocol_sha256': sha256(PROTOCOL), 'selection_sha256': sha256(SELECTION),
        'checks': checks, 'passed_recorded_identity_checks': all(checks.values()),
        'development_init_ids': sorted({row[2] for row in dev}),
        'corrective_source_init_ids': sorted(corrective_inits),
        'm06_forbidden_init_ids': sorted(forbidden),
        'standard_expert_source_groups': sorted(expert),
        'replay_expert_source_groups': sorted(replay),
        'expert_to_libero_init_mapping_known': False,
        'limitation': 'HF expert demonstrations have episode IDs, not verified LIBERO fixed init IDs. '
                      'The recorded source groups do not prove physical state disjointness from M06.',
    }
    atomic_json(EVIDENCE / 'data/lineage_gate.json', row)
    print(json.dumps({'passed_recorded_identity_checks': row['passed_recorded_identity_checks'],
                      'expert_to_libero_init_mapping_known': False, 'checks': checks}, indent=2))
    if not row['passed_recorded_identity_checks']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
