# VLA-Forge M04 — evidence report

**Status: INCOMPLETE.** This report is rebuilt from completed cloud-side runner receipts; it is not a claim that the requested scale target or publication gate passed.

## Real policy attempts
- Completed attempt rows: 79 (12 Success rows, 67 failed rows; all retained).
- Non-episode engineering errors retained separately: 2; operator-interrupted pre-policy runs: 2 (not counted as episodes).
- Artifact hash audit: 237 references, 0 missing, 1 registry hash exception(s); affected action values independently match the paired NPZ exactly. See `artifact_integrity.json`.
- Unique successful action hashes in policy registry: 11; success rows retain full-trajectory hashes; all corrected source action hashes are separately deduplicated and checked by independent replay.
- Independently verified unique correction sequences: 12 (one M03-derived physical replay plus 11 fresh M04 SmolVLA inference sequences).
- M04 RTX 3090 allocated runtime over all fresh policy attempts: 0.573466 GPU-hours; peak allocated memory: 0.898 GiB.
- Episode-level median policy-call latency percentiles (ms): {'p50': 1.8661725043784827, 'p95': 3.0353291920619068, 'p99': 324.7058149299119}; chunk medians p50=322.7 ms, p95=964.8 ms, worst observed per-episode p99=1245.4 ms.
- LeRobotDataset v3 writer: 36.478s; throughput receipt reports 26.21 transitions/s.
- Independent LeRobot v3 audit: 12 episodes / 956 transitions; action/state Linf=0.0/0.0; mean camera error=2.190.
- Demo: `/data/VLA-Forge/Evidence/M04/demo/M04_demo_200s.mp4` (227.93s, 43383032 bytes, 1920x1080 h264 yuv420p)

## Verified LeRobotDataset v3 candidate
`/data/VLA-Forge/Evidence/M04/dataset/lerobot_v3_verified_r7` contains 12 independently audited episodes / 956 transitions: 11 fresh M04 policy corrections plus one M03-derived physical replay. Source provenance spans task ids [6] and M02 init states [6, 7]; the source-limited development dataset does not support cross-task generalization claims.

## Performance decision
PASS: FP32 CUDA Graph warm chunk speedup 6.947x; 18 paired complete chunks; exact matched MuJoCo continuation. Cold capture 1.816s is separate; no end-to-end speedup claimed. BF16 and TF32 remain rejected. GPU-hours cover registered policy-loop windows only, not billed or total production time.

## Public release gate
Private delivery: [{"repo_id": "jlai300/vla-forge-m04-corrections-dev", "url": "https://huggingface.co/datasets/jlai300/vla-forge-m04-corrections-dev", "private": true, "hash_validation": "PASS"}, {"repo_id": "jlai300/vla-forge-m04-evidence", "url": "https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence", "private": true, "hash_validation": "PASS"}]; PUBLIC RELEASE WITHHELD pending asset rights review.

## Alternate-source search
Bounded searches over M02 Goal task 6 alternate init states [0, 1, 2, 4, 6] recorded 12 failed policy attempts and 1 additional verified correction(s); 1 interrupted render(s) are outside the episode denominator. The verified set now spans init states 6 and 7, but remains one task.

## Remaining acceptance gaps
- The 10-sequence count target is met by fresh, unique, independently replayed policy outputs. Verified source states now include M02 init states [6, 7]; the set still covers only Goal task 6.
- Cross-task coverage remains one LIBERO task; no task-generalization claim is supported.
- PASS: FP32 CUDA Graph warm chunk speedup 6.947x; 18 paired complete chunks; exact matched MuJoCo continuation. Cold capture 1.816s is separate; no end-to-end speedup claimed.
- Public Dataset/Space release awaits exact asset redistribution clearance; a one-task private development dataset is a scope-limited deliverable.
- Demo passed codec/length/full-decode validation at 227.93s with 37 unique real episode videos; offline viewer is `demo/visualization/index.html`.

## Evidence files
- `manifest.json`, `episodes.csv`, `failure_cases.jsonl`, `metrics.json`, `status.json`.
- Raw episode NPZ/JSON/MP4 files and independent replay receipts under `artifacts/`.
- LeRobot v3 provenance and independent frame/action/state audit under `dataset/`.
- Both accepted CUDA Graph and rejected reduced-precision receipts under `performance/`.
