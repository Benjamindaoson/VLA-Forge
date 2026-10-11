# M04 focused closeout �� 2026-10-11

Status: core correction/data/FP32 inference engineering verified; full public-release status INCOMPLETE.

- 11 fresh unique policy corrections, each independently physically replayed twice. One additional M03-derived sequence remains separately labeled. No new seed-search sweep was started in this closeout.
- LeRobot v3 r7: 12 episodes / 956 transitions, exact action/state roundtrip. All data are development only, grouped by original failed source; two deliberate cross-split collisions were rejected. One task/two initial states does not establish generalization.
- CUDA Graph uses the frozen FP32 model and unchanged denoising schedule. Dynamic vision remains eager; attention/denoising launches are captured. All 18 full-chunk pairs (three real observations, six noise seeds) have max absolute error zero and identical RNG consumption. P50 391.743 -> 56.386 ms, P95 398.837 -> 61.846 ms, P99 399.072 -> 61.871 ms. Warm speedup 6.947x.
- Cold graph capture: 1.816 s. Approximate microbenchmark break-even is six full action chunks in one retained sampler. Eager remains default for isolated short jobs.
- Real matched MuJoCo continuation: 21 new actions, both baseline and graph Success=True; actions, qpos, qvel, end-effector pose and both cameras are identical. Current cold candidate policy loop 6.965 s versus historical baseline 5.558 s, so no end-to-end speed improvement is claimed. The comparison is not a contemporaneous runtime benchmark or success-rate trial.
- 82 tests passed, one upstream Starlette warning; Ruff and compilation passed. Model hash and dependency versions are retained in manifest/performance/dependency_versions.json receipts.
- Demo: 227.934 seconds, 1920x1080, 30 FPS, H.264/yuv420p; full decode and 37 source hashes verified. File size 43,383,032 bytes.
- Private Hugging Face Dataset and static Space were uploaded; 11 and 82 file hashes verified. Private Space runtime RUNNING. No paid hardware was provisioned. Dataset/Space are not a public release.
- Cloud files are streamed through authenticated SSH; local storage was not used for project assets. Direct cloud viewer is available via SSH forwarding on localhost:18765; HF links remain accessible to the account after login.

## Evidence and commands
Canonical directory: /data/VLA-Forge/Evidence/M04
Source project: /data/VLA-Forge/project/robot-vla
Source commit for validated algorithm: 1836156f8f4ef65014237fed5c6a66f00d23d076
Run `bash /data/VLA-Forge/Evidence/M04/reproduce.sh audit` or `benchmark`; `rollout` requires a new unique M04_RUN_ID and records its output separately from the correction-production registry.

## Remaining boundaries
Public asset redistribution clearance is unresolved; upstream asset repository has no dataset card/license declaration. One historical failed-attempt JSON hash exception is preserved rather than repaired. The registered 0.573466 GPU-hours exclude full setup, replay, encoding and idle costs; do not equate them with the cloud invoice. The 30�C50 example stretch target was not reached, and no training/generalization gain is claimed. M05 was not started.
