# M06 current execution state

Formal scientific protocol `/data/VLA-Forge/Evidence/M06/protocol_frozen.json` was frozen before the first independent test episode. The source freeze has 40 tasks × init IDs 40–49, excludes all six retired contaminated source groups, verifies all 400 official init-state bytes and BDDL hashes, and binds four M05 registry checkpoint hashes. Foundation-model pretraining overlap and LIBERO expert-episode physical mapping are unknown.

The single RTX 3090 formal worker is running under `/data/VLA-Forge/Evidence/M05/gpu.lock`; PID and machine-readable status are in `worker.pid` and `worker_status.json`. `formal/progress.json` is the current count; `formal/attempts.jsonl` retains starts, errors and completions. One completed formal episode was independently checked for 81 H.264 frames, 81 7D executed actions, 82 full available simulator states and exact reset-state match. Subsequent episodes persist individual receipts and hashes. No success-rate decision or model tuning is made from partial results.

A separate CPU watcher `watch_completion.sh` will, only after a zero-exit 1600-episode worker, validate every artifact, compute preregistered statistics, call ModelOps' independent gate, build a 180–240 second private real-footage demo, and generate `REPORT.md`. It records its state in `postprocess_status.json`; failure is not silently promoted to PASS.

VLA-Forge local branch `codex/m06-independent-release-eval` currently contains the corrected FP32 runner and validation through commit `3f7ed1a`. ModelOps local branch `codex/m06-policy-regression-gate` contains the FP32-bound independent release gate through `fafcd9e`. The initial native-BF16 technical attempt (30 receipts) is preserved under `technical_attempts/bf16_native_20261011/` and contributes zero to formal results. VLA M06 unit tests: 13 passed; ModelOps robotics unit tests: 10 passed. Four short real development-state rollouts passed initial-physics and video-decode checks. The authorized GitHub connector published the M06 source to independent branches: VLA-Forge commit `0872811c` and ModelOps commit `92d9ee61`. Local and publication commit IDs differ; the largest VLA runner and ModelOps adapter blob SHA-1 values were independently matched. Compact final results remain pending completion of the formal test.

The acceptance state is **IN PROGRESS**, not PASS. No M07 work or new training has started.

The private demo selector now accepts short, unique real rollout clips; its synthetic selection regression test exposed the former four-second cutoff and passes after repair. This only changes postprocessing, not the frozen simulator or policy protocol. As of 2026-10-11 16:28 UTC, 17/1600 corrected formal receipts were complete with zero recorded errors.

The finalizer reads Registry SQLite resources if an interrupted run lacks the exported events.jsonl; it fails closed if neither evidence source exists. This preserves resource accounting after a recoverable power loss.

On the live unexported Registry run, the SQLite fallback recovered 528 actual GPU/CPU resource samples; this was a read-only check and did not alter the running simulation.

The final cost report will separate measured episode GPU-hours, corrected-worker wall hours, non-episode overhead, and the quarantined BF16 trial; monetary rental cost remains unknown without an invoice.

Scope control: the latest user instruction authorizes M06 only. Older generic VLA-Forge M1–M7 heartbeat text must not be interpreted as permission to start M07, training, or duplicate GPU work. Wait for complete 1600/1600 evidence and perform only M06 acceptance and publication.

2026-10-11T17:15:44.185689+00:00: SSH reconnected after client power cycle. Verified original worker PID 887179, Python PID 887181, watcher PID 887182, GPU lock present, 78/1600 corrected formal receipts, zero errors. No new worker started.
