# M04 Verified Corrections and Efficient VLA Inference

## Intent
Turn simulator-observed failures into replay-verified correction records, persist only qualified corrections, and measure inference optimizations on the frozen SmolVLA/LIBERO workload. All M04 implementation and evidence live within /data/VLA-Forge on the authorized RTX 3090 host.

## Scope
- Reuse frozen M02/M03 policy, dataset, and replay assets without repeating the 400-episode baseline.
- Gate correction admission on source provenance, oracle success, matching initial state, and two independent successful replays.
- Measure workload-matched inference candidates with latency, memory, and outcome telemetry.
- Produce M04 evidence and a 180–240 s source-mapped demo from real experiment artifacts.
- Do not train a new policy or begin M05.

## Acceptance
No correction is called verified unless its full provenance and replay receipt pass. Negative or blocked results remain explicit. Inference speedups are reported only against an identical frozen workload with successful outcome checks.
