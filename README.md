# VLA-Forge

**Failure recovery, replay-verified corrective data, and efficient VLA inference**

VLA-Forge is an auditable research-engineering project that converts failed Vision-Language-Action (VLA) manipulation attempts into provenance-tracked corrective trajectories, verifies those trajectories through independent simulator replay, and evaluates inference optimizations without changing action outputs.

## Research and engineering summary

**Research question:** Can failed manipulation episodes be converted into verified corrective experience that improves policy recovery?

The current implementation combines:

1. Failure provenance and source-trajectory tracking.
2. Controlled SmolVLA inference to generate candidate corrective actions.
3. MuJoCo replay with explicit success signals and bounded execution.
4. Independent replay verification before accepting a correction.
5. LeRobotDataset v3 export with action/state round-trip checks.
6. FP32 CUDA Graph inference optimization with paired correctness checks.

**Compute environment:** M04 GPU experiments used a rented NVIDIA RTX 4090 24GB instance.

## M04 verified results

<details>
<summary><strong>Experiment scope and results</strong></summary>

| Item | Verified result |
|---|---:|
| Registered policy-attempt rows | 79 |
| Successful attempt rows | 12 |
| Failed attempt rows | 67 |
| Fresh M04 corrective trajectories | 11 |
| Independent replay checks per new correction | 2 |
| Total independently verified unique correction sequences | 12 |
| Exported LeRobotDataset v3 | 12 episodes / 956 transitions |
| Dataset writer throughput | 26.21 transitions/s |
| Registered policy-loop GPU time | 0.573466 GPU-hours |
| Peak allocated GPU memory | 0.898 GiB |

The 12 exported episodes comprise 11 fresh M04 corrective trajectories and one historical M03-derived replay. The 12 success rows in the attempt log are not 12 newly generated M04 corrections.

The verified development dataset is limited to LIBERO Goal task 6 and M02 initial-state IDs 6 and 7. It is a task-limited development artifact and does not support cross-task generalization claims.

</details>

## Inference optimization

The FP32 CUDA Graph benchmark compared eager and graph execution across 18 paired observations, using the same checkpoint and source trajectory.

| Metric | Eager FP32 | CUDA Graph FP32 |
|---|---:|---:|
| P50 latency | 391.74 ms | **56.39 ms** |
| P95 latency | 398.84 ms | **61.85 ms** |
| P99 latency | 399.07 ms | **61.87 ms** |

- P50 speedup: **6.95×**
- Action-output maximum absolute error: **0**
- MuJoCo continuation: matched across all paired observations
- Cold CUDA Graph capture: 1.816 s, reported separately

The timing scope is a full raw action chunk with GPU synchronization. Preprocessing and simulator execution are excluded. The result is not an end-to-end task latency claim.

## System workflow

```text
Failed manipulation episode
        ↓
Source episode, state, action and provenance audit
        ↓
Controlled SmolVLA corrective inference
        ↓
MuJoCo execution and native success verification
        ↓
Independent replay verification (two passes per new correction)
        ↓
Deduplication and provenance checks
        ↓
LeRobotDataset v3 export
        ↓
Independent action/state round-trip audit
```

## Reproducibility and evidence

The repository retains the implementation and evidence boundaries separately.

- [M04 evidence report](docs/results/M04/REPORT.md)
- [M04 work report](docs/results/M04/WORK_REPORT.md)
- [M04 result summary](docs/results/M04/results_summary.json)
- [CUDA Graph benchmark](docs/results/M04/cuda_graph_benchmark.json)
- [Physical continuation comparison](docs/results/M04/cuda_graph_physical_comparison.json)
- [M04 reproduction script and dependency snapshot](docs/results/M04/reproduce.sh)
- [M02 LIBERO benchmark branch](https://github.com/Benjamindaoson/VLA-Forge/tree/codex/m02-libero40-benchmark)

The Hugging Face [evidence Space](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence) and [corrective dataset](https://huggingface.co/datasets/jlai300/vla-forge-m04-corrections-dev) are access-restricted. Their contents are not represented as publicly downloadable artifacts.

## Current status and limitations

- **M04 status: INCOMPLETE.** The development evidence is real, but the requested broader scale and publication gates have not passed.
- The current corrective dataset covers one LIBERO task and two source initial states; no task-generalization claim is made.
- M05 controlled retraining on corrective data has not yet been completed. No policy-recovery improvement from retraining is claimed.
- The current results are simulator-based. They do not establish physical-robot deployment.
- Public release of media and dataset artifacts remains withheld pending asset-rights review.
- BF16 and TF32 were not accepted for the reported inference comparison; the accepted comparison uses FP32.

## Project provenance

The implementation builds on the official LeRobot, SmolVLA, LIBERO, robosuite, MuJoCo, PyTorch, and EGL ecosystems. External models, simulators, datasets, and checkpoints retain their respective upstream licenses and redistribution terms. The project does not claim ownership of upstream model architectures or external assets.
