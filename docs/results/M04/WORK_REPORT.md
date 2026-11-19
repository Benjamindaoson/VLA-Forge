# VLA-Forge M04 Work Report

**Execution date:** 2026-10-11  
**GPU environment:** Rented NVIDIA RTX 4090 24GB instance  
**Status:** INCOMPLETE — verified engineering outputs are available, but broader scale and publication gates have not passed.

## 1. Scope

M04 evaluated a controlled SmolVLA failure-recovery workflow in LIBERO/MuJoCo. The objective was to generate corrective manipulation trajectories from failed attempts, verify them independently through simulator replay, and export an auditable LeRobotDataset v3 development artifact.

The experiment does not claim physical-robot deployment, cross-task generalization, or a measured improvement from retraining the policy on corrective data.

## 2. Policy attempts and corrective trajectories

| Item | Result |
|---|---:|
| Registered policy-attempt rows | 79 |
| Successful attempt rows | 12 |
| Failed attempt rows | 67 |
| Non-episode engineering errors | 2 |
| Operator-interrupted pre-policy runs | 2 |
| Fresh M04 corrective trajectories | 11 |
| Replay validations per new correction | 2 |
| Total independently verified unique correction sequences | 12 |

The 12 independently verified sequences consist of 11 fresh M04 SmolVLA inference corrections and one historical M03-derived physical replay. The 12 successful attempt rows are not 12 newly generated M04 corrections.

All failed attempts and engineering exceptions remain recorded rather than being removed from the denominator.

## 3. Dataset export and provenance

The exported LeRobotDataset v3 development artifact contains:

- 12 episodes.
- 956 transitions.
- 11 fresh M04 corrective trajectories.
- One M03-derived historical replay.
- Action and state round-trip maximum absolute error: 0.
- Mean absolute image error: 2.190.
- Dataset writer throughput: 26.21 transitions/second.

The current source scope is limited to LIBERO Goal task 6 and M02 initial-state IDs 6 and 7. The dataset is not task-diverse and does not support cross-task generalization claims.

## 4. FP32 CUDA Graph inference optimization

The eager and CUDA Graph paths were compared across 18 paired complete action chunks using the same checkpoint and source trajectories.

| Metric | Eager FP32 | CUDA Graph FP32 |
|---|---:|---:|
| P50 | 391.74 ms | 56.39 ms |
| P95 | 398.84 ms | 61.85 ms |
| P99 | 399.07 ms | 61.87 ms |

The P50 speedup was **6.95×**. Action outputs were identical across paired observations, and the matched MuJoCo continuation checks passed.

Cold CUDA Graph capture took 1.816 seconds and is reported separately. The timing scope covers a synchronized full raw action chunk; preprocessing and simulator execution are excluded. This is not an end-to-end task latency claim.

## 5. Compute accounting

- Registered policy-loop GPU time across fresh attempts, failures, and duplicates: 0.573466 GPU-hours.
- GPU time per fresh M04 verified correction: approximately 0.052133 GPU-hours.
- Peak allocated GPU memory: approximately 0.898 GiB.

These figures cover registered policy-loop windows only. They exclude model loading, restoration, independent replay, encoding, and idle time, and are not equivalent to a rental invoice.

## 6. Remaining acceptance gaps

1. The current verified development set covers one LIBERO task and two initial states.
2. Cross-task coverage and task-generalization evaluation remain incomplete.
3. M05 controlled retraining on corrective data has not been completed.
4. No policy-recovery improvement from M05 retraining is claimed.
5. Public release of media and dataset artifacts remains withheld pending asset-rights review.
6. The current evidence is simulator-based and does not establish physical-robot deployment.

## 7. Evidence files

- [M04 evidence report](REPORT.md)
- [Machine-readable result summary](results_summary.json)
- [CUDA Graph benchmark](cuda_graph_benchmark.json)
- [Physical continuation comparison](cuda_graph_physical_comparison.json)
- [Reproduction script](reproduce.sh)

Private media, raw trajectories, model checkpoints, and the development dataset remain access-restricted. Their existence is not represented as public redistribution permission.
