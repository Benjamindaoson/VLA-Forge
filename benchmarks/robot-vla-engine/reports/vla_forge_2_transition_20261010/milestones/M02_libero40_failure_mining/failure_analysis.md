# M02 — Standard LIBERO 40 benchmark and failure mining

**Acceptance: PASS.** Benchmark status: PASS. A complete success rate is reported only when all 400 Episodes and required artifacts validate. Manual failure review is 20/20.

## Benchmark results

- spatial: 63/100 (63.0%); 95% Wilson [53.2%, 71.8%].
- object: 75/100 (75.0%); 95% Wilson [65.7%, 82.5%].
- goal: 84/100 (84.0%); 95% Wilson [75.6%, 89.9%].
- long: 50/100 (50.0%); 95% Wilson [40.4%, 59.6%].

- Overall pooled success: 272/400 (68.0%); 95% Wilson [63.3%, 72.4%].
- Overall task-macro success rate: 68.0%.
- The interval is a binomial Wilson interval for the observed fixed-state sample, not a multi-seed stability claim.

## Failure analysis

- Measured failed Episodes: 128.
- Top categories:
- Timeout: 108 (84.4%).
- Placement / Goal Failure: 11 (8.6%).
- Unknown / Insufficient Evidence: 5 (3.9%).
- Sequence / Long-Horizon Failure: 4 (3.1%).
- Verified M03 candidate failures: 128.
- Cause is assigned only for horizon timeout or explicit manual video/trajectory review. Other failures remain Unknown / Insufficient Evidence.
- Evidence integrity: 400/400 paired-camera videos decoded; 400/400 action traces exactly match recorded environment commands; 83827 decoded video frames total.
- Three lowest task success rates:
- `spatial` task 5 — pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate: 0/10 (0.0%).
- `long` task 0 — LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket: 2/10 (20.0%).
- `long` task 4 — LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate: 3/10 (30.0%).

## Manual review sample

The deterministic review sample takes five failures per suite, spread across that suite's sorted failures. Results, evidence notes, and video/contact-sheet hashes are in `manual_failure_review.jsonl`; the current status is 20/20 reviewed.

Review contact sheets:
- `manual_review/stratified_review_page1.jpg` (spatial sample).
- `manual_review/stratified_review_page2.jpg` (object sample).
- `manual_review/stratified_review_page3.jpg` (goal sample).
- `manual_review/stratified_review_page4.jpg` (long sample).

- `spatial` task 0 Episode 2: /data/vla-forge-artifacts/M02_libero40/registry/runs/20261010T030315-982cacc8515f/videos/spatial_task00_episode02_attempt1.mp4
- `spatial` task 3 Episode 7: /data/vla-forge-artifacts/M02_libero40/registry/runs/20261010T030315-982cacc8515f/videos/spatial_task03_episode07_attempt1.mp4
- `spatial` task 5 Episode 1: /data/vla-forge-artifacts/M02_libero40/registry/runs/20261010T030315-982cacc8515f/videos/spatial_task05_episode01_attempt1.mp4
- `spatial` task 6 Episode 9: /data/vla-forge-artifacts/M02_libero40/registry/runs/20261010T030315-982cacc8515f/videos/spatial_task06_episode09_attempt1.mp4
- `spatial` task 9 Episode 7: /data/vla-forge-artifacts/M02_libero40/registry/runs/20261010T030315-982cacc8515f/videos/spatial_task09_episode07_attempt1.mp4
- `object` task 0 Episode 3: /data/vla-forge-artifacts/M02_libero40/registry/runs/20261010T030315-982cacc8515f/videos/object_task00_episode03_attempt1.mp4
- `object` task 2 Episode 4: /data/vla-forge-artifacts/M02_libero40/registry/runs/20261010T030315-982cacc8515f/videos/object_task02_episode04_attempt1.mp4
- `object` task 5 Episode 6: /data/vla-forge-artifacts/M02_libero40/registry/runs/20261010T030315-982cacc8515f/videos/object_task05_episode06_attempt1.mp4
- `object` task 7 Episode 7: /data/vla-forge-artifacts/M02_libero40/registry/runs/20261010T030315-982cacc8515f/videos/object_task07_episode07_attempt1.mp4
- `object` task 9 Episode 9: /data/vla-forge-artifacts/M02_libero40/registry/runs/20261010T030315-982cacc8515f/videos/object_task09_episode09_attempt1.mp4
- `goal` task 2 Episode 3: /data/vla-forge-artifacts/M02_libero40/registry/runs/20261010T030315-982cacc8515f/videos/goal_task02_episode03_attempt1.mp4
- `goal` task 4 Episode 4: /data/vla-forge-artifacts/M02_libero40/registry/runs/20261010T030315-982cacc8515f/videos/goal_task04_episode04_attempt1.mp4
- `goal` task 6 Episode 2: /data/vla-forge-artifacts/M02_libero40/registry/runs/20261010T030315-982cacc8515f/videos/goal_task06_episode02_attempt1.mp4
- `goal` task 6 Episode 7: /data/vla-forge-artifacts/M02_libero40/registry/runs/20261010T030315-982cacc8515f/videos/goal_task06_episode07_attempt1.mp4
- `goal` task 8 Episode 7: /data/vla-forge-artifacts/M02_libero40/registry/runs/20261010T030315-982cacc8515f/videos/goal_task08_episode07_attempt1.mp4
- `long` task 0 Episode 0: /data/vla-forge-artifacts/M02_libero40/registry/runs/20261010T030315-982cacc8515f/videos/long_task00_episode00_attempt1.mp4
- `long` task 1 Episode 8: /data/vla-forge-artifacts/M02_libero40/registry/runs/20261010T030315-982cacc8515f/videos/long_task01_episode08_attempt1.mp4
- `long` task 4 Episode 6: /data/vla-forge-artifacts/M02_libero40/registry/runs/20261010T030315-982cacc8515f/videos/long_task04_episode06_attempt1.mp4
- `long` task 7 Episode 6: /data/vla-forge-artifacts/M02_libero40/registry/runs/20261010T030315-982cacc8515f/videos/long_task07_episode06_attempt1.mp4
- `long` task 9 Episode 9: /data/vla-forge-artifacts/M02_libero40/registry/runs/20261010T030315-982cacc8515f/videos/long_task09_episode09_attempt1.mp4

## Runtime and provenance

- Policy: `lerobot/smolvla_libero@31d453f7edd78c839a8bbc39744a292686daf0de`; weight SHA-256 `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`.
- Dataset: `lerobot/libero@a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`; metadata SHA-256 `630630a1a4f07fc9fc1f1b2b7987ed3bd034bd8cffaaf8a0e6939ae9c468445d`.
- GPU: `NVIDIA GeForce RTX 3090`; PyTorch allocator peak 964068864 bytes; sampled full-device peak 1297.0 MiB from 17592 SQLite resource events.
- Total completed Episode wall time: 15108.9s.
- Mean per-Episode policy inference mean: 9.034809118356536 ms; per-Episode p95 mean: 2.7184603806249994 ms; action-chunk generation mean: 310.01847868874245 ms.
- Code identity and environment versions: `manifest.json`; Episode-level costs and artifact hashes: `episodes.csv`; full events/resources: external SQLite Registry.

## Artifacts and reproduction

- Episode rows: `episodes.csv`; task statistics: `task_metrics.csv`; suite/global metrics: `suite_metrics.json`.
- Failure database: `failure_cases.jsonl`; review queue: `manual_review_queue.csv`.
- Videos, trajectories, action traces and Registry: `/data/vla-forge-artifacts/M02_libero40`.
- Reproduce/resume: `bash reports/vla_forge_2_transition_20261010/milestones/M02_libero40_failure_mining/reproduce.sh`.

## Limitations

This is a single fixed 10-initial-state sample per task under one frozen policy and simulator protocol. It does not establish multi-seed robustness. Visual failure review is evidence-bound and does not create corrected demonstrations. No training or intervention was performed in M02.

The attempt ledger preserves one failed preflight recording attempt caused by a missing video output directory. The directory creation fix was applied before the benchmark queue; that preflight error is retained in `episode_attempts.jsonl` and excluded from the 400 formal Episode denominator. All 400 formal Episodes subsequently passed artifact verification.


## Representative videos mirrored for review

Two small review copies are included locally; their SHA-256 values match the corresponding authoritative external artifacts. The complete 400-video set remains under the external artifact root recorded in `manifest.json` and `episodes.csv`.

- **Success — goal task 0 Episode 0**: [`goal_task00_episode00_attempt1.mp4`](manual_review/sample_videos/goal_task00_episode00_attempt1.mp4); SHA-256 `00cb5b928d874977da8fab45ea1750b7bedcf0a513b17cc6c26444a7f1990eed`.
- **Failure — spatial task 3 Episode 7**: [`spatial_task03_episode07_attempt1.mp4`](manual_review/sample_videos/spatial_task03_episode07_attempt1.mp4); SHA-256 `aa3f6ad3be23902a401e5e07c63e55decd7c7273a4359058b683eb99b6fca307`.
