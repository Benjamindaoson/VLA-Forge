# M02 review artifacts

The contact sheets contain three paired-camera checkpoints per failure Episode at approximately 15%, 55%, and 90% of the video. Review sample selection is deterministic and balanced at five failures per LIBERO suite, spread across the suite's sorted failures. The reviewed Episode IDs and evidence notes are in `stratified_review_ids.csv` and the sibling `manual_failure_review.jsonl`. Video, trajectory, action-trace, and contact-sheet SHA-256 values are recorded with each review.

The `sample_videos/` directory contains two small playable copies (one success and one failure); `sample_video_index.csv` maps them to their canonical external artifact paths and expected hashes. All 400 canonical videos and trajectories remain at the artifact root in `manifest.json`.
