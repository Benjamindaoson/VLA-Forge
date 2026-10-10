## 1. Contracts and tests

- [ ] 1.1 Implement the 40-task identity/coverage, Wilson interval, and evidence-bounded failure-analysis helpers
- [ ] 1.2 Add tests for exact matrix coverage, duplicate/missing/error rejection, resume validation, and failure classification
- [ ] 1.3 Validate M02 OpenSpec artifacts

## 2. Runner and evidence

- [ ] 2.1 Implement a separate single-environment M02 runner reusing the frozen M01 policy contract
- [ ] 2.2 Capture paired camera video, state/action trajectories, timings, and per-Episode hashes
- [ ] 2.3 Implement interruption-safe persistence, retries, manifest, and independent M02 lock
- [ ] 2.4 Implement task/suite/global metrics, failure database, report, and reproduction entrypoint

## 3. Real benchmark

- [ ] 3.1 Verify server identity, GPU, checkpoint, data, disk, and absence of competing jobs
- [ ] 3.2 Run the frozen smoke/first-episode gate and verify artifacts before continuing
- [ ] 3.3 Complete all 400 valid episodes and preserve attempts/errors
- [ ] 3.4 Verify task coverage, videos, trajectories, hashes, and resource metrics

## 4. Failure review and delivery

- [ ] 4.1 Select at least 20 (or all, if fewer) failed episodes and manually review video/trajectory evidence
- [ ] 4.2 Finalize failure analysis, metrics, validation, report, and resume instructions
- [ ] 4.3 Commit M02 code and evidence to the current local branch; publish on an independent GitHub branch when permitted
- [ ] 4.4 Stop at M02 and report the actual acceptance status
