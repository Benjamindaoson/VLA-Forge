# M06 Artifact Index

M06 engineering acceptance is **PASS**. Algorithm release decisions remain separate: Frozen `PROMOTE` for authorized simulation, Standard and Replay `INSUFFICIENT_EVIDENCE`, Corrective `REJECT`.

## GitHub text and numeric evidence

- `REPORT.md`: final evidence-bounded report.
- `validation.json`: 1,600/1,600 coverage, hash, action legality, state matching, and video-frame checks.
- `manifest.json` and `protocol_frozen.json`: frozen identities, code/model/environment versions, and protocol SHA-256.
- `episodes.csv`, `task_metrics.csv`, `suite_metrics.json`, and `paired_outcomes.csv`: episode-, task-, suite-, and paired-level numeric outcomes.
- `target_repair.json`, `skill_retention.json`, and `statistical_analysis.json`: paired target/legacy analysis and uncertainty.
- `performance.json` and `cost_report.json`: measured latency, utilization, memory, and GPU-hour accounting.
- `release_decisions.json` and `modelops_summary.json`: VLA-Forge and ModelOps gate decisions.
- `video_validation.json` and `video_sources.json`: video encoding/source hashes without redistributing the videos.
- `reproduce.sh`: authorized Linux/GPU reproduction entrypoint.
- `backup_inventory.jsonl` and `backup_summary.json`: complete private-backup inventory and root hashes.

## Private large evidence backup

Full videos, RGB observations, action/state trajectories, simulator-state payloads, formal receipts, and technical attempts are stored in the private Dataset:

- <https://huggingface.co/datasets/jlai300/vla-forge-m06-private-evidence>
- Source: `/data/VLA-Forge/Evidence/M06`
- Inventory: 10,648 source files, 1,514,691,560 bytes before inventory/summary metadata
- Inventory SHA-256: `388044911c848aad67de5a99c523d3ca834beb2f1870be3c9b6925253f31bfce`
- Final report SHA-256: `6ab07004dc74696136134a6b643681baae32dd3965e4f259efb2949a064e8166`
- Demo SHA-256: `483ad5daddb3e534fd277efdbdc5806d8069742b99b9f576043bb2de10b6d382`
- Private archive: `archives/VLA-Forge_M06_Evidence_20261012.tar.zst` (1,253,005,454 bytes)
- Archive SHA-256: `7892d4b52c131a856f2f53aa2baaa0c140187626cfe6f99fad37e5ff90724474`
- Verified archive content revision: `58c5ec934b5cb4b3ff96c383acbddf744a59a6a2`
- GitHub evidence branch: `codex/m06-evidence-backup`

The Dataset is intentionally private. Its presence does not grant public redistribution rights for visual assets, meshes, textures, trajectory payloads, or model weights.

