"""Prepare a private cloud delivery from validated evidence; never publishes."""

import hashlib
import json
from pathlib import Path

from vla_forge.splits import LeakageError, assert_no_leakage

ROOT = Path("/data/VLA-Forge/Evidence/M04")
DATASET = ROOT / "dataset/lerobot_v3_verified_r7"


def main():
    provenance = json.loads((DATASET / "provenance.json").read_text())
    records = []
    for source in provenance["sources"]:
        records.append(
            {
                "split_group_id": f"libero_goal:task{source['task_id']}:init{source['init_state_id']}",
                "split": "train",
                "source_episode_id": f"goal_task{source['task_id']:02d}_episode{source['init_state_id']:02d}_attempt1",
                "correction_episode_id": source["episode_id"],
            }
        )
    assert_no_leakage(records)
    groups = sorted({record["split_group_id"] for record in records})
    rejected = 0
    for group in groups:
        duplicate = next(x.copy() for x in records if x["split_group_id"] == group)
        duplicate["split"] = "test"
        try:
            assert_no_leakage(records + [duplicate])
        except LeakageError:
            rejected += 1
        else:
            raise AssertionError("Leakage guard accepted a known source-family collision")
    audit = dict(
        status="PASS",
        records=records,
        development_only=True,
        training_executed=False,
        excluded_from_future_M06_groups=groups,
        negative_control_collisions_rejected=rejected,
        boundary="Development only. Future M06 must enforce these source-family exclusions; no held-out result is claimed.",
    )
    (DATASET / "lineage_validation.json").write_text(json.dumps(audit, indent=2) + "\n")
    card = ROOT / "publication/dataset_card_draft.md"
    text = card.read_text().replace("DRAFT — NOT PUBLISHED", "PRIVATE DEVELOPMENT RELEASE")
    text = text.replace(
        "Independent replay: both correction sequences returned Success=True in the saved M04 receipts; the fresh policy sequence was replayed twice.",
        "Independent replay: each of the 11 fresh correction sequences returned Success=True in two independent physical replays. The M03-derived sequence is identified separately.",
    )
    text = (
        text[: text.index("## License and redistribution")]
        + """## License and redistribution
Private development evidence only. Public redistribution is withheld: the exact pinned LIBERO asset bundle has no dataset card/license declaration. LIBERO software is MIT; the software license alone does not settle third-party mesh/texture provenance. No blanket CC-BY or Apache license is asserted for these generated outputs.

## Attribution
- LIBERO: https://github.com/Lifelong-Robot-Learning/LIBERO
- Asset source: https://huggingface.co/datasets/lerobot/libero-assets
- Policy: https://huggingface.co/lerobot/smolvla_libero

## Splits and leakage
All 12 records are development data (LeRobot train partition). Both source families and all derivatives are excluded from future M06 testing. The existing lineage guard rejected deliberate test-split collisions for both families. No training or held-out improvement is claimed.
"""
    )
    card.write_text(text)
    (DATASET / "README.md").write_text(text)
    space = ROOT / "publication/private_space"
    space.mkdir(exist_ok=True)
    (space / "README.md").write_text(
        "---\ntitle: VLA Forge M04 Evidence\nemoji: 🤖\ncolorFrom: blue\ncolorTo: gray\nsdk: static\napp_file: index.html\npinned: false\n---\nPrivate evidence delivery. Public redistribution review is still pending. No paid hardware requested.\n"
    )
    (space / "index.html").write_text(
        """<!doctype html><html lang="zh"><meta charset="utf-8"><title>VLA-Forge M04</title><style>body{font:17px system-ui;max-width:1120px;margin:30px auto;padding:0 20px;background:#f4f6fa;color:#142439}video{width:100%;background:black}a{color:#175ca0}pre{white-space:pre-wrap}section{background:white;padding:24px;margin:20px 0;border-radius:12px}</style><h1>VLA-Forge M04 · 真实纠正与 FP32 推理加速</h1><p>私人开发证据交付。公开资产再分发审查尚未完成。</p><section><h2>227.9 秒真实项目演示</h2><video controls preload="metadata" src="demo/M04_demo_200s.mp4"></video><p>1920×1080 · H.264 · 30 FPS · 37 条真实 Episode 素材。</p></section><section><h2>已经验证的结果</h2><p>11 条新纠正，每条独立重放两次；另保留 1 条 M03 来源序列。LeRobot v3：12 Episodes / 956 transitions。来源限于一个任务、两个初始状态。</p><p>FP32 CUDA Graph：18 组完整动作块配对，中位 391.7 → 56.4 ms，6.95 倍；动作最大误差 0。21 步真实纠正中动作、物理状态和相机画面均与基线相同，Success=True。</p><p>首次捕获 1.82 秒；短任务可能无法摊平冷启动成本。未声称端到端提速或模型成功率提升。</p><a href="REPORT.md">完整报告</a> · <a href="metrics.json">全部指标</a> · <a href="performance/cuda_graph_benchmark.json">原始推理计时</a> · <a href="performance/cuda_graph_physical_comparison.json">物理对照</a> · <a href="demo/visualization/index.html">原始案例图库</a> · <a href="reproduce.sh">复现入口</a></section></html>"""
    )
    dataset_files = [
        x
        for x in DATASET.rglob("*")
        if x.is_file() and "images" not in x.relative_to(DATASET).parts
    ]
    space_files = [(p, p.relative_to(space).as_posix()) for p in space.iterdir() if p.is_file()]
    for rel in [
        "REPORT.md",
        "metrics.json",
        "manifest.json",
        "validation.json",
        "episodes.csv",
        "failure_cases.jsonl",
        "reproduce.sh",
        "code_commit.json",
    ]:
        p = ROOT / rel
        if p.exists():
            space_files.append((p, rel))
    for sub in ["demo", "performance"]:
        for p in (ROOT / sub).rglob("*"):
            if p.is_file() and (sub == "demo" or p.suffix == ".json"):
                space_files.append((p, p.relative_to(ROOT).as_posix()))

    def entry(path, relative):
        return dict(
            source=str(path),
            target=relative,
            bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    plan = {
        "private": True,
        "public_release": False,
        "repositories": [
            {
                "repo_id": "jlai300/vla-forge-m04-corrections-dev",
                "repo_type": "dataset",
                "files": [entry(p, p.relative_to(DATASET).as_posix()) for p in dataset_files],
            },
            {
                "repo_id": "jlai300/vla-forge-m04-evidence",
                "repo_type": "space",
                "space_sdk": "static",
                "files": [entry(p, rel) for p, rel in space_files],
            },
        ],
    }
    (ROOT / "publication/upload_manifest.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(
        {
            "lineage_records": len(records),
            "negative_controls_rejected": rejected,
            "private_upload_files": [
                (x["repo_id"], len(x["files"]), sum(f["bytes"] for f in x["files"]))
                for x in plan["repositories"]
            ],
        }
    )


if __name__ == "__main__":
    main()
