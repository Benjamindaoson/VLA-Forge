# VLA-Forge 2.0 当前执行状态

- 更新时间：2026-10-10T16:50:33+08:00（Asia/Shanghai）。
- 当前里程碑：M02 标准 LIBERO 40 评测与失败挖掘，**PASS**。本次范围到此停止，等待用户验收。
- M01：PASS，正式 24/40（60%）；M02 使用其冻结的 SmolVLA 权重、Processor/Normalizer、双相机映射、8 维状态输入、7 维相对动作和 action-chunk 配置。
- M02：Spatial、Object、Goal、Long 各完成 10 个任务 × 10 个固定初始状态，共 **400/400 有效 Episode**；272 成功、128 失败，总体 68.0%。
- Suite 成功：Spatial 63/100；Object 75/100；Goal 84/100；Long 50/100。Wilson 95% 区间及 40 个任务明细见 `milestones/M02_libero40_failure_mining/task_metrics.csv` 和 `suite_metrics.json`。
- 完整性：400/400 双相机视频解码成功（83,827 总帧），400/400 动作轨迹与实际传给 `env.step` 的命令逐项相等；400 条视频、轨迹、动作 trace 与 Episode metadata 均按哈希验证。
- 失败数据库：128 条 Episode 级案例均关联真实视频、状态/动作轨迹、动作 trace 和前缀引用。分类：Timeout 108，Placement/Goal 11，Sequence/Long-Horizon 4，Unknown 5。人工复核 20 条，四个 suite 各 5 条；证据与置信信息保存在 `manual_failure_review.jsonl`。
- M03 候选：128 条具有有效视频、轨迹和动作 trace 的失败案例。未开展干预、训练或纠正数据生成。
- 资源：授权 RTX 3090 24 GiB；Episode wall time 合计 15,108.9 秒（4.20 小时），runner 从开始到 rollout 文件结束约 5 小时 15 分钟；整套分析与人工复核完成于约 5 小时 44 分钟。采样峰值显存 1,297 MiB，平均采样 GPU 利用率约 0.78%。
- 正式队列外保留 1 条失败的预检录制尝试（视频目录不存在）；修复后重跑通过，该预检错误仍保存在 `episode_attempts.jsonl`，未计入 400 个正式分母。
- 代码与实验来源：formal runner 使用本地提交 `f43e412f627118b173e3257f71a58a64c96ef768`；分析/抽样回归修复为 `86223c78fb7e3b829040be41d34ebb3e453383a4`。
- 交付：`milestones/M02_libero40_failure_mining/REPORT.md`。复现命令：`bash reports/vla_forge_2_transition_20261010/milestones/M02_libero40_failure_mining/reproduce.sh`。400 条完整视频和 Registry 保存在 manifest 所列的外部制品根目录；成功/失败示例视频已在本地报告目录留存。
- 历史 Stage 00–06 的 repair-baseline Gate 状态仍为 BLOCKED，且与本次 M02 官方策略基线分开记录。
- 下一步：**停止并等待用户检查 M02 实际结果；未经单独授权不启动 M03。**
