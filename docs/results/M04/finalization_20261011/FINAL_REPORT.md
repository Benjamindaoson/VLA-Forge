# M04 最终收尾与 M05 准入检查

本次收尾执行完成；完整视觉公开和历史全成本核算仍有明确外部/缺失证据限制。M05 只通过有条件工程数据准入，未训练任何策略。

| 项目 | 实际结论 |
|---|---|
| 原有成果 | 11 条新纠正、12 Episode、956 transitions、CUDA Graph 与全部视频保留，r7 全文件 SHA256 不变 |
| 数据复核 | 12/12 对齐、源失败、实际动作、状态及标签检查通过；11 条新纠正的 22 份既有成功重放证据通过哈希复核 |
| Registry | 同一旧失败 ID 被复用；2 行记录逻辑隔离，不在训练数据中；原日志未改写 |
| 许可 | 586 个上游文件及 280 个媒体/含图像轨迹逐项登记；585 上游资源与官方固定版本相同 |
| 公开交付 | 原创数值 Dataset 与报告 Space 公开；完整视觉 r7 和 Demo 私有；以 public_publication.json 的匿名下载验证为准 |
| 跨任务 | Goal task 4 / init 4，3 次固定尝试全部失败，328.473 秒后停止，0 新纠正；全部负结果保留 |
| 训练/测试隔离 | 6 个旧预留来源已污染并退役；所有开发源及派生组冻结；400 个替代测试条件无来源组/初始状态字节重叠 |
| 成本 | 分开记录加载、策略搜索、失败、成功、重放、编码/写入；未测历史时间及真实租卡金额为 N/A |
| M03 源码 | 从旧云端目录找回 6 个未提交源码/测试文件，按原字节导入权威项目；原始目录未修改 |
| 验证 | 91 passed；Ruff 通过；M03 CLI --help 通过；终态有界控制器重启为 no-op；实数据审计通过 |

## 可访问交付
- 数值 Dataset：https://huggingface.co/datasets/jlai300/vla-forge-m04-numeric-evidence
- 公开报告 Space：https://huggingface.co/spaces/jlai300/vla-forge-m04-audited-evidence
- 可直接打开页面：https://jlai300-vla-forge-m04-audited-evidence.static.hf.space
- 独立代码分支：https://github.com/Benjamindaoson/VLA-Forge/tree/codex/m04-final-admission
- 原私有视觉 Space：https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence
- Git 同步结果与确切 SHA：git_sync.json。没有强制推送，没有覆盖 main 历史。

## 仍不能声称的成果
完整视觉数据公开未通过：HOPE 的更具体 NC-SA 来源及场景/扫描/TurboSquid 素材的逐项署名/许可链尚未完全协调。许可报告逐文件列出依据和最小解除条件；公开数值子集不包含这些媒体或素材。

完整生产成本不可恢复：历史若干加载/前缀/编码/空闲阶段缺独立计时，且没有租卡发票或费率。已知策略循环 2064.476 秒（失败 1839.316 秒、成功 225.160 秒）；22 个独立重放纠正段 354.716 秒；r7 写入/编码 36.478 秒。再计入已找到的动作迁移失败和 M03 派生重放记录，已知阶段墙钟总和下界 2732.178 秒。不能把该下界或原 0.5735 小时包装成全部 GPU 生产成本。

6.95 倍仅对应既有 CUDA Graph 暖态 action chunk 推理中位数（391.74→56.39 ms），不是端到端吞吐；本次不重跑或改变该实验。单任务、两个源状态的成功纠正不证明跨任务泛化，也未证明新策略效果提高。

## 后续 M05 决策
仅建议在另行授权后，用冻结的 12 条开发样本做有预算的非商业研究可行性试验，保持 Frozen Baseline、数据量和评测条件一致；禁止使用退役测试来源或查看新测试结果后调参。完整图像数据仍留在现有云端目录。未进入 M05。

## 报告与复现
LICENSE_REVIEW.md、license_inventory.csv、REGISTRY_AUDIT.md、COST_REPORT.md、CROSS_TASK_REPORT.md、M05_ADMISSION.md，及对应机器可读 JSON 均位于同一 finalization_20261011 目录。

```bash
cd /data/VLA-Forge/project/robot-vla
bash /data/VLA-Forge/Evidence/M04/finalization_20261011/reproduce.sh
```
该命令复核代码及已冻结证据；不会重新开始已经终止的搜索，不会训练策略。
