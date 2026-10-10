# Multimodal Robot Data Engine & VLA Policy Training

**2026-10-10 接续实施：** 新3090已完成207项Linux测试、真实GPU场景与短程更新的工程验收；只读工作台已本地运行，云端新版本部署因执行审批限制待续。新增[M3纠正输入合同](reports/correction_training_inputs_20261010/README.md)支持已注册纠正的来源复核、场景分组和动作分块。真实纠正仍为0，历史微调权重与Gate0原始证据仍待恢复；这些工程检查不表示修复训练或完整方案完成。[最近云端工程交付](reports/server_3090_20261010/delivery_20261010.md)。

**2026-10-10 新机与总计划：** 已连接 `pytorch-2522`（RTX3090 24GB），新机不需重启；旧项目资产尚未确认迁入。已保存 [VLA-Forge 2.0 详细研发与交付方案](docs/superpowers/plans/2026-10-09-vla-forge-2-master-plan.md)，覆盖迁移、纠正采集、训练回归、研究对照和产品验证。下方旧SSH阻塞描述属于旧4090D入口，不代表当前3090连接失败。

**2026-10-09 当前状态：[Repair Gate 1 来源审计](reports/repair_gate1_sources/README.md)已实际执行。** 两个开发任务76条示范/15,590帧，70条普通示范候选通过检查，6条验证集示范排除；纠正轨迹仍未取得。本地172项测试通过、2项平台相关跳过。旧 SSH 入口在握手前断开，当前远端运行状态未知，四个场景的有界纠正采集计划尚未执行。

**2026-10-08 当前状态：已连接迁移后的单卡 RTX 4090D，五组正式实验的 4,249 个注册产物全部通过核验。** [迁移报告](reports/migration_20261008/README.md)。新路线的 [Repair Gate 0](reports/repair_gate0/README.md) 已实际执行：16 次重复闭环对应 8/8 配对动作与状态一致；公共中途状态恢复 0/24 完整合格；从初态重放动作前缀的四个开发检查全部通过。尚未开展新的策略训练，旧矩阵保持 5/24 完整组。以下 2026-10-07 的队列及 RTX 5090D 描述为历史记录，不代表当前运行状态。

2026-10-07 20:30 UTC：[第五组完整结果](reports/diversity_low_seed42/evaluation/README.md)已核验，SmolVLA diversity_low / seed42为78/400（19.5%）；共同seen 78/160，共同unseen 0/80。累计5次训练／100,000更新、5/24组完整评测／2,000回合／592,954控制步。下一组scale_12 seed42训练已由原协调器启动；三seed结论仍N/A。

[第四组正式结果](reports/diversity_high_seed42/evaluation/README.md)：SmolVLA diversity_high / seed42完成20,000次更新和400回合评测，159/400成功（39.75%）；共同seen为88/160、共同unseen为0/80。该结果属于单seed描述性测量；三seed多样性结论仍为N/A。

一个把真实机器人数据、官方策略实现、闭环轨迹和实验指标连接起来的工程研究项目。研究目标是比较 ACT、Diffusion Policy、SmolVLA，并用固定训练预算分析示范数量与任务多样性。

[EgoMimic 双来源全文件审计](reports/egomimic/full/completion.md)已完成：两个 groceries 文件共67.44GB，Human 50 episodes/14,397帧、Robot 1 episode/5,000帧，官方SHA及全量扫描通过。Robot train/valid重叠，配对和物理时间同步仍缺乏依据；不据此宣称跨具身训练可用。

**ACT、Diffusion与SmolVLA baseline_full的seed42均已完成20,000次正式更新和400次闭环。ACT为0/400，Diffusion为20/400（5%），SmolVLA为149/400（37.25%）。** 三个基线结果均通过来源和完整性核验；加上diversity_high与diversity_low结果，矩阵完成5/24，三seed均值仍为N/A，全项目尚未完成。[SmolVLA结果与8个案例](reports/smolvla_seed42/README.md)、[Diffusion结果与案例](reports/diffusion_seed42/README.md)、[ACT证据与诊断](reports/cases/act_seed42_report.md)保留原始轨迹、视频和模型归属；不把单seed差异写成完整策略排名。

[可核验性能表](reports/performance/README.md)已补齐ACT、Diffusion和SmolVLA首个seed的训练成本、持续吞吐、allocator/整卡采样显存、真实chunk时延与实际rollout频率。缺失seed保留N/A，不把性能指标作为闭环能力或优化收益的替代。

[训练数据后端四轮实测](reports/training_backend_profile/gpu_abba/README.md)：PyAV与TorchCodec的400个真实SmolVLA训练批次内容全部一致；GPU短程ABBA合并吞吐分别45.608与43.257 examples/s，TorchCodec下降5.154%。四轮loss与最终模型哈希一致，完整训练和闭环质量仍未由此证明，正式矩阵继续使用PyAV。

[剩余实验串行队列](reports/gpu_phase/remaining_matrix.md)已通过推理与训练后端ABBA阶段，进入剩余21个正式组合，随后执行三seed优化质量闭环。队列启动不代表实验完成；失败会停止并保留记录。

**实测硬件：RTX 5090 D 32607 MiB、25 CPU cores、90 GiB RAM。** PyTorch allocator 限额 23 GiB，整卡显存另行采样；不是 4090D 测试。阶段证据见 [项目证据与A–P验收](reports/project_evidence.md)、[技术报告](reports/technical_report.md)和[带时间戳进度](reports/gpu_phase/live_progress.json)；[GPU启动阶段](reports/gpu_phase/report.md)及[CPU阶段交接](reports/phase1.md)保留为历史记录。

[正式结果](reports/formal/report.md) · [三种策略完整训练曲线](reports/formal/training_complete.md) · [推理优化准备](reports/inference/report.md) · [π0.5外部参考](reports/external_reference/pi05.md) · [中文履历](reports/resume_project_cn.md) · [English resume](reports/resume_project_en.md)

![真实数据审计](reports/figures/data_audit.png)

## 已建立的证据

| 能力 | 可复查证据 | 边界 |
|---|---|---|
| LIBERO 数据治理 | 1,693 episodes / 273,465 frames / 40 tasks；全量数值审计、74 段视频共 546,930 个相机帧解码 | 无源 success/termination 标签；物理时间同步未认证 |
| 全源解码一致性 | [74视频逐帧复核](reports/full_video_equivalence/README.md)：PyAV/TorchCodec的546,930帧RGB逐像素一致，全部PTS最大差0秒 | 只证明源RGB/PTS；完整dataset查询、训练提速与质量保持尚未验证 |
| 异构 Human/Robot 接口 | H&R 102 文件真实全量审计；[EgoMimic groceries 双来源完整审计](reports/egomimic/full/completion.md)：Human50 episodes/14,397帧、Robot1 episode/5,000帧，361条episode-stream完整扫描、官方SHA匹配 | H&R 不是 EgoMimic；Robot train/valid同为demo_0，时间戳及Human–Robot配对关系未证明 |
| 实验追踪 | SQLite runs/events/artifacts，JSONL/CSV 导出，SHA256 产物校验，失败原样登记 | 没有用空值伪造 0 分或 0 显存 |
| 官方策略接入 | ACT/Diffusion 真实数据 loss/backward；SmolVLA 官方权重严格加载、CPU inference 和真实 batch backward | CPU 工程模型/短更新不进入正式 benchmark |
| 可恢复训练 | model、optimizer、Python/NumPy/Torch RNG、配置身份；原子 checkpoint | CUDA ACT/Diffusion 合成下一步恢复差异 0；SmolVLA 真实 checkpoint 恢复通过，预取 RNG 回归通过 |
| 实验设计 | 嵌套数据子集、固定 384 demonstrations 的 16 vs 32 任务对比、共享未见任务 | 帧数不完全相同，任务难度和预训练数据重叠仍是限制 |

[数据审计报告](reports/data_audit.md) · [源图像](reports/figures/source_observations.png) · [运行架构](docs/architecture/runtime.html) · [执行记录](reports/progress.md)

[六条件存储审计](reports/condition_storage/README.md)补齐实际文件依赖：六种示范预算都引用全部74视频，最小240示范子集的引用载荷仍约1.925GB。示范数、有效训练帧和共享编码文件大小分别报告，不能用帧数比例代替磁盘占用。

## 目录

```text
src/robot_vla/     canonical schema、adapters、audit、registry、metrics、runtime contracts
scripts/          下载、审计、CPU smoke、训练、闭环评测、报告
configs/          固定的数据划分、各条件训练统计、GPU 计算协议（阶段二冻结）
reports/          审计、图表、证据和阶段报告
docs/architecture/ source-evidence architecture JSON + self-contained HTML
data/             不提交 Git 的固定 revision 原始数据/模型
artifacts/experiments/  SQLite registry、逐 run 配置、事件、checkpoint、rollout
external/         固定 commit 的官方依赖（远端）
```

## 环境与运行

远端工作根目录 `/root/autodl-tmp/robot-vla`，官方策略使用项目 `.venv-policy`（Python 3.12）；独立数据审计可使用 `.venv`（Python 3.10+）。所有命令从项目根运行。Windows CPU 验证使用 `.venv/Scripts/python.exe` 替换下方解释器。

可通过 `PYTHON_BIN=runtime/python/bin/python3.12 bash scripts/bootstrap.sh` 按 Linux 环境锁重建项目环境；它依赖系统提供 libosmesa6/libgl1/libegl1/ffmpeg。`scripts/configure_libero.py` 只配置项目内的资产路径。CPU 诊断另有 `.venv-sim`，使用相同项目依赖及纯 CPU torch/torchvision；它不代替 GPU 训练环境。

```bash
.venv-policy/bin/python -m robot_vla.cli --help
.venv-policy/bin/python -m pytest -q
.venv-policy/bin/python -m ruff check src scripts tests
.venv-policy/bin/python scripts/audit_libero.py
.venv-policy/bin/python scripts/audit_libero_video.py
.venv-policy/bin/python scripts/audit_human_robot.py
.venv-policy/bin/python scripts/audit_egomimic_views.py
.venv-policy/bin/python scripts/freeze_data_protocol.py
.venv-policy/bin/python scripts/render_data_report.py
```

真实训练入口有受限 CPU smoke 模式：

```bash
.venv-policy/bin/python scripts/train_policy.py \
  --policy act --condition scale_6 --steps 2 --batch-size 1 --device cpu --smoke
```

`--policy diffusion` 与 `--policy smolvla` 使用同一入口。SmolVLA 的 CPU 参数初始化使用 meta parameters 再严格载入官方权重，避免先分配一整份随机权重。`--smoke` 限制最多 10 次更新、使用缩小的空间分辨率/配置，不能冒充正式实验。恢复时传 `--resume <run>/resume.pt --parent-run-id <run_id>`，其他 recipe 参数须相同。

Linux GPU 正式评估采用 EGL 渲染（OSMesa 保留为诊断选项）：

```bash
export LIBERO_CONFIG_PATH="$PWD/work/libero"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
.venv-policy/bin/python scripts/smoke_libero.py --policy act --steps 3
```

统一 evaluator 接受 registry 中可验证归属的 checkpoint，保存结构化轨迹、每段延迟、真实 chunk 调用间隔和代表视频：

```bash
.venv-policy/bin/python scripts/evaluate_policy.py \
  --checkpoint artifacts/experiments/runs/<run_id>/pretrained_model \
  --training-run-id <run_id> --development --task-limit 1 --steps 3 --device cpu
```

正式模式要求正式训练来源及相同协议；缺任务、缺 episode、异常 rollout 不会被填成成功率。默认每 seed 40 tasks × 10 initial states = 400 rollouts，训练 seeds 42/43/44，固定最终更新 checkpoint。算力/统计成本可在正式运行前采用有记录的新协议调整，不能根据测试结果调预算。

## 两个研究问题的固定数据设计

| 条件 | Tasks | Demonstrations | Frames |
|---|---:|---:|---:|
| baseline_full | 40 | 1,573 | 253,433 |
| scale_6 | 40 | 240 | 40,112 |
| scale_12 | 40 | 480 | 80,120 |
| scale_24 | 40 | 960 | 160,720 |
| diversity_low | 16 | 384 | 63,522 |
| diversity_high | 32 | 384 | 63,208 |

120 validation episodes 不参与训练或统计。多样性条件每个 suite 分层选取任务，共享 8 个未见任务和 16 个共同已见任务；只能解释相应对比，不能把用全 40 任务微调的 baseline 当作无泄漏的 unseen comparator。基础模型的预训练任务重叠未知，报告会保留这一限制。

正式训练前需完成 CUDA/24GB 可行性 profiling，并写入固定的 `configs/compute_protocol.json`。未冻结时训练入口拒绝正式启动。所有条件统一 optimizer updates 与 effective batch；训练样本按 seed/step/microbatch 定位，恢复不依赖重新遍历 DataLoader。

GPU 阶段使用 `--profile --steps 3 --batch-size 1 --device cuda` 在完整模型配置上完成有界探测，最多 10 次更新，登记为 `engineering_profile`；它既不缩小模型也不进入正式结果。之后按实际显存/吞吐结果冻结统一预算。计时字段 `training_loop_gpu_hours` 只覆盖优化循环，注册表 wall time 包含模型构建和导出；两者都不冒充云平台账单时长。

ACT/Diffusion 通过 40 维 oracle task one-hot 获得当前目标，拼接到 8 维 proprioception 后形成 48 维策略输入；原始数据 schema 保持 8 维。SmolVLA 使用语言指令和 8 维 state。这个设计避免把未提供目标信息的策略当作多任务基线；但语言预训练与 oracle task ID 的能力差异仍须明确，不能把 RQ1 解释成完全消除了所有条件差异的纯架构对比。

## 科学结果

| Policy | Spatial SR | Object SR | Goal SR | Long SR | GPU-hours | Peak VRAM |
|---|---|---|---|---|---|---|
| ACT | N/A | N/A | N/A | N/A | N/A | N/A |
| Diffusion Policy | N/A | N/A | N/A | N/A | N/A | N/A |
| SmolVLA | N/A | N/A | N/A | N/A | N/A | N/A |

正式表格只从完整 evaluator 记录生成。CPU smoke 的单次 latency 不与 GPU benchmark 混合；π0.5 若引用公开结果，将单列 external reference。数据曲线与性能提升图在真实结果产生前不会绘制占位趋势。

## 上游与版本

- LeRobot: `200ee53596d464bd28f6595cdb31be69a2b5e379`
- LIBERO 参考源码/任务与资产: `8f1084e3132a39270c3a13ebe37270a43ece2a01`；实际运行 Python 包为 `hf-libero==0.1.4`，并非宣称上述 clone 是实际导入的 Python 实现。缺失的 wheel assets 由该固定 clone 的 assets 补齐。
- `lerobot/libero`: `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`
- `gatech/EgoMimic`: `065ffc0c697069982d58b695db38d9943a28d54a`
- `dannyXSC/HumanAndRobot`: `79c1b0e5f488567aa8fea87f98c1c8a286227cc1`
- `lerobot/smolvla_base`: `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`
- SmolVLM2 processor/config: `7b375e1b73b11138ff12fe22c8f2822d8fe03467`

上游文档：[LeRobot LIBERO](https://huggingface.co/docs/lerobot/libero)、[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)、[EgoMimic](https://github.com/SimarKareer/EgoMimic)。数据/模型的许可证与使用条款遵循各自上游；源资产不加入本仓库 Git 历史。

## 正式结果分析与进度

[结果表](reports/formal/report.md)只纳入来源和完整性检查通过的正式评测；未完成的seed/条件保持N/A。数据多样性的主要对比是16个共同已见任务和8个共同未见任务，全40任务结果标为补充统计。

```bash
.venv-policy/bin/python scripts/analyze_experiments.py
.venv-policy/bin/python scripts/render_training_progress.py
.venv-policy/bin/python scripts/render_formal_figures.py
```

![训练进度快照](reports/formal/training_progress.png)

这是带时间戳的训练快照，loss不代表闭环成功率；原始逐步指标、采样资源和图表来源保存在`reports/formal/`及registry快照中。

[正式图表说明](reports/formal/figures/README.md)包含策略、suite、scaling及diversity图。输入须匹配完整登记的analysis报告，图表重验引用的评测与父checkpoint产物。单seed只显示初步点，三seed齐全才画均值±样本标准差；缺失保持N/A，不能看成0%。

## 推理性能实验

[推理阶段报告](reports/inference/report.md)记录了输入来源和冻结测量方案。[正式 CUDA ABBA 结果](reports/inference/seed42_abba/README.md)已完成：同一 seed42 最终 checkpoint 的完整动作块前向中位耗时为 FP32 153.223 ms、BF16 170.016 ms，BF16 延迟增加 10.960%，未通过提速门槛。22 个交付文件及原始测量通过核验和独立复算；动作差异不代表闭环质量，三 seed 质量保持仍未验证。

[优化闭环工程报告](reports/optimization_quality/implementation.md)说明独立的 BF16 evaluator、真实 20 步 CPU 环境验证及严格配对的质量分析器。[质量结果](reports/optimization_quality/report.md)由已验证的完整 FP32/BF16 同 checkpoint 评测生成，缺测时保持 UNVERIFIED。

```bash
.venv-policy/bin/python scripts/analyze_optimization_quality.py
```

[三策略动作诊断](reports/action_diagnostics/README.md)补齐全部1,200条已完成轨迹：ACT/DP裁剪前超界均为0，SmolVLA超界分量占6.653%；DP另有12,270个执行动作边界值。裁剪前超界与执行边界分别统计，不据此归因失败。
