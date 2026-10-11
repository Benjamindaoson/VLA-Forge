# VLA-Forge

## M04 finalization and M05 admission

[Final report](docs/results/M04/finalization_20261011/FINAL_REPORT.md) · [Public numeric Dataset](https://huggingface.co/datasets/jlai300/vla-forge-m04-numeric-evidence) · [Public report Space](https://huggingface.co/spaces/jlai300/vla-forge-m04-audited-evidence)

The frozen visual r7 dataset retains **12 episodes / 956 transitions: 11 fresh corrections plus one M03-derived correction**. All 12 passed a fresh source/action/state/label audit; the 11 fresh corrections retain 22 independently successful physical replay receipts. Two ambiguous historical failed ledger rows are quarantined. Six formerly reserved test source groups were used during development and have been retired; 400 replacement test conditions are frozen without source-group or initial-state byte overlap.

The new public Dataset is a **numeric-only subset**, with no camera frames, video, meshes, textures or model weights. The complete visual dataset and demo remain private pending asset-specific license reconciliation. The bounded cross-task check produced **0/3 successes** on Goal task 4 and stopped after 328.473 seconds. Historical full production costs and rental billing remain unavailable; warm CUDA Graph chunk acceleration is **not an end-to-end 6.95× speedup**. M05 has conditional engineering admission only; no new policy was trained.


**机器人策略失败恢复、可信纠正数据与高效 VLA 推理**
*Failure recovery, replay-verified corrective data, and efficient VLA inference.*

VLA-Forge 帮助机器人学习团队把一次策略失败变成可追溯、可重放的诊断与纠正样本：保留失败来源，恢复 MuJoCo 物理状态，执行受控干预与新的策略动作，验证成功信号，再独立重放并写入 LeRobotDataset v3。

**当前结果（2026-10-11）：** M04 已完成 11 条新纠正轨迹的独立重放验证，以及 FP32 CUDA Graph 推理优化。数据集和演示已私有交付；完整公开发布仍为 **INCOMPLETE**，等待精确素材版本的再分发许可确认。M05 模型训练尚未开展。

## 演示、视频与工作报告

| 内容 | 入口 | 访问条件 |
| --- | --- | --- |
| 完整项目 Demo | [播放 / 下载 227.934 秒 MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/M04_demo_200s.mp4) | 私有，需要 HF 仓库访问权限 |
| 实验图库 | [Hugging Face Space](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence) | 私有，需要 HF 仓库访问权限 |
| 37 段真实实验素材 | [逐条视频、任务、条件与结果索引](docs/results/M04/VIDEO_INDEX.md) | 索引公开；视频私有 |
| 项目总览截图 | [M04 overview](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/M04_overview.png) | 私有 |
| 干预对照截图 | [M04 comparison](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/M04_comparison.png) | 私有 |
| 中文工作报告 | [成果、验证、成本与限制](docs/results/M04/WORK_REPORT.md) | 本仓库可读 |
| 原始结项报告 | [CLOSEOUT_REPORT](docs/results/M04/CLOSEOUT_REPORT.md) | 本仓库可读 |
| 机器可读结果 | [结果摘要与源文件校验信息](docs/results/M04/results_summary.json) | 本仓库可读 |
| LeRobotDataset v3 | [Corrections development dataset](https://huggingface.co/datasets/jlai300/vla-forge-m04-corrections-dev) | 私有，需要 HF 仓库访问权限 |

视频规格：**1920 × 1080、H.264、yuv420p、30 FPS、43,383,032 bytes**。完整视频已通过解码和 37 条来源视频的哈希验证，包含成功与失败案例。所有机器人画面来自真实 LIBERO/MuJoCo 仿真。

私有链接不等于公开发布，也不会因出现在 README 中而改变访问权限。图库和视频的云端原始文件保持不变。

## 实际工作流程

```text
M02 真实失败轨迹
  → 来源 Episode / 动作 / 状态 / Hash
  → 恢复参考物理状态并验证一致性
  → 受控干预 + 冻结 SmolVLA 生成新的动作
  → MuJoCo 执行 + Success 信号
  → 独立物理重放两次
  → 去重、来源分组与动作/观测对齐
  → LeRobotDataset v3 + 视频 + 实验回执
```

技术栈：官方 `lerobot/smolvla_libero`、LeRobot、LIBERO / robosuite / MuJoCo、PyTorch、EGL 双相机离屏渲染、RTX 3090 24 GB。复用冻结的模型、预处理与动作协议，没有通过更换模型来解释本次加速。

## 纠正数据：哪些已经验证

| 项目 | 实测结果 |
| --- | --- |
| 新产生且去重的 M04 纠正序列 | **11 条** |
| 独立物理重放 | 每条新纠正 **2 次通过** |
| 另行标记的历史派生序列 | 1 条 M03 派生轨迹 |
| LeRobotDataset v3 r7 | **12 Episodes / 956 Transitions** |
| 数据来源范围 | LIBERO Goal task 6；M02 init states 6、7 |
| 写入后独立读取的动作 / 状态最大误差 | **0 / 0** |
| 图像编码后平均绝对误差 | 2.190（像素值单位，来自独立数据集审计） |
| 数据编码吞吐 | 26.21 Transitions/s；写入 36.478 s |
| 全部生产尝试 | 79 条登记记录：12 条 Success 记录、67 条失败记录 |
| 验证与失败保留 | 失败日志、原始轨迹、视频与来源回执保留 |

**12 条 Success 登记记录不等于 12 条新的独立纠正。** 按动作序列去重并独立验证后，新 M04 纠正为 11 条；数据集的第 12 条来自单独标记的 M03 派生轨迹。79 条尝试属于开发搜索过程，不能作为标准测试集成功率。

达到至少 10 条新纠正的数量要求，尚未达到 30–50 条扩展目标。当前仅覆盖一个任务、两个来源初始状态，**不支持跨任务泛化、模型训练提升或真实机器人部署效果的结论**。所有样本仅用于开发；来源及派生场景应整体排除出后续 M06 锁定测试集。本阶段没有执行 M06。

## 推理加速：保持 FP32 与完整动作输出

18 组配对比较来自 **3 个真实观测 × 6 个噪声种子**。计时覆盖 GPU 同步后的完整原始动作块，不含前处理和物理仿真。

| 指标 | Eager FP32 | CUDA Graph FP32 |
| --- | ---: | ---: |
| P50 | 391.74 ms | **56.39 ms** |
| P95 | 398.84 ms | **61.85 ms** |
| P99 | 399.07 ms | **61.87 ms** |
| 预热后 P50 加速比 | 1.00× | **6.95×** |
| 18 组动作块最大绝对差异 | — | **0** |
| 18 组随机数状态一致 | — | **全部一致** |

动态视觉前缀保留 eager 路径，捕获可复用的注意力与去噪计算。保持模型权重、FP32 精度、去噪步骤和动作块配置不变。BF16 / TF32 的探索结果未作为通过正确性验收的优化发布。

**实际物理执行验证：** 在同一冻结来源的 MuJoCo 续跑中，原始版本与 CUDA Graph 版本均在 **21 个纠正动作后 Success=True**；动作、关节位置与速度、末端位姿、双相机帧的最大差异均为 0。[物理对照回执](docs/results/M04/cuda_graph_physical_comparison.json)

**冷启动与适用范围：** 首次图捕获约 **1.82 s**。本次短续跑的冷启动策略循环为 6.965 s，历史 eager 参考为 5.558 s；这不是同时执行的端到端性能对照，也未证明端到端加速。默认保留 eager，CUDA Graph 用于能够持续复用 sampler 的场景；约 6 个动作块的摊销估计只适用于本次微基准。[完整性能回执](docs/results/M04/cuda_graph_benchmark.json)

## 资源、完整性与当前限制

- 已登记的生产策略循环累计 **0.573466 GPU-hours**，每条新可信纠正约 **0.052133 GPU-hours**。包含失败尝试，但不包含完整加载、前缀恢复、独立重放、编码和空闲时间，**不是完整生产成本或云账单**。
- 生产策略循环峰值 PyTorch allocated memory 为 0.898 GiB；CUDA Graph 微基准峰值为 981,761,024 bytes。两者都不等于进程总显存。
- 237 个登记制品引用中无文件缺失；有 **1 条历史失败尝试的 JSON 哈希异常**保留。其动作数组与对应 NPZ 相同，但哈希异常仍然存在。
- 云端结项验证：**82 passed，1 条上游 Starlette 弃用警告**；Ruff 与编译检查通过。这是云端源码版本的验证记录，不能冒充公开 main 分支 CI。
- 上游精确素材版本的再分发许可仍待确认。软件许可证不能自动代替全部模型、数据、纹理与网格的权利核查。
- 没有开始 M05 训练；没有声称纠正数据已经带来模型成功率提升。

## 复现与代码版本

权威研发目录为云端 `/data/VLA-Forge/project/robot-vla`；完整证据为 `/data/VLA-Forge/Evidence/M04`。大型模型、视频与轨迹位于外部制品目录，没有加入 Git。

```bash
# 在已配置的授权 RTX 3090 云环境中运行
bash /data/VLA-Forge/Evidence/M04/reproduce.sh audit
bash /data/VLA-Forge/Evidence/M04/reproduce.sh benchmark

# 可选：新建一条性能验证续跑；必须使用未用过的 ID
M04_RUN_ID=my_unique_graph_check \
  bash /data/VLA-Forge/Evidence/M04/reproduce.sh rollout
```

- 经验证的算法源码提交：`1836156f8f4ef65014237fed5c6a66f00d23d076`。
- 本次文档更新前的云端源码版本：`0dc9a02ab601235e2571dc9ab37800a8b1105f8f`。
- 两个版本属于云端 `codex/m04-verified-corrections-3090` 分支；M04 源码历史尚未同步至 GitHub。本次发布的是文档与文本证据，不能把公开 main 当作已包含全部 M04 实验代码。
- 已发布的 M02 源码：[codex/m02-libero40-benchmark](https://github.com/Benjamindaoson/VLA-Forge/tree/codex/m02-libero40-benchmark)。
- [精确依赖版本](docs/results/M04/dependency_versions.json)、[复现脚本快照](docs/results/M04/reproduce.sh)、[实验快照与来源哈希](docs/results/M04/results_summary.json)。

## 工程结构与早期原型

`src/vla_forge/` 提供失败案例、干预协议、纠正数据验证、来源分组、存储及研究 API；云端 `benchmarks/robot-vla-engine/` 接入真实 SmolVLA/LIBERO Runner、重放验证、数据导出与性能实验。

仅运行早期工程原型：

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev,serve]"
.venv/bin/python -m pytest tests -q
.venv/bin/vla-forge --help
```

早期原型、CLI 命令生成与合成示例不能替代上面的真实实验。Windows 可使用 `.venv\Scripts\python.exe`；正式 GPU 实验仍在云端运行。

- [架构说明](docs/ARCHITECTURE.md) · [实验协议](docs/EXPERIMENT_PROTOCOL.md) · [安全边界](docs/SAFETY.md)
- [README 更新前的历史快照](docs/history/README_before_M04_showcase_20261011.md)：保留早期状态，已不代表当前云端实验进度。

<details>
<summary>展开全部 37 段真实实验视频与条件（私有素材）</summary>

| 来源 Episode | Suite / Task | 结果 | 实验条件 | 视频 |
| --- | --- | --- | --- | --- |
| M02-spatial-t00-e01 | spatial / 0 | Success | frozen SmolVLA baseline | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/01_M02-spatial-t00-e01.mp4) |
| M02-spatial-t00-e02 | spatial / 0 | Failure | frozen SmolVLA baseline | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/02_M02-spatial-t00-e02.mp4) |
| M02-object-t00-e00 | object / 0 | Success | frozen SmolVLA baseline | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/03_M02-object-t00-e00.mp4) |
| M02-object-t00-e03 | object / 0 | Failure | frozen SmolVLA baseline | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/04_M02-object-t00-e03.mp4) |
| M02-goal-t00-e00 | goal / 0 | Success | frozen SmolVLA baseline | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/05_M02-goal-t00-e00.mp4) |
| M02-goal-t02-e03 | goal / 2 | Failure | frozen SmolVLA baseline | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/06_M02-goal-t02-e03.mp4) |
| M02-long-t00-e01 | long / 0 | Success | frozen SmolVLA baseline | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/07_M02-long-t00-e01.mp4) |
| M02-long-t00-e00 | long / 0 | Failure | frozen SmolVLA baseline | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/08_M02-long-t00-e00.mp4) |
| m03-g09-goal-t06-init07_baseline | goal / 6 | Failure | baseline | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/09_m03-g09-goal-t06-init07_baseline.mp4) |
| m03-g09-goal-t06-init07_brightness | goal / 6 | Success | brightness | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/10_m03-g09-goal-t06-init07_brightness.mp4) |
| m03-g01-spatial-t05-init00_baseline | spatial / 5 | Failure | baseline | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/11_m03-g01-spatial-t05-init00_baseline.mp4) |
| m03-g01-spatial-t05-init00_brightness | spatial / 5 | Failure | brightness | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/12_m03-g01-spatial-t05-init00_brightness.mp4) |
| m03-g02-spatial-t05-init01_baseline | spatial / 5 | Failure | baseline | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/13_m03-g02-spatial-t05-init01_baseline.mp4) |
| m03-g02-spatial-t05-init01_occlusion | spatial / 5 | Failure | occlusion | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/14_m03-g02-spatial-t05-init01_occlusion.mp4) |
| m03-g03-long-t00-init00_baseline | long / 0 | Failure | baseline | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/15_m03-g03-long-t00-init00_baseline.mp4) |
| m03-g03-long-t00-init00_brightness | long / 0 | Failure | brightness | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/16_m03-g03-long-t00-init00_brightness.mp4) |
| m03-g04-long-t00-init02_baseline | long / 0 | Failure | baseline | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/17_m03-g04-long-t00-init02_baseline.mp4) |
| m03-g04-long-t00-init02_occlusion | long / 0 | Failure | occlusion | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/18_m03-g04-long-t00-init02_occlusion.mp4) |
| m03-g05-long-t04-init00_baseline | long / 4 | Failure | baseline | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/19_m03-g05-long-t04-init00_baseline.mp4) |
| m03-g05-long-t04-init00_brightness | long / 4 | Failure | brightness | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/20_m03-g05-long-t04-init00_brightness.mp4) |
| m03-g06-long-t04-init01_baseline | long / 4 | Failure | baseline | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/21_m03-g06-long-t04-init01_baseline.mp4) |
| m03-g06-long-t04-init01_occlusion | long / 4 | Failure | occlusion | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/22_m03-g06-long-t04-init01_occlusion.mp4) |
| m03-g07-goal-t04-init04_baseline | goal / 4 | Failure | baseline | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/23_m03-g07-goal-t04-init04_baseline.mp4) |
| m03-g07-goal-t04-init04_brightness | goal / 4 | Failure | brightness | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/24_m03-g07-goal-t04-init04_brightness.mp4) |
| m04-diverse-goal-t06-init06-seed84182 | goal / 6 | Success | input_brightness_gain_1.1_steps0-50_camera_wrist | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/25_m04-diverse-goal-t06-init06-seed84182.mp4) |
| m04-goal-t06-e07-brightness115-first50-seed84075 | libero_goal / 6 | Success | input_brightness_gain_1.15_first50 | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/26_m04-goal-t06-e07-brightness115-first50-seed84075.mp4) |
| m04-goal-t06-e07-wrist110-first50-seed84068 | goal / 6 | Success | input_brightness_gain_1.1_steps0-50_camera_wrist | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/27_m04-goal-t06-e07-wrist110-first50-seed84068.mp4) |
| m04-goal-t06-e07-wrist110-first50-seed84071 | goal / 6 | Success | input_brightness_gain_1.1_steps0-50_camera_wrist | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/28_m04-goal-t06-e07-wrist110-first50-seed84071.mp4) |
| m04-goal-t06-e07-wrist110-first50-seed84072 | goal / 6 | Success | input_brightness_gain_1.1_steps0-50_camera_wrist | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/29_m04-goal-t06-e07-wrist110-first50-seed84072.mp4) |
| m04-goal-t06-e07-wrist110-first50-seed84088 | goal / 6 | Success | input_brightness_gain_1.1_steps0-50_camera_wrist | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/30_m04-goal-t06-e07-wrist110-first50-seed84088.mp4) |
| m04-goal-t06-e07-wrist110-first50-seed84092 | goal / 6 | Success | input_brightness_gain_1.1_steps0-50_camera_wrist | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/31_m04-goal-t06-e07-wrist110-first50-seed84092.mp4) |
| m04-goal-t06-e07-wrist110-first50-seed84093 | goal / 6 | Success | input_brightness_gain_1.1_steps0-50_camera_wrist | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/32_m04-goal-t06-e07-wrist110-first50-seed84093.mp4) |
| m04-goal-t06-e07-wrist110-first50-seed84094 | goal / 6 | Success | input_brightness_gain_1.1_steps0-50_camera_wrist | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/33_m04-goal-t06-e07-wrist110-first50-seed84094.mp4) |
| m04-goal-t06-e07-wrist110-first50-seed84095 | goal / 6 | Success | input_brightness_gain_1.1_steps0-50_camera_wrist | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/34_m04-goal-t06-e07-wrist110-first50-seed84095.mp4) |
| m04-goal-t06-e07-wrist115-first50-seed84067 | goal / 6 | Success | input_brightness_gain_1.15_steps0-50_camera_wrist | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/35_m04-goal-t06-e07-wrist115-first50-seed84067.mp4) |
| m04-spatial-t05-e07-brightness115-first50-seed82057 | spatial / 5 | Failure | input_brightness_gain_1.15_steps0-50 | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/36_m04-spatial-t05-e07-brightness115-first50-seed82057.mp4) |
| m04-long-t04-e04-brightness115-first50-seed85044 | long / 4 | Failure | input_brightness_gain_1.15_steps0-50 | [MP4](https://huggingface.co/spaces/jlai300/vla-forge-m04-evidence/resolve/main/demo/source_videos/37_m04-long-t04-e04-brightness115-first50-seed85044.mp4) |

</details>
