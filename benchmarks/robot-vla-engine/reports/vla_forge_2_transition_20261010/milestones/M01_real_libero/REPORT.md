# M01：SmolVLA → LIBERO/MuJoCo 真实仿真闭环

**验收结果：PASS。** 已在 RTX 3090 上加载官方 SmolVLA LIBERO checkpoint，完成 3 次 smoke 与 40 次正式物理仿真 rollout。Smoke Gate 与正式覆盖均通过；43/43 个视频可解码，43/43 个视频哈希、43/43 条 rollout 轨迹、43/43 条传入 `env.step` 的动作对应关系，以及 21/21 个模型/骨干文件哈希均由独立复核通过。

## 实测结果

| LIBERO suite / 固定 task 0 | 正式成功 | smoke 成功 |
|---|---:|---:|
| spatial / `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate` | 5/10 (50.0%) | 1/3 |
| object / `pick_up_the_alphabet_soup_and_place_it_in_the_basket` | 6/10 (60.0%) | — |
| goal / `open_the_middle_drawer_of_the_cabinet` | 10/10 (100.0%) | — |
| long / `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket` | 3/10 (30.0%) | — |

**正式合计：24/40（60.0%）**；smoke Spatial 为 1/3。结果只代表每个 suite 固定选取的 task 0，各 10 个 Episode，不等同于 suite 全任务或 LIBERO 40-task 基准成绩。失败 Episode 保留在分母内。

## 运行资源与耗时

- GPU：NVIDIA GeForce RTX 3090（24 GiB）；PyTorch 峰值 allocated 964,068,864 bytes，reserved 1,002,438,656 bytes；运行监测的 `nvidia-smi` 峰值显存 1297 MiB。
- Smoke phase wall time 153.5s；正式 40 集 phase wall time 1957.0s；两阶段合计 2110.5s（35.2 分钟）。43 个完成 Episode 的仿真执行时间合计 1711.3s。
- 推理计时分开报告：`select_action` 包括动作队列读取，chunk generation 由 CUDA 同步 profiler 单独测量；逐 suite 的 mean/p50/p95/p99/max 从已保存的原始 Episode latency 数组重新聚合，见 `acceptance_audit.json`。模型推理存在长尾，p95 小于均值的 Episode 单独保留，p99/max 同时给出。

## 冻结模型与软件版本

- Policy：`lerobot/smolvla_libero@31d453f7edd78c839a8bbc39744a292686daf0de`；权重 SHA-256 `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`。
- VLM：`HuggingFaceTB/SmolVLM2-500M-Video-Instruct@7b375e1b73b11138ff12fe22c8f2822d8fe03467`；权重 SHA-256 `b9bfd456c9472c0acd5719d6e514c4b859891af205ee1a736552fd3497b8b0c3`。
- Runtime：lerobot==0.6.2, torch==2.7.1+cu128, transformers==5.5.4, huggingface-hub==1.33.0, hf-libero==0.1.4, mujoco==3.8.1, robosuite==1.4.0, av==15.1.0, safetensors==0.8.0；CUDA 12.8，MuJoCo EGL。
- 固定 action chunk 为 50；由 checkpoint processor 生成动作块，通过相对控制逐动作调用 MuJoCo `env.step`，并保存成功信号与轨迹。LIBERO 官方 API 将 Long suite 命名为 `libero_10`。

## 兼容处理与证据边界

- checkpoint processor 将实测 LIBERO 双目画面映射到 `camera1/2`；第三相机未伪造。
- LIBERO 观测的 8 维 state 与 checkpoint JSON 声明的 6 维模型 feature 不一致；加载的 checkpoint normalizer 实际保存 8 维 state statistics。本次以实测 normalizer 宽度为准保留 pose 与 gripper 全部 8 维；固定版本 `SmolVLAPolicy.prepare_state` 会把状态补齐到 max state width 后送入模型。
- 每条 video 的哈希均与 ledger 一致，所有视频均已完整解码（75–520 帧）；保存的 action_trace 与 trajectory actions 对 43 个 Episode 全部逐元素相等，向量是 runner 传给 `env.step` 的实际命令。MuJoCo 内部 actuator force 未单独采集。
- 两次早期 smoke 修复阶段留下 3 条状态宽度错误尝试，另有一次运行暴露 base Conda 的 cuDNN 动态库覆盖；没有删除这些失败尝试。复现入口已清除该环境变量。最终 CSV 的 43 条 Episode 全部 complete，runtime error rows 为 0。详见 `episode_attempts.jsonl` 与 `run.log`。

## 复核、代码提交与复现

- 独立 artifact audit：`acceptance_audit.json`（43/43 Episode ID 覆盖、视频解码/哈希、轨迹/action 对齐、21 个模型文件、LIBERO metadata 哈希全部通过）。
- 测试与 lint 记录：`validation.json`，包括 RTX 3090 主机完整项目 214 passed 和 M01 专项 7 passed。
- Episode 级指标：`episodes.csv`；聚合指标：`metrics.json`；完整模型/数据/依赖/源码 provenance：`manifest.json`；真实视频：`videos/`；SQLite Registry 与轨迹：`registry/`；完整日志：`run.log`。
- 实验源码 commit `e99cf86d9322c472c4cb62f3bc431a1f45c4d12c`；本地 Git branch `feat/observable-baseline`。完整项目测试 `214` passed；M01 专项 7 passed；远端 Ruff 与 py_compile 通过。远端原 checkout HEAD `f4037c32b87f56a53d005a8f02653a173c90d3a5` 与本地工作仓库历史不同，Manifest 同时保存了 requested source revision 和部署文件 SHA-256。
- 在已配置同一 Linux 环境和 `/data` checkpoint/数据盘的服务器复现：`bash reports/vla_forge_2_transition_20261010/milestones/M01_real_libero/reproduce.sh`。脚本先运行 3 次 smoke 并检查 Gate，然后才允许正式 40 集。
- 43 个最终 completed Episode 的 MP4、Episode NPZ 与 SQLite Registry 已保存在本地报告目录及授权服务器 `/data`；`videos/` 另有 3 个 0-step 失败尝试片段，由 `episode_attempts.jsonl` 标识。为避免把 rollout 二进制制品提交进代码 Git，视频与 Registry 生成目录由仓库 `.gitignore` 排除，Manifest/Audit 保留最终证据的逐项哈希与位置。

## 简历描述（严格限于本次证据）

> 将官方 SmolVLA 与 LeRobot action-chunk 推理接入 LIBERO/MuJoCo，在 RTX 3090 上完成 43 次真实仿真闭环 rollout，覆盖 Spatial、Object、Goal、Long 各 1 个固定任务；正式集成功率 24/40（60%），并建立 Episode 级视频、动作轨迹、资源指标与模型/数据哈希的可复核证据链。

M01 到此停止；待用户检查成果后再决定是否进入 M02。
