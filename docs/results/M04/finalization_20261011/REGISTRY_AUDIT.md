# M04 Registry 与纠正监督审计

状态：数据工程检查通过；历史异常已经隔离，原始证据未修改。

- r7 的 12 Episode / 956 transitions 保持逐文件 SHA256 不变。
- 11 条新纠正与 1 条 M03 派生轨迹逐条核实：源 M02 为失败；任务及 init state 对应；100 步前缀与实际 env_step_argument 精确一致；7D 动作有限且在 [-1,1]；8D 预动作状态与数据集精确一致；成功与 done 仅在最后一帧。
- 11 条新纠正对应 22 份独立物理重放的成功 receipt，NPZ 哈希、动作数组和恢复容差全部通过。
- 新鲜审计没有重新运行以上仿真；引用已经存在的、重新核验了哈希的物理重放证据。旧图像逐帧验证 receipt 也冻结了哈希。

异常：m04-g09-fresh-brightness-115-first50-01 的两行失败记录复用了同一 ID 和 JSON 路径，时间及哈希不同。当前 JSON 对应后一次记录，不能恢复前一次文件的可信字节。两行在 quarantine_manifest.json 逻辑隔离，禁止训练纳入；不删除失败、不篡改旧哈希、不把动作数组相等当成元数据哈希修复。r7 未包含该 ID，11 条新纠正不受影响。

来源组纠正：此前声称预留 M06 的 goal:6:0/1/4/6、object:6:1、spatial:5:7 共 6 组被开发尝试使用。全部退役，不再称为独立测试。所有失败尝试的源组及派生场景也划入开发范围。替代的 40 任务 × init 40–49 共 400 个条件已冻结文件及状态哈希、任务配置哈希和 seed 95000–95399；没有来源组或状态字节与旧开发状态重叠。未运行这些测试，也不声称它们从未出现在基础模型预训练中。

机器可复核文件：registry_audit.json、quarantine_manifest.json、training_freeze.json、source_groups.json、independent_test_freeze.json。
