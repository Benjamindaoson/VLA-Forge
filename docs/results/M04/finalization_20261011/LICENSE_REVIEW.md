# M04 文件级公开许可审查

结论：允许发布原创数值纠正证据和报告；完整视觉数据集、仿真素材、Demo 及含双相机的轨迹维持私有。此结论是基于现有来源证据的发布准入判断，不是关于所有渲染画面是否合法的司法结论。

## 已核查的依据
- LIBERO 固定提交 8f1084e3132a39270c3a13ebe37270a43ece2a01：README 声明代码 MIT、数据集 CC BY 4.0；根 LICENSE 为 MIT。
- HF lerobot/libero-assets 固定 Revision 0b3ea86be5fe169d0fd036ae63d1070ec09e90f6：586 个文件；585 个资源与固定 LIBERO 源码资源逐字节相同，唯一非匹配项是 HF .gitattributes。
- HOPE 原始作者仓库 swtyree/hope-dataset 的 README 明确涵盖纹理网格并声明 CC BY-NC-SA 4.0。LIBERO stable_hope_objects 下的 cream_cheese 是当前 Goal task 6 的目标物体；总体 MIT/CC BY 声明不足以消除更具体来源条款的歧义。
- 场景还涉及 scanned objects、TurboSquid、场景纹理和家具；当前快照没有完整逐物体的原始页面、授权和署名链。

## 文件清单及处理
license_inventory.csv 包含 586 个上游资源文件和 280 个媒体／含图像的轨迹文件，逐项记录路径、SHA256、官方固定版本匹配状态、声明、处理决定和理由。
例：stable_hope_objects/cream_cheese 下 XML/OBJ/MTL/MSH/纹理保留 HOPE 来源限制；turbosquid_objects、scenes、textures 的原始条目授权链未齐。所有列明的 M04 MP4/PNG/NPZ 维持私有。未改动任何原视频。

## 可公开的明确子集
public_numeric_dataset/ 仅含自行产生的机器人 7D 动作、8D 本体状态、成功/终止标记、时间与索引，以及原创来源说明。没有图像、视频、纹理、网格、物体几何、模型权重或第三方完整数据集。以 CC BY 4.0 发布这些原创数值记录，保留 LIBERO 与 SmolVLA 署名；该许可不扩展至被排除的视觉资产。这是数值证据子集，不能声称完整视觉 LeRobot v3 数据已经公开，不能替代 VLA 图像训练输入。

## 完整视觉发布解除条件
补齐 HOPE 各素材的来源/修改/署名与 NC-SA 适用处理；为 scanned、TurboSquid 和场景资源建立精确条目映射及再分发条款。或者在另行授权的未来实验中使用全部明确许可的替代素材重新生成数据。本次不重跑旧实验，不擅自把私人上传改成公开。

官方链接、抓取原文哈希见 license_review.json；抓取原文保存在 license_sources/。原始代码许可证不等同于所有第三方资产的统一许可证。
