# Furniture 阶段 constraint-aware v7 验收报告

## 1. 结论

Furniture 阶段完整模型已在冻结测试集上达到整场硬违规清零率 100%，超过 95% 目标。

这里的“完整模型”有严格定义：

```text
受扰动场景
  -> v7c Relation-aware Graph Transformer
  -> 每件家具的 action logits 与初始 delta
  -> 模型内 HardConstraintRefinementLayer（无可训练参数）
  -> 每件家具的最终 action 与 delta_x / delta_y / delta_yaw
```

`HardConstraintRefinementLayer` 是 `FurnitureRepairNetwork` 持有的正式推理层，并由 `repair_batch()` 单一部署接口调用。它不是评估脚本追加的可选投影。神经主干 raw、完整模型和模型外安全兜底必须继续分别统计，不得混用。

## 2. 冻结验收合同

| 项目 | 值 |
|---|---:|
| Clean scenes | 512 |
| Scene-disjoint train / val / test | 409 / 51 / 52 |
| Test corruptions per clean scene | 16 |
| Frozen test scenes | 832 |
| Test furniture objects | 5,456 |
| Test seed | `20260802` |
| 最大 Furniture 数 | 17 |
| 输出平移范围 | 3.0 m |
| 验收条件 | 房间边界、Furniture 碰撞、Opening clearance 同时为零 |

当前验收不包含 FUNCTIONAL_PARTNER 的功能朝向、人类审美或其他尚无确定真值的语义布局约束。

## 3. v7c 训练

v7c 从 v7b 最佳 checkpoint 微调，主要变化是 25 cm 联合可行标签和强化的可微几何损失：

- hidden dimension 128，4 heads，2 layers；
- 双 H100 训练；
- 50 epochs，最佳 checkpoint 为 epoch 41；
- learning rate `1e-4`；
- geometry loss weight 20；
- geometry clearance 0.12 m；
- train / val samples per scene 为 24 / 8。

Checkpoint：

```text
runs/furniture_clean512_joint_v7c/best.pt
```

SHA256：

```text
21A440DEB77D34240DA1EAA0E2F4966E5E47D6B7209E5284A5D038297DA757A3
```

## 4. 冻结测试结果

| K 分组 | 场景数 | 神经主干 1 pass | 神经主干 4 pass | Constraint-aware 完整模型 |
|---|---:|---:|---:|---:|
| 0 | 19 | 100.00% | 100.00% | 100.00% |
| 1 | 6 | 100.00% | 100.00% | 100.00% |
| 2-3 | 139 | 79.86% | 89.93% | 100.00% |
| 4-5 | 337 | 61.72% | 87.83% | 100.00% |
| 6-7+ | 331 | 37.76% | 79.46% | 100.00% |
| **全部** | **832** | **56.37%** | **85.22%** | **100.00%** |

完整模型的额外审计结果：

- 832/832 整场硬违规清零；
- 残余硬违规总数为 0；
- refinement 未收敛场景为 0；
- 平均 refinement 迭代数为 2.059；
- 最终最大单物件平移为 2.5625 m，未超过 3.0 m 合同；
- K>=6 场景为 331/331 清零。

神经主干逐家具 action accuracy 为 95.12%，整场 action exact match 为 73.56%，raw Translation MAE 为 0.1484 m。action exact match 只衡量是否复现监督求解器选中的同一个联合解；存在多个合法联合解时，它不等于整场修复是否成功。

## 5. 为什么完整模型高于 raw 主干

同一个碰撞通常有多个合法修复方向。逐物件 L1 回归会在多种可行解之间产生连续均值，几厘米的幅值误差也足以留下硬碰撞。v7c 的 4 轮纯神经迭代已将清零率从 56.37% 提升到 85.22%，但继续增加迭代不能可靠跨过严格的零违规边界。

无参数约束层保留神经网络给出的动作与轨迹作为初始解，只使用运行时本来就可获得的房间、bbox、yaw 和 opening clearance 几何，将残余结果细化到可行域。该层不会读取隐藏 clean pose、扰动身份或训练标签。

## 6. 产物

```text
runs/furniture_clean512_joint_v7c/best.pt
runs/furniture_clean512_joint_v7c/test_metrics_1pass.json
runs/furniture_clean512_joint_v7c/test_metrics_4pass.json
runs/furniture_clean512_joint_v7c/test_metrics_constraint_aware_v1.json
runs/furniture_clean512_joint_v7c/test_metrics_constraint_aware_runtime_v1.json
docs/prediction_examples_24_v7c.json
docs/Furniture模型预测修复示例.html
docs/prediction_gallery_v7c_desktop.png
docs/prediction_gallery_v7c_mobile.png
docs/SceneExpert_Furniture推理接口_v1.md
```

## 7. 部署要求

SceneExpert 必须调用 `FurnitureRepairNetwork.repair_batch()`，而不是只调用 `forward()` 后直接执行 raw delta。部署端还必须校验：

- `furniture_constraint_aware_model_v1` 推理合同；
- `furniture_constraint_refinement_v1` 细化合同；
- checkpoint、词表和图 schema hash；
- room-local 坐标系、bbox 和几何 yaw 合同；
- 3.0 m 最大输出范围。

若将来增加新的硬约束，必须升级 refinement schema、重新冻结测试集并重新验收，不能沿用本报告的 100% 数字。

## 8. Runtime 集成复验

新增 `FurnitureRepairEngine` 与 `tools.infer_furniture` 后，模型内 refinement 增加了 checkpoint `ActionRange` 门禁和 learned-proposal 缩放保护。变更后重新运行同一冻结测试，结果仍为 832/832、K>=6 为 331/331、残余违规 0、最大平移 2.5625 m。

真实 accepted JSON 已完成 CPU smoke test；越界、碰撞和 opening 三类故障 demo 已分别在本机 CPU 与 `sceneexpert-cci` H100 BF16 上完成 5 -> 0 硬违规修复。超出 3 m 动作合同的输入会被显式拒绝，不计入成功结果。
