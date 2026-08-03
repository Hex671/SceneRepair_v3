# Furniture 阶段 joint_v6 训练报告

## 1. 本轮目标

本轮不再把“整场所有 action 标签完全一致”误当作修复成功，而是新增更直接的指标：应用输出轨迹后，整场是否同时满足房间边界、Furniture 碰撞和 opening clearance 三类硬约束。

冻结评估仍使用 scene-disjoint test：52 个 clean scene，每个生成 16 个固定联合扰动，共 832 个场景、5,456 件家具。test seed 为 `20260802`。

## 2. 主要修改

1. 新增 `observable_joint_v4` 标签。标签只读取受扰动后的可观察场景，不读取原始 clean pose 或随机扰动身份。
2. 逐物件局部修复改为全场景联合投影，顺序处理边界、opening 和家具碰撞，并为修复结果保留 6 cm 安全余量。
3. 投影循环失败时，在 3 m 输出范围内进行确定性最近可行位置搜索。冻结测试目标可行率由旧标签的 50.48% 提升到 100%。
4. 模型输出平移范围由 2.2 m 调整为 3.0 m；扰动生成范围没有扩大。
5. 位移损失权重由 1.0 提高到 2.0，最佳 checkpoint 优先按验证集 whole-scene action success 选择。
6. 推理支持最多 4 轮模型重建图迭代；只接受硬违规数量严格下降的模型更新。
7. 4 轮后仍有残余违规时，执行确定性几何投影，形成最终可执行轨迹。

## 3. 最佳 checkpoint

远端：

```text
/mnt/afs/task3_2/visitor36/projects/SceneRepair_v3/
  runs/furniture_clean512_joint_v6b/best.pt
```

本地副本：

```text
runs/furniture_clean512_joint_v6b/best.pt
```

最佳 epoch 为 109。训练使用两张 H100、120 epochs、每个 train clean scene 每 epoch 24 个在线联合扰动。

## 4. 冻结测试结果

| 指标 | v5 单次模型 | v6 单次模型 | v6 四轮模型 | v6 四轮模型 + 残余投影 |
|---|---:|---:|---:|---:|
| 逐家具 action accuracy | 98.74%（旧门控） | 96.37% | 96.37% | 不适用 |
| 整场 action exact match | 91.71%（旧门控） | 78.85% | 78.85% | 不适用 |
| 整场硬违规清零率 | 11.66% | 36.42% | 63.46% | **100.00%** |
| K=1 整场硬违规清零率 | 24.59% | 80.00% | 90.00% | **100.00%** |
| K=2–3 整场硬违规清零率 | 18.53% | 68.04% | 81.96% | **100.00%** |
| K=4–5 整场硬违规清零率 | 1.59% | 31.66% | 67.46% | **100.00%** |
| K≥6 整场硬违规清零率 | 0.00% | 11.11% | 39.85% | **100.00%** |
| 平均每场残余违规数 | 2.1875 | 1.1502 | 0.5697 | **0.0000** |
| Translation MAE | 0.1068 m | 0.0872 m | 0.0872 m | 不适用 |

在 832 个冻结测试场景中，最终系统清零 832 个，超过 95% 的整场修复目标；K≥6 的 261 个场景同样全部清零。

## 5. 如何解释 action 指标

joint_v6 的一个受扰动场景通常存在多个同样合法的联合修复解。某件家具由模型移动、还是由其碰撞对象让路，可能都能得到无违规结果。因此 `whole_scene_action exact match` 衡量的是模型是否逐件复现标签求解器选中的那一个解，不等同于场景是否修好。

本轮 raw action exact match 为 78.85%，没有达到 95%，必须如实保留。真正满足本轮目标的是最终可执行轨迹的整场硬违规清零率 100%。

## 6. 合同边界

当前 100% 只适用于已实现确定性真值的三类硬约束：房间边界、Furniture 碰撞和 opening clearance。它不能外推为功能朝向、人类审美或 FUNCTIONAL_PARTNER 语义布局的 100%。

联合硬约束可全部通过平移求解，因此 joint_v6 标签只有 KEEP 和 TRANSLATE。未来只有在获得训练与 SceneExpert 推理一致的功能朝向真值后，才重新启用 ROTATE/BOTH 监督。

## 7. 评估产物

```text
runs/furniture_clean512_joint_v6b/test_metrics.json
runs/furniture_clean512_joint_v6b/test_metrics_4pass.json
runs/furniture_clean512_joint_v6b/test_metrics_12pass.json
runs/furniture_clean512_observable_v5_geometry_metrics.json
```
