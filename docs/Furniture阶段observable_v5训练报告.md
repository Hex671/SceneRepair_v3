# Furniture 阶段 observable_v5 训练报告

## 1. 达标定义

本轮“整体准确率”采用严格的 whole-scene action success：一个场景内所有家具的 action 均正确才计为成功。评估固定使用 scene-disjoint test 的 52 个 clean scene，每个生成 16 个可复现联合扰动，共 832 个场景、5,456 个家具。

## 2. 结果

最佳 checkpoint：

```text
/mnt/afs/task3_2/visitor36/projects/SceneRepair_v3/
  runs/furniture_clean512_observable_v5/best.pt
```

| 指标 | 模型 raw argmax | 约束门控推理 |
|---|---:|---:|
| 逐家具 action accuracy | 97.31% | 98.74% |
| 整场 action accuracy | **83.65%** | **91.71%** |
| K=6/7+ 整场 accuracy | **78.23%** | **93.55%** |
| Translation MAE | 0.1068 m | 0.1068 m |
| Yaw MAE | 0.0709 rad | 0.0709 rad |

raw 模型和完整推理系统均超过用户要求的 70% 整场准确率。完整报告：

```text
runs/furniture_clean512_observable_v5/test_metrics_raw.json
runs/furniture_clean512_observable_v5/test_metrics.json
```

## 3. 有效改动

1. 随机数只负责产生 corrupted candidate，不再直接决定监督 action。
2. `observable_v3` 从当前场景计算硬违规，并规划最小可验证平移或旋转；标签不读取 clean pose 和“谁被扰动”的隐藏随机事件。
3. 任何实质边界、碰撞或 opening 违规都必须 CHANGE；求解器暂时不能一次消除全部违规时，监督当前最佳下降方向，允许迭代修复。
4. 动作头接收从既有边特征聚合的 12 维约束摘要：四侧 margins、碰撞/opening 穿透深度、escape 和数量。
5. 四分类损失增加 move/rotate 分量级辅助损失。
6. 推理可选确定性约束门控：无硬违规强制 KEEP；有硬违规禁止 KEEP，动作子类型与 delta 仍由模型决定。

## 4. 数据合同边界

observable_v5 当前只覆盖具有确定性几何真值的约束：房间边界、Furniture 碰撞和 opening clearance。训练自然产生约 43% KEEP、55% TRANSLATE、2% ROTATE；这些约束下平移总能构成有效最小修复，因此没有合理的 BOTH 目标。

该 83.65%/91.71% 不能外推为尚未定义真值的功能朝向或人类审美布局准确率。只有获得训练/推理一致的 desired orientation/gap 标注后，才应新增 FUNCTIONAL_PARTNER 旋转与 BOTH 监督，不能为了类别齐全人工制造标签。
