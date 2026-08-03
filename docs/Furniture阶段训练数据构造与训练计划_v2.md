# Furniture 阶段训练数据构造与训练计划 v2

本文档替代 v1 中“随机 action 等于监督 action”的部分。网络结构仍以 `Furniture阶段新版网络结构设计_v1.md` 为基础。

## 当前合同

```text
clean512_v1 clean scene
        ↓
逐家具独立随机位姿扰动，仅用于产生 candidate
        ↓
从 candidate 重算 ROOM / OPENING / FURNITURE 和四类边
        ↓
ObservableRepairPlanner 检查边界、碰撞、opening clearance
        ↓
KEEP / TRANSLATE / ROTATE + 最小可验证 delta
        ↓
2-layer relation-aware Transformer + constraint summary
```

监督标签不得读取随机 action、clean pose 或对象是否曾被扰动。无可观察违规为 KEEP；有实质硬违规必须 CHANGE。平移和旋转均能单独消除违规时，选择归一化代价更小者；暂时无法完全求解时输出最佳下降方向，由迭代推理继续修复。

## 模型改动

- 主干保持 128 hidden / 4 heads / 2 layers；
- 从现有 typed edges 聚合 12 维 constraint summary，经 32 维 MLP 后与 Furniture state 拼接；
- action head 仍输出四类兼容接口；
- loss 增加 move/rotate component loss；
- decode 支持 constraint gate，但必须同时报告 raw 和 gated 指标。

## 接受门禁

- scene-disjoint test 至少 800 个固定扰动场景；
- raw whole-scene action accuracy >= 70%；
- gated whole-scene action accuracy >= 70%；
- `K=6/7+` 单独报告；
- 保存 seed、manifest hash、vocabulary hash、checkpoint contract 和完整 confusion matrix。

observable_v5 已满足上述 action 门禁。后续阶段应增加修复后重新验收和 FUNCTIONAL_PARTNER 可靠真值，不得用本轮准确率替代这些尚未实现的指标。
