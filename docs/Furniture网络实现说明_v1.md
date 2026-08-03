# Furniture 网络实现说明 v1

## 1. 实现状态

设计文档中的 Furniture v1 网络已经在 `scene_repair_v3` 包中实现。当前范围包含图数据合同、图构建、网络前向、动作解码、训练损失、checkpoint 合同、accepted/SceneExpert scene JSON 适配、在线联合扰动数据集、scene-disjoint 清单、单卡/DDP 训练入口，以及 constraint-aware JSON-to-JSON 推理引擎。SceneExpert 进程内调用胶水和运行时事务提交仍属于后续集成工作。

训练数据与执行合同见 `docs/Furniture阶段训练数据构造与训练计划_v1.md`。

## 2. 代码映射

| 文件 | 职责 |
|---|---|
| `scene_repair_v3/vocabulary.py` | 冻结无碰撞词表、UNK 和 SHA-256 合同 |
| `scene_repair_v3/contracts.py` | 三类节点、四类边、disjoint batch、动作范围和张量校验 |
| `scene_repair_v3/geometry.py` | geometric yaw、OBB signed separation、minimum escape、房间 margin |
| `scene_repair_v3/graph.py` | 来源无关场景输入、完整有向图、typed edge set 和 batch collate |
| `scene_repair_v3/functional_partners.py` | 加载 HSSD 冻结 partner 表，按具体资产 ID 构造候选功能边 |
| `scene_repair_v3/data.py` | accepted JSON 适配、逐家具联合随机扰动和监督标签 |
| `scene_repair_v3/dataset.py` | clean scene 冻结清单、scene-disjoint split 和在线数据集 |
| `scene_repair_v3/model.py` | 类型专用编码器、四个 EdgeEncoder、两层关系图 Transformer、双动作头 |
| `scene_repair_v3/repair.py` | 三类可观测硬约束验证与固定的全场景可行化算法 |
| `scene_repair_v3/inference.py` | 严格 checkpoint 加载、批量完整模型推理、场景 JSON 回写与 audit |
| `scene_repair_v3/losses.py` | 四类动作监督、米制 Smooth-L1、周期 yaw loss 和 KEEP 正则 |
| `scene_repair_v3/checkpoint.py` | 模型、词表、坐标系和动作范围的严格保存/加载校验 |
| `tools/train_furniture.py` | 单卡或双卡 DDP 的 BF16 训练、验证、指标和 checkpoint |
| `tools/infer_furniture.py` | SceneExpert Furniture JSON 的正式批量推理 CLI |

网络不包含旧版的 TASK/WALL/VIOLATION/RELATION_GROUP 节点，也不包含 scene route、candidate query、mixture、violation decoder、validity/ranking head。

默认网络宽度冻结为：

```text
hidden_dim       = 128
attention_heads  = 4
head_dim         = 32
edge_dim         = 32
geometry_dim     = 48
category_dim     = 32
behavior_dim     = 48
layers           = 2
```

默认配置约 54 万个可训练参数。128维与4头同步缩放后仍保持每头32维；两层传播职责不变。

## 3. 固定张量合同

### 3.1 节点

| 类型 | 连续输入 |
|---|---|
| ROOM | `length_m, width_m` |
| OPENING | normalized clearance `center_xyz, size_xyz`，interior normal XY，共 8 维及逐字段 mask |
| FURNITURE | normalized `x,y`，`sin(yaw),cos(yaw)`，normalized bbox `width,depth,height`，共 7 维及逐字段 mask |

Furniture 另有 category embedding，以及 `family embedding + function multi-hot + 两个分支 validity`。HSSD ID 和对象 ID 不进入网络。

HSSD ID 可作为 `FurnitureInput.hssd_id` 传入图构建前的冻结关系查表，但不会写入节点张量。`configs/functional_partner_rules_hssd_v1.json` 从 10,963 个 HSSD 资产中冻结了 10,050 个资产自身已有的 partner 标注；无标注资产不继承同 category 规则并产生 0 条功能边。推理前使用 `FrozenFunctionalPartnerRules.apply(scene)` 统一生成边，禁止与手写边混合。完整来源、缺失策略和复现命令见 `docs/FUNCTIONAL_PARTNER冻结规则_v1.md`。

### 3.2 边

四类边分别保存在独立 `TypedEdgeSet`，不会先填充进共享槽位：

| 类型 | 连续维度 | 类别字段 |
|---|---:|---|
| ROOM_MEMBERSHIP | 6 | anchor direction/mode |
| FURNITURE_SPATIAL | 8 | 无 |
| FUNCTIONAL_PARTNER | 4 | orientation mode/target strategy |
| OBJECT_OPENING | 4 | 无 |

每条边另有整体 `active` mask。无效可选关系不会进入 attention，也不会被零消息伪装成存在的关系。

## 4. 坐标和归一化

- 坐标合同固定为 `room_local_z_up`，四元数顺序为 `wxyz`，yaw wrap 为 `[-pi,pi)`。
- Furniture 节点位置使用 `x/room_length, y/room_width`。
- bbox width/depth 分别除以 room length/width；高度除以 `max(length,width)`。
- FURNITURE_SPATIAL 的 relative XY 保持 room frame，并分别除以 length/width，不旋转进 source-object frame。
- signed separation 除以 `max(length,width)`；escape XY 分别除以 length/width。
- 动作监督和执行仍使用真实米/弧度，不使用上述输入归一化尺度。

## 5. 动作范围

网络只输出 `tanh` 后的 normalized delta。`ActionRange(max_translation_m, max_yaw_rad)` 没有默认值，训练 loss、checkpoint 和解码必须显式共享同一个实例或等值合同。正式训练前需要在数据集版本中确定唯一范围。

## 6. 运行验证

在项目根目录执行：

```powershell
python -m pytest -q
python -m tools.smoke_furniture_network --max-translation-m 0.5 --max-yaw-deg 60
```

smoke 参数只用于验证接口，不代表最终动作范围决策。

## 7. 词表边界

`configs/furniture_vocab_bootstrap_v1.json` 用于代码联调和测试，其中 family/function 来自已确定语义闭集，category 只含当前常见示例。正式训练前必须统计完整训练/推理 category 闭集，生成新的版本化词表并冻结 SHA-256；不能直接把 bootstrap category 列表当作最终数据合同。

## 8. Constraint-aware 完整推理合同

冻结测试证明，连续回归主干即使经过四轮重建图迭代，整场硬违规清零率仍为 85.22%。因此正式部署模型包含一个由 `FurnitureRepairNetwork` 持有的无参数 `HardConstraintRefinementLayer`：

```text
FurnitureRepairNetwork.repair_batch(
  graph_batch,
  source_scenes,
  ActionRange(3.0, pi),
)
  -> neural action/delta
  -> internal hard-constraint refinement
  -> final per-Furniture action/delta and repaired scenes
```

部署端必须调用 `repair_batch()`；只调用 `forward()` 得到的是可单独审计的 neural raw 输出，不是完整模型输出。细化层只使用当前场景可见的 room boundary、Furniture OBB 和 opening clearance，不读取 clean pose 或扰动身份。

推理合同版本为 `furniture_constraint_aware_model_v1`，细化层版本为 `furniture_constraint_refinement_v1`。冻结 832 场景上的完整验收结果和适用边界见 `docs/Furniture阶段constraint-aware_v7验收报告.md`。
