# Furniture 阶段网络改进措施 v1

## 1. 文档目的

本文档用于记录对当前 Furniture 阶段网络的审查结论、已确定的修改措施及其验证要求。讨论中的猜想和候选方案必须与最终决策分开记录，只有状态为“已确定”的条目才可作为后续实现依据。

当前审查基线：

- 网络：`FurnitureRepairModel`
- 模型 schema：`5.0`
- 特征版本：`local-geometry v3`
- 旧版实际计算图（历史对照）：`docs/visualizations/furniture_model_v2_actual_architecture_legacy.svg`
- 当前部署边界：Shadow；神经网络之外仍包含规则候选、执行器、`FurnitureVerifier`、Critic、物理验证和事务回滚

## 2. 记录规则

每项措施使用以下状态之一：

- `待讨论`：问题已提出，但尚未确定是否修改或如何修改。
- `已确定`：修改目标、实现边界和验证方式已经达成一致。
- `已拒绝`：经过讨论决定不采用，并记录理由。
- `已实现`：代码和配置已经完成修改，但尚未完成全部验证。
- `已验证`：实现通过约定的离线、Shadow 或消融验证。

每项“已确定”措施至少需要写明：现状、问题证据、修改方案、影响范围、兼容性、验证实验和验收标准。

## 3. 当前网络基线

当前节点初始状态由以下分支直接相加得到：

```text
NodeMLP(48D geometry)
+ NodeTypeEmbedding(7 -> 256)
+ CategoryEmbedding(512 hash buckets -> 256)
+ HSSDLabelMLP(82D -> 256, gated by bundle-valid)
= H0
```

`H0` 随后进入四层 typed sparse graph Transformer。网络使用场景路由与 OOD 头、固定 learned candidate queries、对象 change/pose 解码器、violation 解码器以及 candidate validity/quality/ranking 头产生候选分数。完整执行和最终安全判定不在网络内部。

## 4. 已确认的编码事实

### 4.1 Node type

节点类型不是显式的 7 维 one-hot，而是整数索引，经 `Embedding(7, 256)` 转换为可学习向量：

| ID | Node type |
|---:|---|
| 0 | `TASK` |
| 1 | `ROOM` |
| 2 | `WALL` |
| 3 | `OPENING` |
| 4 | `FURNITURE` |
| 5 | `VIOLATION` |
| 6 | `RELATION_GROUP` |

### 4.2 Category identity

当前 `category_ids` 不是 HSSD 469 类词表中的精确类别 ID。图构建器先取得节点类别字符串，再计算：

```text
int(SHA256(category_string)[:8], 16) mod 512
```

结果作为 `Embedding(512, 256)` 的索引。Furniture 节点的类别字符串取自 `CanonicalObject.normalized_category`，在当前 HSSD 数据中通常对应 HSSD `raw_category`；其他输入源可以来自适配器提供的 `metadata.category` 或 `object_type`。

数据管线还保存了精确的 `raw_category_ids`，但当前 `model.forward()` 不消费该字段。HSSD asset ID 也不直接作为该 embedding 的索引。

## 5. 改进措施清单

| 编号 | 模块 | 状态 | 当前问题 | 最终措施 | 验证状态 |
|---|---|---|---|---|---|
| FUR-NET-001 | Category identity encoding | 待讨论 | 512 桶哈希已确认存在大量真实碰撞，且已落盘的精确 `raw_category_ids` 未被网络消费 | 待确定；倾向改为按 node type 分域的无碰撞显式词表 | 统计完成，结构与消融未开始 |
| FUR-NET-002 | Node type encoding | 已确定 | 显式 one-hot 与 embedding lookup 在线性映射下等价 | 使用 `Embedding(3, 256)`；不做显式 one-hot 改写 | 设计已确认；待随新 schema 实现 |
| FUR-NET-003 | Node ontology | 已确定 | `TASK` 无独立任务差异；`RELATION_GROUP/VIOLATION` 与边重复；当前 WALL 只有未训练代码容量 | 删除 `TASK / RELATION_GROUP / VIOLATION / WALL`，保留 3 类实体/容器节点 | 设计已确认；实现与消融未开始 |
| FUR-NET-004 | Furniture spatial graph | 已确定 | 固定 3m、最多 16 邻居的局部图只覆盖当前 v5 全部有序家具对的约 33%，多数 Furniture 子图不连通 | Furniture–Furniture 改为无自环完整有向图；统一几何边改名为 `FURNITURE_SPATIAL` | 拓扑已确认；边特征合同与实现未开始 |
| FUR-NET-005 | Graph Transformer depth | 已确定 | 1层不能用已融合上下文再次选择关系；2层覆盖全局汇总回传；3层以上不再扩大感受野 | Graph Transformer固定为2层；1层和4层仅作消融对照 | 结构已冻结；实现与验证未开始 |
| FUR-NET-006 | Constraint representation | 已确定 | Violation 因子节点、`COLLISION_CLEARANCE` 和实体关系边重复表达同一违规 | 删除 Violation 节点、`VIOLATION_TARGET` 与 `COLLISION_CLEARANCE`；违规状态写入对应实体关系边 | 设计已确认；边特征与监督合同未实现 |
| FUR-NET-007 | Semantic edge vocabulary | 已实现 v1 | `FACING / SUPPORT` 在当前 v5 中均为 0 条边，代码入口没有形成 Furniture 训练能力 | 三类语义边中只保留 `FUNCTIONAL_PARTNER`；删除 `FACING / SUPPORT` | 已按具体 HSSD ID 冻结 10,050 条资产级 partner 标注；缺失不补全 |
| FUR-NET-008 | Wall representation | 已确定 | 当前200个 clean records 只有 room dimensions，没有 WALL object 或 `OBJECT_WALL` 训练样本 | 删除 WALL 节点与 `OBJECT_WALL`；矩形墙面信息写入 Furniture–Room 边 | 设计已确认；非矩形房间门禁未实现 |
| FUR-NET-009 | HSSD behavioral descriptor | 已确定 | 原82D混合大量重复、低覆盖或运行时不稳定字段 | 禁止HSSD-ID embedding；v1只保留family、function multi-hot和各自validity mask，不再维持固定82维 | 字段合同已确认；内部宽度待消融 |
| FUR-NET-010 | Initial node feature fusion | 已实现 | 四条异构分支各映射到256维后直接相加，缺少统一融合归一化与分支级门控 | 48/32/48紧凑分支编码后 concat + learned FusionMLP + LayerNorm；统一输出128维 | v3代码与测试已完成 |
| FUR-NET-011 | Type-specific node schemas | 已实现 | 同一48维槽位对不同节点类型含义不同，共享NodeMLP编码时尚未看到node type | ROOM/OPENING/FURNITURE分别定义输入schema和局部encoder，再映射到共享128维图空间 | v3代码与测试已完成 |
| FUR-NET-012 | Relation-aware Graph Transformer | 已确定 | 四类异构边共用16维槽位和edge MLP；完整空间图与稀疏关系在同一softmax中竞争 | 四种边各设独立MLP编码器；relation-wise attention后门控融合，避免ROOM/OPENING被完整家具图掩盖 | v1边字段已确认；编码宽度待消融 |
| FUR-NET-013 | Per-furniture primary action head | 已确定 | 旧scene/candidate-query/pair/mixture链路复杂，当前数据又没有多目标模式监督 | 每个Furniture的H2并行进入ActionTypeMLP与PoseDeltaMLP；不设独立count/confidence head | 结构已确认；动作范围合同待统一 |

## 6. 措施详情

### FUR-NET-001：Category identity encoding

**状态：已确定**

已确认现状：

- 所有节点共享同一套 512 桶 category embedding。
- Furniture 的 `normalized_category` 在 HSSD 训练数据中通常是 HSSD `raw_category` 字符串。
- 哈希映射是确定性的，但不同字符串可能进入同一桶。
- `raw_category_ids` 已由数据管线保存，但没有传入当前模型计算。
- 82D HSSD label tensor 是另一条独立输入分支，不能自动等同于精确类别身份。

类别空间与碰撞预审统计（统计口径待复核，暂不作为最终设计依据）：

| 统计范围 | 类别数 | 占用桶数 | 碰撞桶数 | 被合并的额外类别数 | 最大单桶类别数 |
|---|---:|---:|---:|---:|---:|
| HSSD 全部 raw categories | 469 | 299 | 127 | 170 | 4 |
| HSSD 词表含 6 个 reserved tokens | 475 | 303 | 128 | 172 | 4 |
| Furniture 阶段 `S01` 候选集合 | 193 | 161 | 27 | 32 | 3 |
| 当前 200 个 Furniture clean scenes | 40 | 40 | 0 | 0 | 1 |

关键解释：

- 当前 clean bank 的 40 类没有碰撞只是小样本集合上的偶然结果，不能代表正式运行类别空间。
- `S01` 类别政策允许 193 类进入 Furniture 阶段，其中 59 类位于多类别桶，产生 32 个确定的身份合并。
- 全量 HSSD 469 类中只有 299 个不同桶，170 类没有独立 category embedding。
- 已观察到的实际碰撞包括 `chair / hall tree`、`armchair / candle`、`bench / loudspeaker`、`double bed / space heater`、`floor lamp / credenza` 等语义明显不同的组合。
- 当前 v5 训练张量中的 violation category 只出现 `collision`、`opening_clearance`、`room_bounds` 三种；当前数据未出现 `WALL` 和 `RELATION_GROUP` 节点。这属于训练覆盖问题，不能通过改变 one-hot 或 embedding 形式解决。

需要讨论并决定：

1. 精确 HSSD raw category 是否应成为 Furniture 节点的直接模型输入。
2. 非 HSSD 对象、非 Furniture 节点和未知类别应如何编码。
3. 是否保留哈希 category embedding 作为跨数据源的兼容分支。
4. 修改是否需要升级模型 schema、feature version，并拒绝旧 checkpoint 直接加载。
5. 应使用哪些消融和跨资产切分证明修改有效，而不是让模型记忆资产或类别频率。

当前候选方向（尚未确定）：

1. Furniture 节点直接使用冻结的 HSSD raw category vocabulary ID。现有词表为 469 类加 6 个 reserved tokens，共 475 项；`Embedding(475, 256)` 的参数量还略小于当前 `Embedding(512, 256)`。
2. `ROOM / OPENING` 不再伪装成通用 category；它们的身份已经由 node type 表达，可使用统一 `NONE` category token。
3. 已删除的 `TASK / VIOLATION / RELATION_GROUP` 不再占用 category vocabulary；非 HSSD、缺失映射和未来新增类别进入明确的 `UNKNOWN/OOV` token，并继续触发语义有效性门禁或 `ESCALATE`，不在运行时动态扩词表。
4. 任一无碰撞词表方案都需要升级模型 schema 与 feature version；旧 checkpoint 只能有选择地热启动其他兼容层，不能直接按新 ID 推理。

建议的具体落地结构：

```text
FURNITURE
  exact HSSD raw_category_id (0..474)
  -> FurnitureCategoryEmbedding(475, category_dim)
  -> projection to 256

ROOM / OPENING
  category contribution = 0
  identity由 NodeTypeEmbedding 表达
```

这里的 `raw_category_id` 来自冻结的 `raw_category_vocabulary_v2.json`，不是 HSSD asset ID，也不是运行时动态生成的序号。建议第一轮消融先使用 `category_dim=256` 形成与旧模型同形状的最小替换；只有在证明类别分支过拟合或融合尺度异常后，再单独评估较小 category embedding 与 concat fusion，避免一次实验同时改变身份编码和融合结构。

当前数据缺口：正式 S01 政策集合包含 193 类，而当前 200 个 clean scenes 只覆盖 40 类。新 embedding 中未见过训练样本的类别行不会得到有效学习，因此“取消哈希”和“补齐类别覆盖/未知类别门禁”必须作为同一改进措施验收，不能只改网络层。

每种 node type 的二级类别基数审计：

| Node type | 当前 v5 节点数 | 当前 v5 实际二级类别 | 正式允许范围 | 审计结论 |
|---|---:|---:|---:|---|
| `TASK` | 4,217 | 1：`task` | 1 个 `NONE` | 已明确；无需独立 category embedding |
| `ROOM` | 4,217 | 1：`room` | 1 个 `NONE` | 已明确；无需独立 category embedding |
| `WALL` | 0 | 0 | 新 schema 删除 | 当前数据没有墙对象或训练覆盖；矩形墙面改由 Furniture–Room 边表达 |
| `OPENING` | 8,679 | 1：`opening` | 1 个 `NONE` | 已明确；门窗几何差异由连续特征表达 |
| `FURNITURE` | 44,622 | 87 个哈希桶；94 个 raw IDs；语义有效 raw category 仅 40 类 | 全局冻结词表 475；其中 S01 政策候选 193 类 | 词表已明确，训练覆盖不足 |
| `VIOLATION` | 3,066 | 3：`collision / opening_clearance / room_bounds` | release geometry 代码还可产生 `invalid_room_geometry / finite_transform` | 尚未冻结正式输入词表；需决定后两类是进入模型还是确定性 `ESCALATE` |
| `RELATION_GROUP` | 0 | 0 | 当前 schema 接受任意 relation string；activity 模式可产生 `FACING / FUNCTIONAL_DISTANCE` | 未形成闭集且没有训练覆盖；不能直接分配正式 embedding 大小 |

当前 v5 实际只训练到 5 种 node type：`TASK / ROOM / OPENING / FURNITURE / VIOLATION`。因此，七种 node type 的一级 ID 定义虽然完整，但 `WALL` 和 `RELATION_GROUP` 目前只是代码容量，不是已训练能力。

最终方案：待讨论后填写。

### FUR-NET-002：Node type encoding

**状态：已确定**

用户提出的方向：因为 node type 固定为 7 类，可考虑改成显式 7D 0-1 one-hot，再映射到网络隐空间。

技术判断：

```text
Embedding(type_id) == one_hot(type_id, 7) @ embedding_weight
```

因此，在两者最终都通过线性权重得到 256 维向量的前提下，当前 `Embedding(7, 256)` 与“7D one-hot 再乘 `7 x 256` 矩阵”数学等价。显式构造 one-hot 不会天然更容易学习，也不会增加类别可分性。

真正需要进一步审查的是融合位置：当前实现将 geometry MLP、node type embedding、category embedding 和 HSSD label MLP 的 256 维输出直接相加。若各分支的数值尺度差异较大，某些身份或语义分支可能被压制。可讨论的改进是分支归一化、可学习门控或 concat 后统一投影，而不是仅把 embedding 改写成 one-hot。

还需要注意，当前 v5 Furniture 训练集只实际出现 `TASK / ROOM / OPENING / FURNITURE / VIOLATION` 五种节点。`WALL` 与 `RELATION_GROUP` 对应的 embedding 行没有从这份数据获得有效训练；新 Furniture 图合同已决定删除这两类节点，运行时不能把未训练代码容量当成模型能力。

最终方案：使用 `Embedding(3, 256)`，不显式构造 one-hot。3 类 ID 映射由 `FUR-NET-003` 的新 node ontology 冻结。四路输入直接相加是否需要改为归一化、门控或 concat fusion，作为独立问题继续审查，不与本项混合。

### FUR-NET-003：Node ontology

**状态：已确定**

审查标准：一种 node type 应至少具有独立的状态 schema、独立的图交互职责或独立的输出职责；不能仅因为上游数据中有一个名字就建立节点。全局 token、物理实体和约束因子可以共存，但必须说明其消息传递语义。

当前初审：

| Node type | 初步判断 | 主要依据与问题 |
|---|---|---|
| `TASK` | 删除 | 当前 Furniture 模型只有一个固定目的，无需用独立节点表达任务；它不是物理实体，当前 checkpoint 也没有有效文本语义分支，并与 `ROOM` 的全局广播和 scene pooling 职责重复。 |
| `ROOM` | 基本合理 | 房间是 Furniture 几何修复的全局容器，提供长宽高和对象/开口数量；但当前矩形房间只有一个 room，且它同样连接所有节点并被显式 pooling，可考虑与 `TASK` 合并为 `SCENE_CONTEXT`。 |
| `WALL` | 删除 | 当前 v5 完全没有 WALL 节点；200个 clean records 全部用 `room_geometry.length/width/wall_height` 表达矩形边界，且 `mounting_surfaces` 均为空。墙距与越界 margin 可直接写入 Furniture–Room 边，无需合成四个墙节点。 |
| `OPENING` | 合理，建议保留 | 门窗/开口有多个实例、独立几何和 clearance box，并与 Furniture 建立 `OBJECT_OPENING` 边；它对当前 opening clearance 修复具有不可替代信息。 |
| `FURNITURE` | 核心类型，但成员判定不合理 | 它是动作解码器的对象轴，应保留。当前构图却把所有非 `wall` 的 `scene.objects` 都标成 `FURNITURE`，随后全部加入动作候选；需要显式的 stage/target eligibility，而不能依赖“不是 wall”。 |
| `VIOLATION` | 删除 | 当前 Furniture 违规均可落到实体关系：collision 属于 Furniture–Furniture，opening clearance 属于 Furniture–Opening，room bounds 属于 Furniture–Room。继续使用因子节点会与实体关系边重复，并额外引入 violation decoder。 |
| `RELATION_GROUP` | 删除 | relation 应由 typed edge 表达；同时建立 relation node 和成员间 direct edges 会重复表示。当前 checkpoint 也没有该节点训练样本。 |

当前图还存在两个结构性现象：

1. `TASK` 与所有节点双向相连，`ROOM` 也与除自身和 TASK 外的所有节点双向相连；四层 Transformer 后又同时 pooling `task H / room H / mean(H)`，存在两个全局 hub 和三路全局汇聚的冗余风险。
2. `RELATION_GROUP` 对部分 relation 使用因子节点，同时仍在所有成员之间建立 direct typed edges；这不是统一的 edge graph，也不是统一的 factor graph。

已确定的 node type 集合：

| ID | Node type |
|---:|---|
| 0 | ROOM |
| 1 | OPENING |
| 2 | FURNITURE |

对应编码改为 `Embedding(3, 256)`。任务上下文和 verifier 结构化违规记录仍可保留在运行日志、数据 provenance、监督生成和确定性阶段配置中，但不再构造成神经图节点，也不进入 learned scene representation。

这是一次不向后兼容的图模式变更。后续检查点必须记录 node type 词表及其版本；旧的 7 类节点检查点不能直接加载到新模型中，只能有选择地复用形状与 node type ID 无关的参数。

删除 `TASK` 的实现影响：删除 TASK node features、`task_index`、全图 `TASK_CONTEXT` 边，并将当前 `[task H; room H; mean(H)] -> 256` 的 scene pooling 重新设计。新的 pooling 结构必须作为后续独立讨论项确认，不能简单用零向量代替 task state。

删除 `RELATION_GROUP` 的实现影响：删除 relation group node、`RELATION_MEMBER` edge type 和对应 category 编码。binary relation 继续使用 typed direct edges；未形成正式 pairwise 展开合同的高阶 relation 在当前 Furniture 模型中 fail-closed，不能静默展开成 clique。

删除 `VIOLATION` 的实现影响：删除 violation node features/category、`violation_indices`、`violation_mask`、`VIOLATION_TARGET`、`violation_head`、`violation_raw` 以及 resolved/progress 辅助损失。上游 verifier 记录继续存在，但只用于生成对应实体边的约束特征、训练监督和最终确定性验收。

删除 `WALL` 的实现影响：删除 wall node classification/category 和 `OBJECT_WALL` edge type。矩形房间四侧的 signed boundary margin、最近墙面方向与越界深度由 Furniture–Room 边连续表达；非矩形房间在形成新版本的显式边界合同前 fail-closed。

`FURNITURE` 的 stage/action-target 判定如何收紧，继续作为独立问题审查，不影响本轮删除四类节点的决定。

### FUR-NET-004：Furniture spatial graph

**状态：已确定**

当前实现只在 Furniture 中心距离不超过 3m 时添加 `SPATIAL_NEAR` 有向边，并为每个源节点最多保留 16 个邻居。Furniture 节点自身携带房间归一化绝对位置、yaw 和 bbox 尺寸；近邻边携带局部/世界相对位移、中心距离及相对 yaw。当前 `SPATIAL_NEAR` 调用没有传入源/目标物体，因此其 AABB gap 与 OBB penetration 特征实际为 0；只有额外的 `COLLISION_CLEARANCE` 边携带对应几何量。

当前 v5 全部 4,217 个样本的只读审计结果：

| 指标 | 结果 |
|---|---:|
| Furniture 数量 | 平均 10.58；P99/最大均为 17 |
| 当前 `SPATIAL_NEAR` 有向边 | 平均 35.10；最大 120 |
| 3m 内有序家具对覆盖率 | 100% |
| 全部有序家具对覆盖率 | 33.10% |
| 触发 16 邻居截断的 Furniture | 0 |
| 去重图中 Furniture 近邻子图不连通 | 2,022 / 3,271（61.8%） |
| 去重图中存在近邻孤立 Furniture | 1,064 / 3,271（32.5%） |

最终拓扑：对每个含 `F` 个 Furniture 的图，为所有 `i != j` 的有序对象对建立 `i -> j` 相对几何边，共 `F * (F - 1)` 条，不建立自环。新 schema 将该边命名为 `FURNITURE_SPATIAL`，不继续使用已经不符合语义的 `SPATIAL_NEAR`。完整图只作用于 Furniture 子图，不把 `ROOM / OPENING` 彼此全连接。

`FUNCTIONAL_PARTNER` 继续作为独立 typed edge 与统一几何边并行存在。`FACING / SUPPORT / COLLISION_CLEARANCE` 不再作为平行边保留；碰撞的 signed clearance、gap 和 penetration 信息并入 `FURNITURE_SPATIAL` 连续特征。

当前证据范围内 `F <= 17`，统一几何边最多 272 条。结合 `FUR-NET-003` 删除 `TASK_CONTEXT` 和 `RELATION_MEMBER` 后，按现有 v5 张量重算的总边量如下：

| 图方案 | 平均总边数 | P99 | 最大 |
|---|---:|---:|---:|
| 当前 7 类节点、3m 近邻图 | 142.7 | 282 | 296 |
| 删除三类节点及重复约束边、保留 3m 近邻图 | 109.6 | 236 | 246 |
| 删除三类节点及重复约束边、Furniture 完整图 | 180.6 | 398 | 398 |

因此完整图相对当前旧模型的平均总边数为 1.27 倍，相对完成节点/重复边删除后的近邻控制组为 1.65 倍。当前 Graph Transformer 的边相关时间和激活显存近似随 `E` 线性增长，但每层会为每条边执行两套 `16 -> 256 -> 256` edge MLP、attention bias 和消息聚合，不能把 272 条边视为零成本。当前 `F <= 17` 的范围内确认采用完整图；更大 Furniture 数量尚未获得计算证据，后续必须设置明确的运行时边预算并单独验证，不能静默退回旧的 3m 规则。

本项只冻结拓扑。`FURNITURE_SPATIAL` 的最终连续特征合同（尤其是 room-scale normalization、signed surface clearance、bbox gap/penetration 和方向表达）继续作为下一轮独立讨论，不能直接照搬当前含无效零维度的 `SPATIAL_NEAR` 特征。

### FUR-NET-005：Graph Transformer depth

**状态：已确定为2层**

当前模型使用4层、hidden dim 256、8 heads 的edge-sparse Graph Transformer。结合已经确定的完整图和边方向，逐层信息范围为：

| 深度 | Furniture可获得的信息 | 结构判断 |
|---:|---|---|
| 0 | 自身geometry/category/behavior | 尚无关系信息 |
| 1 | 所有其他Furniture的原始状态；ROOM原始状态；所有OPENING原始状态及成对clearance | 已覆盖全部直接约束，但attention query尚未包含关系融合后的上下文 |
| 2 | 其他Furniture经过ROOM/OPENING上下文更新后的状态；`MEMBER_TO_ROOM -> ROOM_TO_MEMBER` 的场景汇总回传；room-conditioned opening/furniture交互 | 覆盖当前合同中的必要两跳推理 |
| 3及以上 | 不增加新的可达节点或关系，只进行重复refinement | 收益必须由实验而非拓扑需求证明 |

1层虽已获得全图输入，但计算attention时的Furniture query仍是H0，无法根据刚收到的ROOM/OPENING上下文重新选择哪些其他Furniture更关键，因此不作为默认。2层允许第二层query使用H1中的房间、开口和功能上下文，并完成ROOM全局汇总的往返传播，是最小且完整的结构深度。

3至4层在完整Furniture图上不会扩大感受野，同时增加计算、参数和过平滑/关系反复计权风险。`FUR-NET-012` 又增加了四个独立edge encoder和relation fusion gate，单层表达能力已高于旧层，继续默认4层的依据进一步减弱。

正式结构固定为 `layers=2`。`layers=1` 只作为能力下界，`layers=4` 只作为旧模型消融控制组，不参与默认架构选择；不预设3层实验。固定完整图、hidden dim、heads、训练数据、随机种子策略、参数初始化策略和输出头，至少使用多个seed报告均值与方差。

验收同时覆盖 held-out route/target/pose、verifier hard-valid candidate rate、新增碰撞率、opening clearance、涉及ROOM边界与多Furniture交互的分桶指标、训练/推理吞吐和峰值显存。验证用于发现实现或能力问题，不再作为2层与4层之间的结构投票；若2层暴露无法接受的问题，应重新开启本条设计讨论并定位原因，不能静默回退旧4层。

### FUR-NET-006：Constraint representation

**状态：已确定**

删除 `VIOLATION` 节点及其 `VIOLATION_TARGET` 边。当前 Furniture 阶段的关系型违规按实体关系承载：

| 违规类型 | 承载边 | 最低必要信息 |
|---|---|---|
| `collision` | `FURNITURE_SPATIAL` | 各轴 signed gap、surface clearance、penetration、置信度/有效位 |
| `opening_clearance` | `OBJECT_OPENING` | signed separation、penetration、最小 escape vector、hard/tolerance/有效位 |
| `room_bounds` | Furniture–Room 关系边 | 四侧 signed boundary margin、越界方向/深度、有效位 |

违规由连续 margin 的符号和大小表达，不因“从合法跨到违规”而增删边，避免图拓扑在阈值处突变。`COLLISION_CLEARANCE` 与 `VIOLATION_TARGET` 同步从 edge vocabulary 删除。Furniture–Room 边升级为 `ROOM_MEMBERSHIP` 新schema，不能只在旧 `ROOM_CONTAINS` 上追加含义而不升级版本。

`invalid_room_geometry / finite_transform` 等输入合法性问题不是实体间关系，不构造伪造的 constraint edge：它们继续由确定性数据门禁直接 `ESCALATE/REJECT`。候选修复后的安全性仍由全量 verifier 验收，不能因为删除 Violation 节点而删除 verifier。

当前 violation decoder 的 resolved/progress 辅助目标随节点删除。若后续证明需要显式约束辅助监督，应对对应实体边增加 edge-level constraint loss，并单独验证其收益；不保留没有节点语义的旧 `violation_raw` 接口。

### FUR-NET-007：Semantic edge vocabulary

**状态：已确定**

当前 v5 的 4,217 个训练图中，`FUNCTIONAL_PARTNER` 共 27,318 条并覆盖 3,819 个图（90.6%）；`FACING` 与 `SUPPORT` 均为 0 条。`FACING` 依赖默认关闭的 activity relation 推导或上游显式 relation，无法仅从可能已损坏的当前朝向推断“应该朝向谁”；`SUPPORT` 依赖 `parent_surface_id -> mounting surface -> owner object` 的完整层级，当前 Furniture 数据没有形成对应样本，实际更接近 Manipuland 阶段职责。

三类语义边中只保留 `FUNCTIONAL_PARTNER`。它继续作为有方向的软关系边与 `FURNITURE_SPATIAL` 平行存在；target object 由边的目标端点表示，不编码对象 ID。v1 只采用 `hssd_annotation_lookup.json.gz` 中具体 HSSD ID 自身的 `interaction_clearance.functional_partners.partners` 标注：不把同 category 其他资产的规则继承给它，也不使用常识或 LLM 补全缺失标注。标注是类别级 partner 候选而不是唯一实例配对；场景中同一目标类别出现多个实例时保留多个候选，由图网络结合 `FURNITURE_SPATIAL` 判断。HSSD 字段没有稳定给出的 `orientation mode / target strategy / desired gap-range` 在冻结 v1 中分别保持 `none / none / invalid mask`，不得根据当前几何现状补造。

冻结文件为 `configs/functional_partner_rules_hssd_v1.json`，只读 builder 位于 `scene_repair_v3/functional_partners.py`。10,963 个源资产中 10,050 个有非空标注；其余 913 个以及运行时缺失 HSSD ID 的对象均生成 0 条功能边。HSSD ID 只用于图构建前查表，不进入节点特征或 learned embedding。

结合此前所有已确定修改，新 edge vocabulary 收敛并冻结为：

| ID | Edge type | 连接范围 | 拓扑与职责 |
|---:|---|---|---|
| 0 | `ROOM_MEMBERSHIP` | ROOM ↔ OPENING/FURNITURE | 保留双向消息；内部区分 `ROOM_TO_MEMBER` 与 `MEMBER_TO_ROOM`；Furniture–Room 边还承载 room-boundary margin |
| 1 | `FURNITURE_SPATIAL` | Furniture -> Furniture | 无自环完整有向图；连续相对几何及 collision margin |
| 2 | `FUNCTIONAL_PARTNER` | Furniture -> Furniture | HSSD 已标注的有向 partner 候选；目标实例由边端点表达 |
| 3 | `OBJECT_OPENING` | Opening -> Furniture | 单向约束消息；向每件Furniture提供开口几何及该家具对应的opening-clearance margin，不更新Opening以汇总家具 |

因此新 schema 使用 4 种 edge type，Graph Transformer 每层对应的 edge key/value/bias embedding 词表大小均为 4。`TASK_CONTEXT / SPATIAL_NEAR / FACING / SUPPORT / OBJECT_WALL / COLLISION_CLEARANCE / VIOLATION_TARGET / RELATION_MEMBER` 全部删除或被替换。edge type ID 必须按这份新闭集写入 checkpoint metadata；旧 edge embedding 不能按原 ID 直接加载。

### FUR-NET-008：Wall representation

**状态：已确定**

当前200个 v5 clean records 包含2,164个 Furniture object、0个 Wall object、0个非空 `mounting_surfaces`；全部200个场景都有 `room_geometry.length/width/wall_height`，400个 Opening 均通过 `wall: north/south/east/west` 引用隐式矩形墙面。GraphBuilder 只会为 `scene.objects` 中 `object_type == "wall"` 的对象创建 WALL 节点，因此当前训练张量中的 WALL 节点和 `OBJECT_WALL` 边均为0。

新 Furniture schema 删除 WALL 节点和 `OBJECT_WALL`。对于当前矩形房间，墙相关信息不通过四个合成墙节点表达，而由每条 Furniture–Room 边编码到四侧边界的 signed margin、最近墙面方向、最近墙面距离和越界深度。Opening 自身继续保留 wall direction/interior normal，因此删除 WALL 不会删除开口依附方向。

该决定只适用于当前矩形 room contract。非矩形、多边形、独立墙段或需要墙面挂靠的输入，在建立带版本的显式 boundary/wall schema 和训练覆盖前必须 `ESCALATE/REJECT`，不能把不规则边界压缩为 length/width 后继续推理。

### FUR-NET-009：HSSD behavioral descriptor

**状态：v1 输入字段已确定；旧 82D 重设计方案已否决**

当前82D `hssd_label_features_v2` 是 family/function/context/stage/support/validity/confidence 的结构化语义，不包含 HSSD asset identity。对当前200个 clean records 的373个不同 HSSD ID 审计发现：同 raw category 内共有54组“不同HSSD ID、相同82D签名”，涉及346个ID（92.8%），最大一组30个ID。这个结果说明原82D不能区分资产编号，但不意味着应增加编号embedding：冻结 `asset_semantics_v2.jsonl` 有10,963个HSSD资产，当前范围只覆盖373个（3.4%），显式ID表会让模型记忆少量已见资产，并给绝大多数未见ID留下无监督随机行。

已确定原则：HSSD ID 不作为任何 learned embedding 的索引，只用于冻结元数据 lookup、asset-disjoint split、provenance 和审计。模型对未见资产的布局行为必须来自可迁移标签与几何；同category资产只有在功能、上下文、布局先验或物理几何确实不同时才应被区分。如果这些行为相关信息等价，复用同一修复规律是期望行为，不需要强行识别编号。

后续逐字段审查否决了“为了保持82维而填满更多语义标签”的思路。对单步 `Δx/Δy/Δyaw` 修复，candidate/resolved context、functional detail、semantic role、allowed surface、semantic front、keep-clear、placement DOF、environment anchor、confidence 和 provenance 要么运行时不稳定，要么属于阶段门禁/关系边，要么与 category/function 重复。v1 `BehavioralEncoder` 只接收：

```text
family embedding
function multi-hot
family_valid
function_valid
```

HSSD ID 仍只用于冻结 lookup、asset-disjoint split 和审计。未见训练资产只有在其 HSSD ID 能被冻结语义注册表映射到 category/family/function 时，才能复用已学习的标签语义；注册表也未知时使用显式 `UNK`/invalid mask，并按安全合同处理。BehavioralEncoder 不再保留固定 82 维接口，内部宽度由消融确定。

验收必须使用HSSD-ID严格隔离的validation/test，并按“未见ID但标签已见”“未见标签组合”“缺失/冲突标签”分别报告。关键消融为 exact category + geometry、再加 behavioral descriptor、再加 `FUNCTIONAL_PARTNER`；只有行为标签在未见HSSD ID上改善 target/pose/verifier-hard-valid 指标，才证明模型学到标签含义而不是资产频率。

### FUR-NET-010：Initial node feature fusion

**状态：已确定；分支宽度待消融确定**

当前实现把 `NodeMLP(geometry)`、`NodeTypeEmbedding`、`CategoryEmbedding` 和 `HSSDLabelMLP` 的256维输出直接求和。直接相加并非天然错误：原始 Transformer 将 token embedding 与 positional encoding 相加，BERT 将 token、segment、position embedding 相加。但这些分量都是附着在同一个token上的少量、语义对齐的修饰信息，不能直接证明任意异构字段都适合等权相加。

更接近当前问题的经典做法有两类。Graphormer把节点局部属性放进node state，而把最短路距离、边等成对关系放进attention score/bias；FT-Transformer把不同表格字段保留为独立feature token，再让attention学习字段交互。共同原则是：信息应按语义进入适合的通道，不因都能投影到同一宽度就提前压成一个和向量。

当前 direct-sum 的具体风险：

- geometry/HSSD两个MLP输出带LayerNorm，type/category embedding没有同等尺度约束，求和后也没有独立fusion normalization；
- 第一层图注意力在其 `norm1` 之前就用该求和结果计算Q/K/V和attention logits，分支尺度会直接影响第一次edge softmax，后置LayerNorm不能撤销已经发生的信息选择；
- 所有分支固定以系数1相加，不能按节点、标签可靠性或缺失状态调整权重；
- 求和后来源身份不可直接恢复，category与behavioral descriptor的重叠语义可能重复计权；
- 单个 `bundle_valid` 会使缺少局部标签与整条语义分支缺失产生同样结果。

已确定不增加一层完整feature Transformer，采用更简单的 late fusion：

```text
z_geometry = GeometryEncoder(geometry)
z_type     = TypeEmbedding(node_type)
z_category = CategoryEncoder(category)
z_behavior = BehavioralEncoder(labels, per_branch_valid_masks)

H0 = LayerNorm(
       FusionMLP(concat(
         norm(z_geometry), norm(z_type), norm(z_category),
         norm(z_behavior), per_branch_valid_masks
       ))
     )
```

这种结构让每条分支在融合前保持来源身份，`FusionMLP` 可学习不同分支的权重与交互，最终输出128维node state。各分支没有必要先扩展到共享hidden width；v1默认 geometry/category/behavior 分别为48/32/48维。若后续证据表明复杂标签交互明显受限，再比较更宽分支或“每个语义分支一个feature token + 小型attention pooling”，不把它作为第一版默认实现。

至少比较以下消融，其他训练配置和参数预算尽量对齐：

| 方案 | 融合方式 | 目的 |
|---|---|---|
| A | 当前4个256维分支直接相加 | 基线 |
| B | 分支归一化 + learned scalar gate + 求和 + final LayerNorm | 判断尺度/门控是否已足够 |
| C | 紧凑分支编码 + concat + FusionMLP + LayerNorm | 首选候选 |
| D | 独立feature tokens + 小型attention pooling | 仅在C仍不足时评估 |

验收除总体修复指标外，还需报告：HSSD-ID隔离测试、随机屏蔽单分支后的性能退化、各分支梯度/输出范数、缺失标签鲁棒性。目标不是证明 concat 必然优于 addition，而是确认融合方式没有让某一分支长期压制其他信息，并能在未见资产上稳定利用共享标签。

论文依据：Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)；Devlin et al., [BERT](https://arxiv.org/abs/1810.04805)；Ying et al., [Graphormer](https://papers.nips.cc/paper/2021/hash/f1c1592588411002af340cbaedd6fc33-Abstract.html)；Gorishniy et al., [Revisiting Deep Learning Models for Tabular Data / FT-Transformer](https://papers.neurips.cc/paper_files/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html)。

### FUR-NET-011：Type-specific node schemas

**状态：已确定；具体字段待逐类审查**

保留的三种节点不应继续共享一个含义混杂的48维输入schema。当前 `_node_features` 对所有节点先构造48维向量，再由同一个 `NodeMLP(48→256)` 编码，node-type embedding 只在编码后相加。相同槽位因此存在类型别名：例如第0维对ROOM是房间length/10，对OPENING和FURNITURE却是x/room-length；第3维对ROOM是opening count，对OPENING是hard-clearance，对FURNITURE是sin(yaw)。共享NodeMLP第一次读取数值时尚不知道节点类型，只能依赖后续网络间接消歧。

当前有效槽位也极不均衡：ROOM只显式使用前5维，OPENING主要使用0至20维，其余大量补0；FURNITURE几乎使用全部48维，但其中category hash、allowed-surface、functional-partner、environment-anchor等又与独立category/behavior分支或图边重复。统一48维张量只是存储形状一致，不代表合理的共享特征空间。

已确定的数据和模型边界：

```text
ROOM schema      -> RoomEncoder -----------┐
OPENING schema   -> OpeningEncoder --------┼-> shared 128D node space
FURNITURE schema -> FurnitureFusionEncoder ┘
                                              -> Graph Transformer
```

- `RoomEncoder`：只接收房间容器自身的几何与有效性；对象数、opening数是否保留需与图degree/pooling做去重。
- `OpeningEncoder`：接收opening种类、中心/尺寸、所属墙面或内法向、净空体和有效性；与Furniture的相对量放在 `OBJECT_OPENING` 边，不重复写入opening节点。
- `FurnitureFusionEncoder`：按 `FUR-NET-010` 融合本体几何、category和behavioral descriptor；相对墙面、opening、其他家具的信息分别进入对应边。

三条encoder最终都输出同一128维，表示它们需要进入同一个图消息传递空间，并不要求输入字段、输入维度或内部参数相同。node type只用于选择专属encoder和构造direction role，不再额外叠加共享宽度的type embedding。

不为三种节点复制三个完整大网络。首版采用“小型类型专属encoder + 共享Graph Transformer”，既避免输入语义冲突，又让三种节点在后续关系推理阶段共享参数。是否需要共享基础几何子层，应在字段schema稳定后用参数对齐的消融验证。

### FUR-NET-012：Relation-aware Graph Transformer

**状态：已确定；边字段与编码宽度待设计**

当前 `GraphTransformerLayer` 的基本骨架可保留：按有向入边计算稀疏multi-head attention，edge type/feature进入key与value，之后使用residual、LayerNorm和FFN；在 `F <= 17` 的当前范围内没有必要改成稠密 `N×N` attention。hidden dim 256、8 heads以及无显式self-loop但有residual的设计，也没有仅凭结构审查必须修改的证据。

需要修改的是边的编码与多关系聚合：

1. **边特征存在类型槽位冲突。** 当前四类边都会先经过同一套 `Linear(16→256) -> GELU -> Linear(256→256)` key/value MLP，edge-type embedding在共享MLP之后才加入。新合同中，同一槽位会分别承载Furniture碰撞margin、room boundary margin、opening clearance或functional confidence，共享MLP第一次解释数值时并不知道关系类型。这与旧48维节点槽位冲突是同一类问题。
2. **完整图造成关系数量偏置。** 当前softmax在目标节点的全部入边上统一归一化。若一个Furniture有16条 `FURNITURE_SPATIAL` 入边和1条ROOM入边，在logit相近时空间关系组天然取得约16倍总权重；家具数量变化还会改变这一比例。稀疏但关键的ROOM、OPENING和FUNCTIONAL关系不应仅依赖一个静态relation bias抵消degree差异。
3. **关系方向必须符合节点职责。** `ROOM_MEMBERSHIP` 保持双向，内部区分 `ROOM_TO_MEMBER` 与 `MEMBER_TO_ROOM`；ROOM先汇总成员、再向成员广播全局场景信息。`OBJECT_OPENING` 只保留 `OPENING_TO_FURNITURE`：Opening是不可移动约束源，每条边直接携带目标Furniture对应的relative geometry与clearance margin；不建立 `FURNITURE_TO_OPENING` 汇总路径。边编码仍显式看到source/target node type。

已确定保留共享图层主体，采用轻量的relation-aware修改。修改目标归纳为两项：

1. `ROOM_MEMBERSHIP / FURNITURE_SPATIAL / FUNCTIONAL_PARTNER / OBJECT_OPENING` 四种边分别使用独立的schema-specific MLP编码器，不再让含义不同的原始槽位先通过同一个MLP。
2. 四类入边分别执行邻居attention聚合，再由relation gate融合各类消息，缓解完整 `FURNITURE_SPATIAL` 图因边数多而掩盖ROOM和OPENING信息的问题。

对应计算结构：

```text
raw edge fields
  -> EdgeEncoderMLP[relation_type](type-specific schema -> compact edge state)
  -> add DirectionRoleEmbedding(source_type, relation_type, target_type)
  -> shared/per-layer edge key and value projection

for each target node i and relation r:
  m_i^r = softmax over incoming edges of relation r
          -> weighted message sum

H_i' = RelationFusionGate({m_i^r}, relation_present_masks, H_i)
       -> residual + LayerNorm + FFN
```

四个 `EdgeEncoderMLP` 按各自关系读取冻结的连续特征schema，并分别映射到紧凑公共edge space；之后可复用每层key/value投影，不复制四套完整256维Graph Attention网络。relation-wise softmax消除关系组大小带来的先验权重，`RelationFusionGate` 再学习空间、房间、开口、功能四组消息对当前节点的重要性。不存在的关系组必须由presence mask排除，不能用伪造零消息参与门控。

第一版明确保留：

- directed incoming-edge message passing；
- edge feature同时影响attention key/logit与message value；
- node residual、FFN、dropout；
- shared Graph Transformer参数主体；
- Graph Transformer按 `FUR-NET-005` 固定为2层；1层和4层仅作消融对照。

第一版不采用完整HGT式的每种node/edge type独立Q/K/V矩阵，以避免在当前数据量下无依据地放大参数。`FUR-NET-010` 已使每层输入经过最终LayerNorm；在仅2至4层的范围内，post-LN是否改为pre-LN不列为必改项，只有出现梯度或收敛不稳定证据时再做对照。

最低消融：当前统一softmax/共享edge MLP；改为四个独立 `EdgeEncoderMLP`；再加入direction role；最后加入relation-wise softmax与relation gate。除修复指标外，需记录每类关系的attention总质量、gate值、目标节点度数分桶性能，以及稀疏ROOM/OPENING/FUNCTIONAL消息是否被完整空间边长期压制。

论文依据：Ying et al., [Graphormer](https://papers.nips.cc/paper/2021/hash/f1c1592588411002af340cbaedd6fc33-Abstract.html) 将节点局部信息与attention中的空间/边编码分开注入；Hu et al., [Heterogeneous Graph Transformer](https://arxiv.org/abs/2003.01332) 使用依赖source node type、relation type与target node type的异构attention/message参数。这里采用的是适合当前3种节点、4种边和有限数据量的轻量版本，而不是完整复刻HGT。

### FUR-NET-013：Per-furniture primary action head

**状态：已确定；动作范围合同待统一**

Transformer之后的第一目标不是先生成全局candidate state，而是直接为每个Furniture给出主要动作建议：该对象保持、只平移、只旋转或同时平移旋转，以及世界/房间坐标系下移动多少、应转动多少。`H_i^2` 已经融合全部Furniture、ROOM、OPENING和边约束，因此第一版不再额外拼接scene pooling或learned candidate query。

建议结构：

```text
H_i^2 (Furniture, 128D)
       |-> ActionTypeMLP: Linear 128->128 -> GELU -> LayerNorm -> Linear 128->4
       |                  -> KEEP / TRANSLATE / ROTATE / BOTH logits
       |
       `-> PoseDeltaMLP:  Linear 128->128 -> GELU -> LayerNorm -> Linear 128->3 -> tanh
                          -> normalized (dx_i, dy_i, dyaw_i)

dx_m   = normalized_dx   * max_translation_m
dy_m   = normalized_dy   * max_translation_m
dyaw   = normalized_dyaw * max_yaw_rad
```

输出合同只按Furniture节点轴返回 `action_type_logits[N_f,4]` 与 `primary_delta[N_f,3]`。`move_probability = 1 - P(KEEP)`，场景预测变更数量可按 `sum_i(1-P_i(KEEP))` 派生，无需重复的MoveHead或ChangeCountHead。位姿是相对当前状态的残差，不是绝对目标位姿；平移使用稳定的房间世界XY坐标，旋转使用有界相对yaw。当前动作范围在不同代码/数据合同中存在 `0.4m/45deg` 与数据集metadata `0.55m/60deg` 的不一致，正式实现必须从版本化dataset/checkpoint contract读取唯一范围，禁止模型、loss和decoder各自硬编码。

两个MLP直接共享Transformer输出 `H_i^2`，不再额外增加共同ActionAdapter。分类与连续回归使用独立参数，避免把不同损失尺度强塞进一个7维联合输出；两项任务仍会通过共同的Furniture encoder和Graph Transformer共享场景表示。

执行时由动作类型确定delta有效分量：`KEEP -> (0,0,0)`，`TRANSLATE -> (dx,dy,0)`，`ROTATE -> (0,0,dyaw)`，`BOTH -> (dx,dy,dyaw)`。

`movable/immutable/transform-valid` 是确定性资格门禁：不可移动或transform无效对象在动作头外直接mask，`move_probability=0`、delta不执行，不能要求网络从样本频率自行学会硬约束。

训练目标：

- `ActionTypeHead` 对每个Furniture使用加权4类cross-entropy；类别由target delta确定，未改变为KEEP、仅XY非零为TRANSLATE、仅yaw非零为ROTATE、两者非零为BOTH；
- translation只在TRANSLATE/BOTH且pose-valid的节点上使用以米为单位的Smooth-L1；
- yaw只在ROTATE/BOTH且pose-valid节点上使用周期损失 `1-cos(pred-target)`；
- unchanged节点增加轻量zero-delta正则，避免被mask区域产生任意大残差；
- 不对多个mode先求期望后再回归，第一版根本不输出无监督的mixture参数。

当前 `hssd_furniture_geometry_v2` 7,000条样本审计结果：5,800条POSE_REPAIR样本均只有1个target/1个target cluster，另外1,200条没有target；没有任何样本提供多个等价有效target cluster。虽然5,800条修复样本各有4个candidate pool项，但只有一个被选为有效target。因此当前数据不能训练可靠的多峰动作分布；旧3分量translation/yaw mixture最终又先求期望得到单个 `pose_raw`，scale/kappa也没有直接概率损失约束，不能视为已学会多解。

动作类型审计进一步显示：40,949个changed Furniture中，23,676个（57.8%）只平移，8,810个（21.5%）只旋转，8,463个（20.7%）同时平移与旋转；不存在delta全零却标记changed的节点。因此独立translation/rotation执行门控有直接标签依据。有效目标的changed Furniture数量虽覆盖1至12，但它等于逐节点非KEEP标签之和且分布明显受合成corruption策略影响，不足以证明需要独立ChangeCountHead。

明确不进入动作头输出合同的信息：

- **Change count：** 它是逐节点动作类别的确定性派生量。独立ROOM count head会与节点预测形成两套可能冲突的结论，后续还必须引入top-k规则强行裁决。若实验发现数量校准不足，只在训练中对 `sum_i(1-P_i(KEEP))` 增加可选一致性loss，不新增推理head。
- **Eligibility：** `movable/immutable/transform-valid` 是输入数据已知的硬资格，不是学习目标；在动作头外mask即可。
- **Confidence/OOD scalar：** action-type softmax可以校准后作为分类置信参考，但单独再训练一个confidence标量不能可靠表达未见HSSD资产的认知不确定性。OOD/abstention需要结合标签有效性、asset-disjoint验证及后续专门方法讨论，不能伪装成动作头已解决的问题。
- **Post-action validity/ranking：** 碰撞、越界、opening clearance和候选合法性由应用动作后的确定性Verifier计算，不让动作头近似猜测。

节点级主建议与联合候选生成必须分层：本条只产生每个Furniture的 `action type + primary delta`。哪些Furniture应共同执行、如何构造多个可验证候选、如何在等价修复之间保持多样性，作为后续独立模块讨论，不能重新塞回本动作头。

## 7. 决策记录

| 日期 | 条目 | 结论 |
|---|---|---|
| 2026-07-27 | 文档初始化 | 建立审查基线；确认 node type 与 category identity 的实际编码；`FUR-NET-001` 保持待讨论，不视为已确定修改。 |
| 2026-07-27 | `FUR-NET-001/002` 第一轮讨论 | 完成 HSSD 全量、S01 政策集合和当前 clean bank 的哈希碰撞统计；确认显式 one-hot 与 embedding lookup 的线性等价性。两项仍待最终方案确认。 |
| 2026-07-27 | `FUR-NET-002` 决策 | 保留 embedding 编码机制，不改成显式 one-hot；结合 FUR-NET-003 的 ontology 决策，最终使用 `Embedding(3, 256)`。 |
| 2026-07-27 | 二级类别基数审计 | 完成当前 v5 七种 node type 的节点数与实际类别覆盖统计；确认 Violation 正式词表和 Relation Group 合同尚未闭合。 |
| 2026-07-27 | `FUR-NET-003` 第一轮审查 | 类别数量统计口径暂不作为决策依据；转向审查 7 类节点本身。确认三类核心节点、两个全局 hub 的冗余风险、未训练 WALL、重复表示的 RELATION_GROUP，以及 FURNITURE 成员判定过宽。 |
| 2026-07-27 | `FUR-NET-003` 决策 | 删除 `TASK / RELATION_GROUP / VIOLATION / WALL`；节点类型从7类收敛为 `ROOM / OPENING / FURNITURE` 3类，使用 `Embedding(3, 256)`。 |
| 2026-07-27 | `FUR-NET-004` 决策 | Furniture–Furniture 使用无自环完整有向图；统一相对几何边改名为 `FURNITURE_SPATIAL`。当前决定的证据范围为 `F <= 17`。 |
| 2026-07-27 | `FUR-NET-005` 实验决策 | 在完整图上将 4 层作为控制组，正式试验 2 层 Graph Transformer；最终层数等待效果、吞吐与显存消融后确认。 |
| 2026-07-27 | `FUR-NET-005` 拓扑复审 | 最终边方向下，2层是完成关系上下文化和ROOM汇总回传的最小完整深度；实验矩阵调整为2层主方案、1层下界、4层旧控制，3层仅作退化后的补测。 |
| 2026-07-27 | `FUR-NET-005` 最终决策 | Graph Transformer固定为2层；1层和4层只保留为消融对照，不再参与默认结构选择。 |
| 2026-07-27 | `FUR-NET-006` 决策 | 删除 Violation 节点、`VIOLATION_TARGET`、`COLLISION_CLEARANCE` 及 violation decoder；collision/opening/room-bounds 违规改由对应实体关系边的连续 margin 特征表达。 |
| 2026-07-27 | `FUR-NET-007` 决策 | 三类 Furniture 语义边中只保留有训练覆盖的 `FUNCTIONAL_PARTNER`；删除 `FACING / SUPPORT`。 |
| 2026-07-27 | `FUR-NET-008` 决策 | 删除未训练的 WALL 节点与 `OBJECT_WALL`；矩形墙面约束统一写入 Furniture–Room 边。最终节点词表3类、边词表4类。 |
| 2026-07-27 | `FUR-NET-009` 第一轮设计 | 证实现有82D无法区分同category的多数HSSD资产；曾提出32D资产ID与50D语义的候选。 |
| 2026-07-27 | `FUR-NET-009` 方向决策 | 否决显式HSSD-ID embedding；82D改为完全由可迁移行为标签组成，目标是在未见HSSD ID上复用已学习的标签语义。 |
| 2026-07-27 | `FUR-NET-010` 决策 | 删除四个256维异构分支的直接等权相加；改用紧凑分支编码、concat、learned FusionMLP与最终LayerNorm，输出仍为256维。 |
| 2026-07-27 | `FUR-NET-011` 决策 | ROOM、OPENING、FURNITURE分别使用类型专属schema与小型局部encoder，统一输出256维后进入共享Graph Transformer；具体字段逐类审查。 |
| 2026-07-27 | `FUR-NET-012` 决策 | 四种边类型各设独立schema-specific MLP编码器；各关系分别执行attention聚合后由relation gate融合，避免完整Furniture空间边在统一softmax中掩盖ROOM/OPENING信息。 |
| 2026-07-27 | `ROOM_MEMBERSHIP` 方向决策 | 将 `ROOM_CONTAINS` 改名为中性关系大类 `ROOM_MEMBERSHIP` 并保持双向；内部显式区分成员汇总到ROOM的 `MEMBER_TO_ROOM` 与ROOM向成员广播的 `ROOM_TO_MEMBER`。 |
| 2026-07-27 | `OBJECT_OPENING` 方向决策 | 只保留 `OPENING_TO_FURNITURE` 单向边；Opening作为不可移动约束源向各Furniture提供成对几何与clearance margin，不建立用途未获证据支持的Furniture→Opening汇总路径。 |
| 2026-07-27 | `FUR-NET-013` 输出方向决策 | 每个Furniture直接输出 `KEEP/TRANSLATE/ROTATE/BOTH` 与主Δx/Δy/Δyaw；不设重复且可能冲突的ChangeCountHead，不把已知eligibility或未经证明的confidence/OOD标量作为动作头输出。 |
| 2026-07-27 | `FUR-NET-013` 结构决策 | 每个Furniture的256维H2分别进入 `ActionTypeMLP(256→128→4)` 与 `PoseDeltaMLP(256→128→3)`；两个任务头不共享额外ActionAdapter，动作类型在执行和loss中门控delta有效分量。 |
| 2026-07-28 | 节点/边字段 v1 决策 | 冻结来源精简后的节点与四类边输入合同：behavior仅保留family/function及分支mask；连续几何只保留对单步Δx/Δy/Δyaw有直接作用且运行时可确定构造的字段。 |
| 2026-07-28 | Geometric yaw 决策 | Furniture 只编码 canonical transform 的几何 yaw，不增加 semantic-front angle/offset/validity；训练与 SceneExpert 推理冻结相同的 room frame、资产轴、四元数顺序、yaw 提取/正方向/wrap。方向无意义物件的 yaw 相关性由类别、几何上下文和监督学习，数据生成器不得只因其 yaw 不同制造旋转目标。 |
| 2026-07-28 | 网络宽度决策 | v1默认改为 `hidden=128 / heads=4 / head_dim=32 / edge=32 / geometry=48 / category=32 / behavior=48 / layers=2`，约54万参数；保留每头32维和两层传播职责，192/256维仅作为容量消融。 |
| 2026-07-28 | ACPM 边界决策 | 网络schema不依赖ACPM。ACPM只可作为经过引用/时效校验的可选关系证据或训练教师；模型读取来源无关的wall-anchor/functional relation，target object由边端点表达，不编码对象ID，provenance只进入审计sidecar。 |

## 8. 变更记录

| 版本 | 日期 | 说明 |
|---|---|---|
| v1 | 2026-07-27 | 创建文档；登记当前网络基线、node type 讨论及 category identity 实际碰撞证据。 |
