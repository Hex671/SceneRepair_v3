# SFUR v1 指标合同

## 1. 指标定位

SFUR（Scene Functional Usability Rate，整场功能可用率）用于判断 Furniture 修复后的完整房间是否满足可执行的功能布局约束。它覆盖硬几何约束、家具功能关系、使用净空和核心目标可达性，但不评价审美、风格或完整的人类主观自然度，因此不等同于 HLAR。

冻结规则文件：`configs/sfur_rules_v1.json`

规则 schema：`scene_repair_v3_sfur_rules_v1`

规则 SHA256：`d7055aba6de593a145f5b18437ca0d16d659ebd5888b42744e94cff38229af40`

## 2. 整场通过条件

每个场景必须同时满足：

1. 房间边界、家具碰撞和 opening clearance 三类硬约束全部通过。
2. 核心路径目标可达率为 100%。
3. 功能综合分不低于 0.90。

功能综合分为：

```text
functional_score =
    0.40 * functional_relation_pass_rate
  + 0.35 * interaction_clearance_pass_rate
  + 0.25 * core_reachability
```

数据集级指标为：

```text
SFUR = 通过整场条件的场景数 / 全部评估场景数
```

SFUR 是严格的整场指标。任一硬约束残留、任一核心路径不可达，或者综合分低于 0.90，整场均判为失败。

## 3. 功能关系

v1 仅冻结 clean512 数据中稳定、可校准的强功能类别对：

| 规则 | 主体 | 允许间隙 | 朝向约束 |
|---|---|---:|---:|
| chair-table | chair | 0.00-0.35 m | chair 45 deg |
| chair-desk | chair, desk | 0.00-0.35 m | chair 25 deg, desk 40 deg |
| bench-table | bench | 0.00-0.35 m | bench 25 deg |
| sofa-coffee_table | sofa | 0.25-1.30 m | sofa 45 deg |
| bed-nightstand | nightstand | 0.00-0.31 m | 无强制朝向 |

候选 FUNCTIONAL_PARTNER 边存在多个目标时，评估器按规则主体选择最近的有效目标，不要求同一主体同时满足所有候选边。HSSD 未提供的功能关系不得手工补边。

## 4. 使用净空与路径合同

SFUR 从场景 JSON 的以下字段构造功能合同：

```text
audit.use_clearance_zones
audit.path_targets
```

世界坐标净空和目标点在读入时转换为家具局部坐标；评估和修复时，它们随当前家具姿态更新。运行时不读取 clean 绝对姿态，也不读取扰动对象身份。

路径搜索使用 0.10 m 网格和 0.28 m 人体半径。核心 path target 必须全部从门口可达。

## 5. 校准与适用范围

- 512/512 authored clean 场景全部通过 SFUR v1，证明该合同没有系统性否定训练参考布局。
- 受扰动场景会显著失败，证明指标不是恒真检查。
- 当前仅支持单房间 Furniture 阶段，单场最多 17 件家具。
- 输入缺少功能合同时，只能运行硬约束模式，不得声称处于 SFUR 模式。
- SFUR v1 不能替代真实用户研究，也不能证明修复结果具备完整的人类审美接受度。
