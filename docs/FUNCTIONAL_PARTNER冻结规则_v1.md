# FUNCTIONAL_PARTNER 冻结规则 v1

## 冻结来源

规则只提取自 `hssd-annotations/data/hssd_annotation_lookup.json.gz` 的
`interaction_clearance.functional_partners.partners` 字段。冻结文件为：

`configs/functional_partner_rules_hssd_v1.json`

每条规则以具体 HSSD ID 为键。相同 category 的其他资产不会继承该规则；没有
HSSD ID、ID 不在冻结表中或原字段为空时，builder 不生成边。系统也不使用
commonsense、LLM 或手写类别规则补全缺失标注。

## 推理语义

冻结规则提供的是 HSSD 已标注的 partner 类别候选，不是场景级人工标注的唯一
实例配对。若场景内存在匹配类别的目标 Furniture，builder 从标注资产指向目标
实例生成 `FUNCTIONAL_PARTNER` 候选边。多个目标匹配时保留多个候选，让图网络
结合并行的 `FURNITURE_SPATIAL` 几何边判断其重要性。

HSSD partner 字段本身没有稳定提供 orientation mode、target strategy 和 desired
gap。因此 v1 对这些字段分别输出 `none`、`none` 和 invalid gap mask，不从几何
现状或常识补造约束。

HSSD ID 只参与图构建前的冻结查表，不进入 Furniture 节点特征，也不会获得
learned embedding。

## 使用方式

```python
from scene_repair_v3 import FrozenFunctionalPartnerRules

rules = FrozenFunctionalPartnerRules.load(
    "configs/functional_partner_rules_hssd_v1.json"
)
scene_with_relations = rules.apply(scene_without_relations)
graph = graph_builder.build(scene_with_relations)
```

若 `scene_without_relations` 已包含关系，`apply` 会拒绝混合手写边与冻结边。

## 可复现提取

```powershell
python tools/freeze_hssd_functional_partners.py
```

冻结文件记录源 lookup 的 SHA-256、规则内容 SHA-256、源资产数、已标注资产数和
严格缺失处理策略。加载时会验证 schema、policy、数量和规则 checksum。
