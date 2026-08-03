# SceneExpert Furniture 推理接口 v2（SFUR）

## 1. 正式部署配置

SFUR 完整推理链路为：

```text
scene JSON
  -> type-specific graph encoding
  -> dense_delta semantic pass x 4（每轮重新构图）
  -> hard-constraint refinement
  -> functional refinement
  -> repaired JSON + hard/SFUR audit
```

正式 checkpoint：`runs/furniture_clean512_sfur_v10_final/best.pt`

正式 SFUR 规则：`configs/sfur_rules_v1.json`

checkpoint 自带 deployment profile；runtime 加载后自动选择 `dense_delta`、4 passes 和功能细化。

## 2. 命令行调用

```powershell
D:\anaconda\envs\SceneRepair_v2\python.exe -m tools.infer_furniture `
  --input data\sceneexpert_inputs `
  --output-dir runs\furniture_runtime_sfur_v1 `
  --checkpoint runs\furniture_clean512_sfur_v10_final\best.pt `
  --vocab data\training\clean512_v1\furniture_vocab_clean512_v1.json `
  --functional-rules configs\functional_partner_rules_hssd_v1.json `
  --compatibility-rules configs\functional_partner_category_compatibility_v1.json `
  --sfur-rules configs\sfur_rules_v1.json `
  --batch-size 32 `
  --device auto
```

不传 `--sfur-rules` 时只运行向后兼容的硬约束模式，输出不得解释为 SFUR 修复结果。

## 3. Python 调用

```python
from scene_repair_v3 import FurnitureRepairEngine

engine = FurnitureRepairEngine.load(
    checkpoint="runs/furniture_clean512_sfur_v10_final/best.pt",
    vocab="data/training/clean512_v1/furniture_vocab_clean512_v1.json",
    functional_rules="configs/functional_partner_rules_hssd_v1.json",
    compatibility_rules=(
        "configs/functional_partner_category_compatibility_v1.json"
    ),
    sfur_rules="configs/sfur_rules_v1.json",
)

predictions = engine.repair_scene_jsons(scene_json_objects)
repaired_json = predictions[0].repaired_scene_json
audit = predictions[0].audit
```

输入对象不会被原地修改。

## 4. 输入要求

继续使用 Furniture 阶段的单房间 JSON 合同，且 `F <= 17`。SFUR 模式额外强制要求：

```text
audit.use_clearance_zones
audit.path_targets
```

这两个键必须显式存在；其值可以是空数组。缺失时 runtime 直接拒绝 SFUR 推理，不会静默降级后仍声称功能修复成功。

功能合同只提供家具局部语义意图。runtime 不读取 authored clean 绝对姿态，也不读取当前输入由哪些扰动生成。

## 5. 输出审计

每个 `<scene>.repair_audit.json` 包含：

- 输入、checkpoint 和规则的可追踪信息；
- 实际 deployment mode、action policy 和 semantic passes；
- 修复前、神经输出和最终硬约束计数；
- 硬约束与功能细化迭代数和收敛状态；
- SFUR 整场结果、功能综合分和三个子指标；
- 每件家具的最终动作、累计 delta 和前后姿态。

`repair_manifest.json` 汇总：

```text
all_scenes_feasible
sfur_mode
sfur_passes
sfur
```

## 6. 拒绝条件

以下情况直接失败：

- checkpoint、模型配置、词表、图合同或 ActionRange 不匹配；
- SFUR 规则 schema 非法；
- SFUR 模式缺少功能合同；
- Furniture 行顺序与模型输出不一致；
- 最终仍存在任何硬约束；
- 累计动作超过 checkpoint 的 ActionRange；
- 输出已存在但未显式传入 `--overwrite`。

功能细化没有完全收敛时不会伪造通过；实际 SFUR 失败和具体 violations 会保留在 audit 中。

## 7. 已验证结果

冻结测试：`753/832 = 90.5048% SFUR`，硬约束 `832/832 = 100%`。

真实 accepted Scene JSON 的 GPU BF16 runtime smoke test 已确认：模型自动使用 `dense_delta + 4 passes + functional refinement`，输出 SFUR audit 与零硬违规结果。
