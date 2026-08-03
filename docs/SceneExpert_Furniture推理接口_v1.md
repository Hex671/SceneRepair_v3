# SceneExpert Furniture 推理接口 v1

## 1. 正式部署配置

当前正式模型是 `furniture_clean512_sfur_v10_final`。`FurnitureRepairEngine`
从 checkpoint 读取并执行以下冻结配置：

- `action_policy = dense_delta`
- `semantic_passes = 4`
- 模型内硬约束修正
- 模型内 SFUR 功能细化
- `ActionRange = 3m / pi`

本地 checkpoint：`runs/furniture_clean512_sfur_v10_final/best.pt`。
其 SHA256 为
`90BE006F2F6596F042FF457310D1D83429D6822854E17989DE84A2A34FE9CCFD`。

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

`--input` 可接收一个或多个 JSON 文件或目录。默认拒绝覆盖已有输出；
确认重跑时显式使用 `--overwrite`。

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
```

输入对象不会被原地修改。

## 4. 输入合同

继续使用 `stage = furniture` 的单房间 JSON，家具数必须满足 `F <= 17`。
除 ROOM、OPENING、FURNITURE 的标准几何和语义字段外，SFUR 模式明确要求：

```text
audit.use_clearance_zones
audit.path_targets
```

两个字段必须存在，允许是空数组。它们会在输入姿态处转换为家具局部合同，随后随家具
移动和旋转，不读取 authored clean 的绝对姿态。

若缺少这两个字段，SFUR 模式直接报错，不会静默降级并声称功能修复已经执行。
不传 `--sfur-rules` 时仍可运行旧的纯硬约束兼容模式，但该结果不能标记为 SFUR 模型结果。

## 5. 输出与审计

```text
output-dir/
  scenes/<input-stem>.repaired.json
  audits/<input-stem>.repair_audit.json
  repair_manifest.json
```

逐场 audit 包含：

- checkpoint 路径、SHA256 和发布元数据；
- `dense_delta / 4 passes / functional refinement` 部署配置；
- 修复前、神经输出和最终硬约束计数；
- 功能细化迭代数、收敛状态和 SFUR 子分数；
- 每件家具的最终动作、累计 delta 和前后姿态。

引擎拒绝残余硬违规和超出 checkpoint `ActionRange` 的结果。功能细化未收敛会在
audit 中如实记录，不会伪装成 SFUR 通过。

## 6. 已验证结果

冻结测试为 52 个未见 clean 场景，每场 16 个固定扰动，共 832 场景，seed
`20260802`。正式模型达到：

- SFUR：`753 / 832 = 90.5048%`
- 硬约束整场通过率：`832 / 832 = 100%`
- 最大累计平移：`2.9981m`
- 最大累计 yaw：约 `pi`

真实 accepted JSON 的命令行 smoke test 已确认正式 checkpoint 能自动加载四轮语义
修复和 SFUR 功能细化，并输出完整 runtime audit。
