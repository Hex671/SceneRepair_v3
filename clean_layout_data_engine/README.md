# 纯净布局数据管线 v1

## 目标

本管线用于验证“模型负责逐场景布局决策，代码负责确定性编译和验证”的数据生产方式。当前版本是单场景 pilot，不会把结果直接写入正式训练集。

Python 可以决定房间 brief、开口、候选 HSSD 资产和验证规则，但不得用固定模板决定家具位置、朝向或功能分组。家具的选择、位置和布局意图由 API 模型逐场景给出。

## 数据流

1. `run_pilot.py` 根据房间类型和随机种子生成房间尺寸、门窗 brief。
2. 从 HSSD lookup 中提取候选资产卡，包含冻结类别、真实 bbox 和已有功能关系标注，并优先选择正式干净集中尚未出现的资产。
3. API 模型返回严格 JSON：家具位置、朝向意图、使用净空意图、路径目标及逐物件理由。模型不得直接输出 yaw、净空坐标或 `FUNCTIONAL_PARTNER` 边。
4. `orientation_intent_compiler.py` 按 HSSD local `-Y` front 合同，将语义朝向编译成几何 yaw，将净空意图编译成确定性矩形区域。
5. 编译器执行语义合同，拒绝以下几类“几何可能通过、语义不可用”的结果：
   - 座椅朝向边桌、床头柜或落地灯；
   - 边桌、茶几、脚凳或落地灯拥有独立使用净空；
   - 用未经批准的 `allowed_overlap_object_ids` 绕过净空碰撞。
6. 现有 authored-scene validator 检查 schema、HSSD metadata、bbox、边界、碰撞、门窗净空、路径、使用净空、模型输入字段和多样性。
7. `FUNCTIONAL_PARTNER` 只由冻结规则构建，API 输出中不得手写功能边。
8. 编译或硬验证失败时，结构化错误会反馈给同一 author，最多尝试配置中的次数。

## 文件

- `config.json`：本地 API 密钥和连接参数；已被 Git 忽略。
- `config.example.json`：不含真实密钥的配置格式样例。
- `pipeline_config.json`：房间类型、候选类别和重试次数。
- `run_pilot.py`：API author、自动返修和 staging 编排。
- `orientation_intent_compiler.py`：朝向/净空编译及语义合同。
- `tests/`：配置读取、坐标合同和语义合同单元测试。

## 运行

先只在本目录的 `config.json` 中填写 API 信息，然后从 `SceneRepair_v3` 根目录运行：

```powershell
D:\anaconda\envs\SceneRepair_v2\python.exe -m clean_layout_data_engine.run_pilot `
  --scene-id home_office_intent_pilot_002 `
  --room-type home_office `
  --output-root data\orientation_intent_pilot_v2 `
  --seed 20260730
```

引擎只在运行时读取配置并构造 API 客户端，不打印配置内容，也不把配置复制到日志或场景产物。`config.json` 是本机明文秘密文件，不应提交、发送或用于其他用途。

## 晋级条件

`pilot_status.json` 的 `hard_valid_pending_independent_review` 只表示机器硬验证通过，不能直接进入训练集。正式接收仍需：

1. 独立审核者逐物件检查位置、朝向、功能组和人类使用习惯；
2. 审核通过后写入不可变 accepted manifest；
3. 不通过时必须形成具体问题并重新生成或返修，不允许审核脚本静默移动家具；
4. 生产批处理只并发独立的单场景任务，不把三个场景交给一次模型输出。

## 正式批量引擎

`batch_engine.py` 是正式批量入口。它本身不解析 API 配置内容，只把 `config.json` 路径传给相互隔离的 author/reviewer 子进程。

一个场景只有同时满足以下条件才会进入 `accepted/scenes`：

1. author 严格 JSON 输出成功；
2. 朝向、净空和语义合同编译成功；
3. schema、HSSD、边界、碰撞、开口、路径、使用净空、功能关系覆盖和模型输入合同全部通过；
4. 独立的新 API response 在看不到作者 placement rationale 的 reviewer packet 上逐物件审核通过；
5. 按 batch plan 固定顺序，与旧 clean scenes 和此前 accepted scenes 重新执行多样性检查并通过。

Reviewer 只能给出 `pass/revise/reject` 和问题说明，不能修改场景。`revise/reject` 会淘汰当前候选，由候选池中的新场景补位。

### 仅创建计划

此命令不读取 `config.json`，也不联网：

```powershell
D:\anaconda\envs\SceneRepair_v2\python.exe -m clean_layout_data_engine.batch_engine `
  --batch-id clean_v1_batch_0001 `
  --output-root data\clean_layout_production `
  --target-count 24 `
  --plan-only
```

如需明确房型配额，可替换 `--target-count`：

```powershell
  --room-count bedroom=6 `
  --room-count living_room=6 `
  --room-count dining_room=6 `
  --room-count home_office=6
```

### 启动或断点续跑

填写本地 `config.json` 后，去掉 `--plan-only`。同一个命令重复执行会读取不可变的 `batch_plan.json` 并从各场景状态继续，不会重新生成已经 accepted 的场景。

```powershell
D:\anaconda\envs\SceneRepair_v2\python.exe -m clean_layout_data_engine.batch_engine `
  --batch-id clean_v1_batch_0001 `
  --output-root data\clean_layout_production `
  --target-count 24
```

同一生产根目录只允许一个 batch coordinator。当前网关的保守默认值为单连接、单场景波次；连续小批次稳定后再逐步提高 `production_config.json` 的 `max_workers` 和 `wave_size`。崩溃残留锁只有在确认没有进程运行后才能使用 `--recover-stale-lock`。

### 只读状态

```powershell
D:\anaconda\envs\SceneRepair_v2\python.exe -m clean_layout_data_engine.batch_engine `
  --batch-id clean_v1_batch_0001 `
  --output-root data\clean_layout_production `
  --status-only
```

### 产物结构

- `batches/<batch_id>/batch_plan.json`：确定性场景 ID、随机种子、配额和合同文件哈希。
- `batches/<batch_id>/jobs/`：每个候选的恢复状态和运行次数。
- `batches/<batch_id>/staging/`：intent、编译结果、验证、可视化和盲审包。
- `batches/<batch_id>/logs/`：各子进程日志，不包含 API 配置内容。
- `batches/<batch_id>/manifests/`：本批 accepted/rejected/pending JSONL。
- `accepted/scenes/`：最终可训练 clean scenes。
- `accepted/manifest.jsonl`：跨批次 accepted 清单和 SHA-256。

### 2000 场景生产

不要把 2000 个目标放入单个不可审计的大批次。建议每批目标 24 到 50 个，使用不同 batch ID 串行提交；所有批次共享 `accepted/scenes`，因此后续批次会自动与此前结果做跨批多样性检查。最终以全局 `accepted/manifest.jsonl` 的有效、哈希一致记录数作为完成数量，而不是以 API 调用数或候选数计算。

训练前必须执行全局完整性审计：

```powershell
D:\anaconda\envs\SceneRepair_v2\python.exe -m clean_layout_data_engine.verify_dataset `
  --output-root data\clean_layout_production `
  --minimum-scenes 2000
```

只有 `audits/dataset_integrity.json` 中 `passed=true` 的 accepted 数据集才允许交给扰动和训练管线。审计会检查场景哈希、全部硬验证字段、盲审、多样性证据、重复 manifest 记录和未登记场景文件。

正式 2000 场景生产应使用新的空输出根目录。`data/clean_layout_production`
包含引擎迭代期间不同合同版本生成的资格验证样本，只用于审计和人工查看，不应直接作为训练集起点。例如首批正式生产使用：

```powershell
D:\anaconda\envs\SceneRepair_v2\python.exe -m clean_layout_data_engine.batch_engine `
  --batch-id clean_v1_prod_0001 `
  --output-root data\clean_layout_production_v1 `
  --target-count 24
```

## 2026-07-29 正式生产资格验证

`clean_v1_batch_0011` 和 `clean_v1_batch_0012` 在源代码和全部 JSON 合同哈希固定且运行期间无漂移的条件下，分别接受了 1 个客厅、1 个餐厅、1 个家庭办公室和 1 个卧室场景。四者均通过硬验证、独立语义盲审和跨批次多样性检查。接受场景分别包含 6/5/6/5 件家具和 10/3/5/2 条由冻结 HSSD 规则生成的功能关系；餐厅使用的 HSSD 餐桌 bbox 高度为 `0.76 m`。

生产默认值已经固定为单 API 连接、单候选波次和 `3.0x` 候选容量。API 运行时错误不消耗候选的生成/审查质量预算，并按 15/30/60 秒退避；连续 3 个完整失败波次才暂停批次。批次计划同时固定 JSON 合同与 5 个运行时源文件的 SHA-256，恢复或运行期间发现漂移会在继续调度前停止。

当前硬语义合同还包括：沙发-茶几边缘间距 `0.35..0.65 m`、面向茶几的扶手椅间距 `0.35..0.60 m`、餐厅 `table` 资产 bbox 高度 `0.65..0.90 m`，以及互不相关的家具使用净空不得重叠。数据引擎测试为 `34 passed`；资格验证根目录的完整性审计为 `18/18` 场景证据通过。

## 2026-07-29 Pilot 结果

`home_office_intent_pilot_002` 使用 7 个当前干净集未见 HSSD 资产。首次结果仅有书柜越界 `0.2 mm`，第二次根据机器错误自动修正。最终通过全部硬检查和多样性检查，生成 5 条冻结 `FUNCTIONAL_PARTNER` 边，且所有新版 Furniture 网络输入字段可用。

该场景证明当前“生成 -> 编译 -> 语义合同 -> 硬验证 -> 自动返修”链路可运行。它仍处于独立审核前 staging，尚不足以证明四类房间和大规模生产的整体合格率。
