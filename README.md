# SceneRepair_v3

Relation-aware Furniture-stage scene repair and clean-layout data tooling.

This repository contains the independently runnable v3 implementation, its
data contracts, validation/training utilities, tests, and design records. It
does not contain private API credentials, training checkpoints, or the large
production batches used during internal experiments.

## Repository layout

- `scene_repair_v3/`: model, graph contracts, datasets, losses, and checkpoint validation.
- `clean_layout_data_engine/`: deterministic compilation, validation, and batch orchestration for clean layouts.
- `tools/`: training, evaluation, smoke-test, visualization, and dataset utilities.
- `tests/` and `clean_layout_data_engine/tests/`: unit and contract tests.
- `configs/`: frozen vocabulary, geometry, and functional-partner rules.
- `docs/`: architecture, experiment, and acceptance records.
- `fig/` and `docs/visualizations/`: documentation figures and small prediction galleries.
- `data/clean_layouts_gpt56_v1/`, `data/clean_layouts_gpt56_v2_authored/`, and `data/orientation_intent_pilot_v1/`/`v2/`: small pilot/example datasets.

Large production data under `data/clean_layout_production*/` and model outputs
under `runs/` remain local. To reproduce training or evaluation, place the
corresponding dataset and checkpoint paths outside the repository and pass
them through the command-line arguments documented below.

## Setup

Python 3.10-3.12 is supported. Install the package and development tools with:

```powershell
python -m pip install -e ".[dev]"
```

The data engine's `config.json` is intentionally local-only. Copy
`clean_layout_data_engine/config.example.json` to that filename and fill in
your own endpoint credentials; never commit the resulting file.

纯净布局批量数据引擎位于 [clean_layout_data_engine](clean_layout_data_engine/README.md)。正式生产入口为：

```powershell
python -m clean_layout_data_engine.batch_engine --help
```

该目录用于新版 SceneRepair 的独立实现。Furniture 阶段 v1 网络、图数据合同、训练损失和 checkpoint 校验已在 `scene_repair_v3` 包中实现。旧版代码和实验结果仍保留在 `SceneRepair_v2`，没有复制进新包。

Furniture v1 默认主干为 `128 hidden / 4 heads / 32 per head / 2 layers`。

## Furniture v1 设计入口

- [新版网络结构设计](docs/Furniture阶段新版网络结构设计_v1.md)
- [网络实现说明](docs/Furniture网络实现说明_v1.md)
- [训练数据构造与训练计划](docs/Furniture阶段训练数据构造与训练计划_v1.md)
- [训练数据构造与训练计划 v2](docs/Furniture阶段训练数据构造与训练计划_v2.md)
- [sceneexpert-cci 双 H100 执行手册](docs/sceneexpert-cci双H100训练执行手册_v1.md)
- [clean512_v1 首轮训练报告](docs/Furniture阶段clean512_v1首轮训练报告.md)
- [observable_v5 达标训练报告](docs/Furniture阶段observable_v5训练报告.md)
- [joint_v6 联合修复训练报告](docs/Furniture阶段joint_v6训练报告.md)
- [改进措施与决策记录](docs/Furniture阶段网络改进措施_v1.md)
- [新版网络结构图](docs/visualizations/furniture_model_architecture_v1.svg)
- [旧版实际结构图（历史对照）](docs/visualizations/furniture_model_v2_actual_architecture_legacy.svg)

## 快速验证

```powershell
python -m pytest -q
python -m tools.smoke_furniture_network --max-translation-m 0.5 --max-yaw-deg 60
```

`clean512_v1` 已冻结 512 个 accepted clean scene、完整 category 词表和训练动作范围。训练入口为 `python -m tools.train_furniture --help`，双卡使用 `torchrun --nproc_per_node=2`。

后续实现以 v3 文档为唯一设计基线；若设计继续变更，应先更新本目录合同，再修改代码和训练数据。
