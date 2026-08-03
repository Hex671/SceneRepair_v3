# sceneexpert-cci 双 H100 训练执行手册 v1

## 固定路径

```text
服务器：sceneexpert-cci
项目：/mnt/afs/task3_2/visitor36/projects/SceneRepair_v3
数据：项目内 data/clean_layout_production_v2/accepted/scenes
准备产物：项目内 data/training/clean512_v1
训练输出：项目内 runs/furniture_clean512_v1
```

服务器已确认有 2 张 NVIDIA H100 80GB HBM3，CUDA 12.4，系统 Python 中已有 CUDA 版 PyTorch 2.3 开发版本。系统 ONNX 与 protobuf 不兼容，项目使用 `.python_packages/protobuf==3.20.3` 作为隔离兼容层，启动时必须设置 `PYTHONPATH=.python_packages`，不得修改全局包。

## 数据准备

```bash
python -m tools.prepare_furniture_training_data \
  --scenes-dir data/clean_layout_production_v2/accepted/scenes \
  --output-dir data/training/clean512_v1 \
  --expected-scenes 512
```

成功条件：512 个 JSON 的 SHA-256 被写入 manifest，512 个场景全部可构建完整图，报告中 `validated_graphs=512`。

## 双卡冒烟

```bash
PYTHONPATH=.python_packages python3 -m torch.distributed.run \
  --standalone --nproc_per_node=2 -m tools.train_furniture \
  --manifest data/training/clean512_v1/clean512_manifest_v1.json \
  --vocab data/training/clean512_v1/furniture_vocab_clean512_v1.json \
  --functional-rules configs/functional_partner_rules_hssd_v1.json \
  --compatibility-rules configs/functional_partner_category_compatibility_v1.json \
  --output-dir runs/furniture_clean512_v1_smoke \
  --epochs 1 --batch-size 8 --workers 2 --max-batches 2
```

成功条件：两个 NCCL rank 启动，训练与验证均完成，写出 `run_config.json`、`metrics.jsonl`、`last.pt` 和 `best.pt`。

## 正式训练

```bash
bash tools/launch_sceneexpert_cci_training.sh
```

训练启动后记录 PID，并检查：

```bash
nvidia-smi
tail -f runs/furniture_clean512_v1.stdout.log
```

不根据短时 GPU utilization 直接判断训练异常。首先检查每 epoch 时长、DataLoader 等待、两个 rank 是否存活和 metrics 是否持续写入。

## 首轮状态

`furniture_clean512_v1` 已在 2026-08-01 完成 100 epoch。训练、最佳 checkpoint 和固定 test 指标分别位于：

```text
runs/furniture_clean512_v1/metrics.jsonl
runs/furniture_clean512_v1/best.pt
runs/furniture_clean512_v1/last.pt
runs/furniture_clean512_v1/test_metrics.json
```

结果解释见 `docs/Furniture阶段clean512_v1首轮训练报告.md`。
