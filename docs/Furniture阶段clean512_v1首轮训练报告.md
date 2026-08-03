# Furniture 阶段 clean512_v1 首轮训练报告

## 1. 运行结果

| 项目 | 结果 |
|---|---|
| 服务器 | `sceneexpert-cci` |
| GPU | 2 x NVIDIA H100 80GB HBM3，DDP 两 rank |
| Clean scene | 512 |
| Split | train 409 / val 51 / test 52 |
| Epoch | 100，全部完成 |
| 累计 epoch 计算时间 | 296.60 s |
| 最佳 checkpoint | `runs/furniture_clean512_v1/best.pt` |
| 最佳 val loss epoch | 92 |
| 最佳 val loss | 0.99717 |
| 最佳 val macro-F1 epoch | 94，0.73988 |

训练 master seed 为 `3214581895698610966`，已写入 `run_config.json` 和 checkpoint。每个 train clean scene 每 epoch 在线生成 16 个完整联合扰动；validation 每 scene 固定 4 个扰动。

## 2. 固定 test 结果

test 使用 seed `20260802`，52 个 scene 各生成 16 个联合扰动，共 832 个 corrupted scene、5,456 个家具监督。

| 指标 | 结果 |
|---|---:|
| Action accuracy | 0.72379 |
| Action macro-F1 | 0.72161 |
| Translation MAE | 0.50993 m |
| Yaw MAE | 0.23524 rad |
| Whole-scene action success | 0.13221 |

按需要修复家具数 K 分组：

| K | 场景数 | Action macro-F1 | Whole-scene action success | Translation MAE | Yaw MAE |
|---|---:|---:|---:|---:|---:|
| 0 | 0 | 无随机样本 | 无随机样本 | - | - |
| 1 | 7 | 0.94545 | 0.85714 | 0.34848 m | 0.07420 rad |
| 2-3 | 155 | 0.67780 | 0.17419 | 0.50111 m | 0.24050 rad |
| 4-5 | 408 | 0.71361 | 0.13725 | 0.51754 m | 0.22112 rad |
| 6-7+ | 262 | 0.72942 | 0.08015 | 0.50482 m | 0.24881 rad |

完整 confusion matrix 和计数见远端 `runs/furniture_clean512_v1/test_metrics.json`。

## 3. 结论

首轮训练证明以下链路已经成立：

- accepted JSON 可以稳定适配为新版三节点、四边图；
- 全场家具可以在同一个样本中独立随机抽取 action；
- 两张 H100 能通过 DDP 完成在线建图、BF16 训练和 checkpoint；
- `K=6/7+` 在 test 中有 262 个样本，不存在只训练单家具错误的问题；
- 高 K 的逐家具 action macro-F1 没有明显崩溃。

但该 checkpoint 不能视为部署合格模型：

- `K=6/7+` 的整场 action 全正确率只有 8.0%；
- 平移 MAE 约 0.51 m，仍然偏大；
- 当前评估只检查 action 与 clean delta，尚未执行修复后的碰撞、越界、opening clearance 和功能关系验收；
- 随机 action 过程自然没有抽到 `K=0` test 样本，需要额外的全 clean 只读诊断集验证模型是否误动合理房间；
- 当前 split 是 scene-disjoint，不是严格 HSSD asset-disjoint。

下一轮应先实现执行后几何验收、全 clean 诊断和 asset-disjoint split，再根据错误分析决定是否调整扰动分布、损失或网络，而不是直接增加 epoch。
