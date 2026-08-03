# Furniture 阶段训练数据构造与训练计划 v1

> 历史文档：随机 action 直接作为监督的部分已被实验否定。当前合同见 `Furniture阶段训练数据构造与训练计划_v2.md`。

| 项目 | 冻结值 |
|---|---|
| 文档状态 | v1 训练基线 |
| Clean 数据版本 | `clean512_v1`，512 个 accepted 单房间场景 |
| 场景切分 | train 409 / val 51 / test 52，先按 clean scene 切分 |
| 网络 | 128 hidden / 4 heads / 2 layers |
| 模型输出 | 每件家具四分类 action + `[dx,dy,dyaw]` |
| 训练输入 | 仅由最终 corrupted scene 重新计算的节点和边特征 |

## 1. 学习目标

网络需要从当前房间的几何、类别、行为标签和关系图中判断每件家具是否应保持，并给出需要修复家具的平移和旋转残差。训练目标不是记忆 HSSD asset ID，也不是识别某个人工违规生成器的固定模式，而是学习以下可迁移关系：

- 家具与房间边界的合理关系；
- 家具与 opening clearance 的关系；
- 家具之间的空间分离、相对位置和相对朝向；
- HSSD 已标注的 `FUNCTIONAL_PARTNER` 在具体场景中的相对关系；
- 同一家具在合理位置与被扰动位置之间的区别。

## 2. Clean 数据合同

训练源固定为 `data/clean_layout_production_v2/accepted/scenes` 中冻结清单记录的 512 个 JSON。清单逐文件保存 SHA-256；任何文件变化都会使数据加载失败，禁止训练过程中静默加入后来生成的场景。

准备产物位于 `data/training/clean512_v1`：

- `clean512_manifest_v1.json`：场景级 train/val/test 切分和逐文件 SHA-256；
- `furniture_vocab_clean512_v1.json`：从 512 场景冻结的无碰撞词表；
- `preparation_report.json`：房型、家具数量、category 和功能边统计。

切分必须发生在扰动之前。同一个 clean scene 的不同 corrupted 版本只能属于同一个 split。

## 3. 在线整场联合扰动

正式训练主体是完整房间联合扰动，不是单家具样本。每次访问 clean scene 时，对所有可移动家具独立抽取两个随机比特：

```text
move rotate -> action
  0      0  -> KEEP
  1      0  -> TRANSLATE
  0      1  -> ROTATE
  1      1  -> BOTH
```

不预先指定每个房间的错误家具数量 `K`，也不设置房间级扰动强度。一个样本由完整 action 向量 `A=[a1,a2,...,aN]` 定义。因此同一次前向传播可包含 0 到 N 件需要修复的家具，包括 6 或 7 件同时需要修复。

这里不存在“没有概率分布”的随机过程；两个无偏随机比特会自然形成四类 action 的等边际分布。该设计不按房间强制配额，但训练必须记录实际 action 与 K 分布，防止模型借助经验先验走捷径。

随机性合同：rank 0 使用操作系统安全随机源生成 master seed，随后广播到所有训练 rank。每个样本的实际 seed 由 master seed、epoch、scene index 和 corruption index 确定。master seed 写入 run config 和 checkpoint，使首次抽样不可预测、事后可以精确重放。seed 和 action 向量不得进入模型输入。

## 4. 连续扰动

平移：

```text
distance  ~ Uniform(0, 2.0m)
direction ~ Uniform(-pi, pi)
dx = distance * cos(direction) + eps_x
dy = distance * sin(direction) + eps_y

eps_x, eps_y ~ TruncatedNormal(
  mean=0, std=0.067m, bounds=[-0.2m, 0.2m]
)
```

旋转：

```text
rotation_limit ~ Categorical(
  0.10pi: 40%, 0.25pi: 30%, 0.50pi: 20%, 1.00pi: 10%
)
base_dyaw ~ Uniform(-rotation_limit, rotation_limit)
eps_yaw ~ TruncatedNormal(
  mean=0, std=0.05pi/3, bounds=[-0.05pi, 0.05pi]
)
```

所有家具独立采样。`KEEP` 不添加位姿噪声；`TRANSLATE` 只改变 XY；`ROTATE` 只改变 yaw；`BOTH` 同时改变。最终 yaw wrap 到 `[-pi,pi)`。

## 5. 输入重建与监督

最终 corrupted pose 确定后，必须重新构建完整图：`ROOM`、`OPENING`、`FURNITURE` 节点，双向 `ROOM_MEMBERSHIP`，Furniture 无自环完整有向 `FURNITURE_SPATIAL`，从冻结 HSSD 规则重新生成的 `FUNCTIONAL_PARTNER`，以及 `OPENING -> FURNITURE` 的 `OBJECT_OPENING`。

禁止把 clean 坐标、采样 action、扰动向量或 clean 图边特征写入输入。监督只由最终 pose 差计算：

```text
target_dx   = x_clean - x_corrupted
target_dy   = y_clean - y_corrupted
target_dyaw = wrap(yaw_clean - yaw_corrupted)
```

动作类别由最终有效 delta 产生，而不是盲信最初抽样结果。当前阈值为平移 2 cm、旋转 1 degree。模型的显式回归范围冻结为逐轴 `2.2m` 和 yaw `pi`。

## 6. 多解风险

纯随机扰动自然产生碰撞、越界、opening 侵入及功能关系退化，不再按违规类型定向移动家具。但大幅联合扰动可能使原 clean pose 不再是唯一合理答案。因此：

- v1 先保留随机样本并统计“扰动后仍完全合理”的比例；
- 不以 `K=6/7` 本身作为过滤理由；
- 若 pilot 证明大量样本存在监督歧义，再引入版本化、可解释的可辨识性门禁；
- 禁止用手写目标违规方向替代可辨识性检查。

## 7. 训练方法

每个 batch 包含多个完整 corrupted scene。图经两层 relation-aware Graph Transformer 后，一次输出所有家具：

```text
action_type_logits[F,4]
primary_delta[F,3]
```

损失为：

```text
L = L_action + L_translation + L_yaw + 0.02 * L_keep_zero
```

- action 使用四分类交叉熵，不预设 class weight；
- translation 只在 `TRANSLATE/BOTH` 上计算 Smooth-L1；
- yaw 只在 `ROTATE/BOTH` 上计算周期损失 `1-cos(error)`；
- KEEP 只施加轻量零 delta 正则；
- AdamW，初始学习率 `3e-4`，weight decay `1e-2`，cosine schedule；
- H100 上使用 BF16；
- 默认每个 train clean scene 每 epoch 在线生成 16 个联合扰动，validation 固定生成 4 个。

单家具四状态反事实是可选消融/辅助 warm-up，不属于 clean512_v1 主训练流。只有实验证明联合随机训练存在明确身份捷径时才加入，并单独版本化，不能替代多家具联合训练。

## 8. 双 H100 执行

正式入口使用 PyTorch DDP：

```bash
PYTHONPATH=.python_packages python3 -m torch.distributed.run \
  --standalone --nproc_per_node=2 -m tools.train_furniture \
  --manifest data/training/clean512_v1/clean512_manifest_v1.json \
  --vocab data/training/clean512_v1/furniture_vocab_clean512_v1.json \
  --functional-rules configs/functional_partner_rules_hssd_v1.json \
  --compatibility-rules configs/functional_partner_category_compatibility_v1.json \
  --output-dir runs/furniture_clean512_v1 \
  --epochs 100 --batch-size 64 --workers 8
```

该模型约 54 万参数、每场最多 17 件家具，单步计算很小；双 H100 DDP 可以并行训练，但不应预期高 GPU 利用率。主要瓶颈可能是 CPU 在线建图和小图调度。是否增加每卡 batch、DataLoader workers 或改为多实验并行，必须依据 profiler 数据，而不是仅凭显存空闲决定。

## 9. 验证和接受标准

训练日志至少保存 loss、action accuracy、action macro-F1、translation MAE、yaw MAE、整场 action 全正确率。最终评估还必须增加：

- 按 `K=0,1,2/3,4/5,6/7+` 分组的指标；
- KEEP 误报率与四类 precision/recall；
- 碰撞、越界、opening clearance 和 FUNCTIONAL_PARTNER 恢复率；
- scene-disjoint test；
- HSSD asset-disjoint test；
- 修复执行后的整场成功率，而不只看 delta 回归误差。

在上述评估完成前，训练 loss 下降不等于模型具备可靠布局修复能力。
