# Furniture 阶段 SFUR v10 最终验收报告

## 1. 最终结论

`furniture_clean512_sfur_v10_final` 已超过冻结目标 `SFUR > 90%`：

```text
SFUR                       753 / 832 = 90.5048%
hard-constraint pass      832 / 832 = 100.0000%
mean functional score                  0.967883
functional relation pass               0.990986
use-clearance pass                      0.935651
core reachability                      0.976042
```

正式 checkpoint：`runs/furniture_clean512_sfur_v10_final/best.pt`

checkpoint SHA256：`90be006f2f6596f042ff457310d1d83429d6822854e17989de84a2a34fe9ccfd`

冻结评估报告：`runs/furniture_clean512_sfur_v10_final/test_sfur_v2.json`

## 2. 冻结测试合同

- clean512 split：409 train / 51 val / 52 test。
- 测试规模：52 个从未参与训练的 test clean scenes，每场 16 个确定性扰动，共 832 场景。
- seed：`20260802`。
- 标签模式：`semantic_restore_v1`。
- 动作解码：`dense_delta`。
- 神经语义迭代：4 pass，每轮重新构图。
- 完整模型：神经网络 + 模型内硬约束层 + 模型内功能细化层。
- 最大动作范围：累计平移 3 m，累计 yaw 为 pi。
- 验收必须通过单次正式 `FurnitureRepairNetwork.repair_batch()`，不允许评估脚本在模型外补投影。

## 3. 分扰动家具数量结果

| 需要恢复的家具数 K | 场景数 | SFUR |
|---|---:|---:|
| 1 | 7 | 100.00% |
| 2-3 | 144 | 95.14% |
| 4-5 | 415 | 88.43% |
| 6-7+ | 266 | 90.98% |

整体指标已达标，但 K=4-5 子组仍低于 90%，是后续继续优化时最明确的薄弱区间。

## 4. 运行时审计

```text
max cumulative translation             2.998122 m
max absolute cumulative yaw             pi
all actions within ActionRange          true
residual hard-constraint scenes         0
residual hard-constraint count          0
mean functional refinement iterations   0.326923
max functional refinement iterations    4
functional refinement unconverged       79
```

79 个未收敛场景正好对应 832 - 753 个 SFUR 失败场景，没有被硬约束清零或报告逻辑掩盖。

残余违反项计数：

```text
functional relation   28
use clearance        277
reachability          67
```

当前主要误差来源是使用净空，其次是路径可达性。

## 5. 发布合同

最终 checkpoint 内写入：

```text
release_id                         furniture_clean512_sfur_v10_final
deployment action policy           dense_delta
semantic passes                    4
functional refinement              true
requires functional contract       true
sfur rule schema                    scene_repair_v3_sfur_rules_v1
```

发布工具会拒绝 SFUR 不高于 90%、硬约束通过率不为 100%、checkpoint 哈希不匹配或评估配置不是 dense_delta + 4 pass + functional refinement 的候选模型。

## 6. 验证证据

- 本地全量自动测试：105 passed。
- 远端目标回归测试：18 passed。
- 最终 checkpoint 在自身 SHA256 下完整重跑 832 场景，结果仍为 90.5048%。
- SceneExpert JSON-to-JSON 入口已使用最终 checkpoint 做 GPU BF16 冒烟测试，自动选择 4 pass SFUR profile，输出硬违规为 0、SFUR 为 1.0。
- 最新 30 个示例由最终 checkpoint 重新生成，包含 23 个通过和 7 个失败案例，不做成功样本筛选伪装。

## 7. 能力边界

该结果证明模型在冻结合同定义内达到整场功能可用率 90.50%，不证明所有输出都符合完整人类审美或主观布局习惯。HLAR 当前明确不在本轮验收范围内。
