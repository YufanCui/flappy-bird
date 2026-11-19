# Round 5 · 预先登记（2026-11-19，训练前）

**环境不变：** 开口随机 80–120 + 5% 按键延迟（`gap80_120+sticky5`），原版 reward。
**背景：** Round 4 的 agents 约 20 根；理论上限（知道柱子位置的最优策略）前 500 根内平均约 425 根（`../theory_sticky_v1_20261119/`）。
瓶颈在 agent。本轮检验两个假设（都只是假设）：
- H1 信息不足：agent 不直接知道“上一步实际执行的动作”，而延迟时下一步可能重复它。→ `prevact`：把它加进 observation。
- H2 训练不够：延迟让学习信号更嘈杂，3M 可能不够。→ 10M steps。

| 组 tag | spec | 训练量 | 对照 |
|---|---|---|---|
| `_prev` | `gap80_120+sticky5+stack4+prevact` | 3M | round 4 `_orig`（3M，20.1） |
| `_10M` | `gap80_120+sticky5+stack4` | 10M | round 4 `_orig` |
| `_prev10M` | `gap80_120+sticky5+stack4+prevact` | 10M | 以上两组 |

全部：learning rate 线性衰减，seeds 0、1、2。
评估：最终 checkpoint，deterministic，episode seeds 230000–230019，每局上限 40,000 步；
条件 `gap80_120+sticky5`（主要）、`gap80_120`、`gap100_100`、`gap100_100+sticky5`。全部 seeds 报告。仍是 development 评估。
