# Improvement round 3 · 预先登记（2026-11-19 00:00，训练前）

**背景：** Round 2 中唯一明显有效的改动是 learning rate 线性衰减（`_lr`）：随机开口 80–120 上 49.9 根，
比 stack4_1M baseline 的 22.7 高一倍多；而 3M 常数 learning rate 并没有更好（训练曲线不稳定）。

目标环境不变：`gap80_120`（随机开口 80–120，原始 reward，原游戏水平间距）。spec 均为 `gap80_120+stack4`。

| 变体 tag | 改动 | 训练量 | seeds | 要回答的问题 |
|---|---|---|---|---|
| `_lr3M` | LR 线性衰减 + 训练量 3M | 3M | 0,1,2 | LR 衰减时，更多训练能否继续提升？ |
| `_lr` | LR 线性衰减（与 round 2 相同） | 1M | 3,4,5 | round 2 的 LR 效果在**新 seeds** 上能否复现？ |
| `_base` | 无改动（= E3 stack4_1M） | 1M | 3,4,5 | 新 seeds 上的对照组 |

评估：最终 checkpoint，deterministic，episode seeds 230000–230019；条件 gap80_120（主要）、gap100_100、gap121_140、gap60_79；
每局上限 10,000 步。全部 seeds 报告，不挑最好的 seed。
注意：所有之前的评估结果都是 development data；这里仍然不是 final test。
