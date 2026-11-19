# Improvement round 2 · 预先登记（2026-11-18 22:25，训练前）

目标环境不变：`gap80_120`（随机开口 80–120，原始 reward，原游戏水平间距）。
新 baseline：`gap80_120+stack4`，1M steps（E3 中最好：22.7 根，seeds 0–2）。
每个变体只在 baseline 上加 **一个** 改动；seeds 0,1,2；每 100k 存 checkpoint。

| 变体 tag | 唯一改动 | 训练量 |
|---|---|---|
| `_lr` | learning rate 从 3e-4 线性衰减到 0（PPO 稳定性） | 1M |
| `_cur` | curriculum：前 50% 训练中开口下限从 120 线性降到 80，之后保持 80–120 | 1M |
| `_ent` | ent_coef 0.01（鼓励探索；SB3 默认 0） | 1M |
| `_big` | policy/value 网络 128×128（默认 64×64） | 1M |
| `_3M` | 训练量 3M | 3M |

评估：最终 checkpoint，deterministic，episode seeds 230000–230019；条件 gap80_120（主要）、gap100_100、gap121_140、gap60_79；每局上限 10,000 步。
比较对象：E3 的 `gap80_120+stack4_1M`（同 seeds）。全部 seeds 报告。
