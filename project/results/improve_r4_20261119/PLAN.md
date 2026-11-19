# Round 4 · 预先登记（2026-11-19，训练前）

**新环境（两种 randomness，都尽量贴近真实游戏）：**
1. 上下开口大小随机 80–120（老师要求的 randomness）；开口高低保持原游戏的随机；水平距离恢复原游戏（144/144/164）。
2. **sticky actions p = 0.05**：每一步有 5% 的概率，机器不执行这一步选择的动作，而是重复上一步的动作（模拟按键延迟/不精确）。
   p 的选择见 `../sticky_probe_20261119/README.md`。
spec：`gap80_120+sticky5`。

**训练方法：** 目前最好的设定：stack4（叠加 4 帧）+ learning rate 线性衰减 + 3M steps，seeds 0、1、2。

| 组 | reward | 要回答的问题 |
|---|---|---|
| `_orig` | 原版 0.4.0（proximity −0.5，top −0.5） | 在新环境里训练能飞多远？ |
| `_topB` | 6 月 Exp B：top penalty −2（按邮件记录的改动重新实现，原代码未找回） | 你调过的 reward 在新环境里是否更好？ |
| 参照 | round 3 的 `lr3M`（训练时**没有** sticky） | 用 sticky 训练是否比不用更好？（randomness 有没有帮助） |

**评估：** 最终 checkpoint，deterministic，episode seeds 230000–230019；条件：
`gap80_120+sticky5`（主要）、`gap80_120`、`gap100_100`（原游戏）、`gap100_100+sticky5`；
**每局上限 40,000 步（约 1,000 根柱子）**，参照组用同样的上限重新评估。训练时每局上限仍为 10,000 步。
全部 seeds 报告。仍是 development 评估，不是 final test。
