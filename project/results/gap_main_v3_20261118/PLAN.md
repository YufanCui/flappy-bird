# Gap main study v3 · 预先登记（2026-11-18，训练开始前）

老师要求（用户转述）：上下两根管之间的开口大小 (vertical gap size) 作为 randomness；并进一步增加环境随机性。
环境：`env_v3.py`（vendor 不改）；默认原始 reward（flappy-bird-gymnasium 0.4.0，proximity −0.5）；水平间距默认原游戏 144/144/164；gap center 原分布。
`gap100_100` 与恢复的 0.4.0 原文件逐步一致（22,437 步）。

## E1 主实验：开口随机是否改变表现
- Groups：`gap100_100`（原游戏）vs `gap80_120`（每对新 pipe 抽 G~U{80..120}，均值 100）
- Seeds 0–4；301,056 steps/model（300k requested）；每 50k 存 checkpoint；训练 episode 上限 10,000
- PPO：SB3 默认；LiDAR 180 rays

## E2 增加 randomness：开口随机 + 水平间距随机
- Group：`gap80_120+sp120_180`；seeds 0–4；其余同 E1（与 E1 的 gap80_120 对比）

## E1/E2 评估
- 最终 checkpoint；episode seeds 230000–230019；deterministic 与 stochastic；每局上限 10,000 steps
- 条件：gap100_100、gap80_120、gap60_79（更窄，训练外）、gap121_140（更宽，训练外）、gap80_120+sp120_180
- 主要 metric：pipes_cleared（full-body）；分析单位 training seed；显示全部 seed

## E3 改进（目标：飞不停）——在 gap80_120 环境中一次只改一个因素
- `gap80_120`（只增加 budget 到 1M）、`gap80_120+stack4`、`gap80_120+features`、`gap80_120+noprox`
- Seeds 0–2；1,000,000 steps；每 100k 存 checkpoint
- 评估：deterministic，条件 gap80_120、gap60_79、gap121_140、gap100_100；“完美局”= 达到 10,000 步上限（约 270 根）
