# Diagnostics v1 · 2026-11-18（development data，无 learning）

## 做了什么

- 对象：`gap_retry_20261118` 的全部 9 个 checkpoints（3 groups × seeds 0,1,2），不挑 seed。
- 评估：deterministic policy；两个 conditions：fixed100、random80_120；新的 development episode seeds 220000–220019（每 checkpoint 每 condition 20 局），共 360 episodes、27,567 steps。没有使用曾提议的 validation（200000–）/final（300000–）panel。
- 工具：`diagnose_policies_v1.py`（逐步记录）、`summarize_diagnostics_v1.py`（汇总）。源码快照在 `source/`。
- 逐步记录：action、player_y、vel_y、rot、下一根 pipe 的 x/top/bottom/gap、鸟中心相对 gap center 的 offset、LiDAR 最小距离（pixel）、proximity 来源（pipe/ground）、reward case、crash 类型。
- **Reward case 按 vendor `step()` 的真实执行顺序重建**（score +1 → 否则 proximity −1 → 否则 alive +0.1 → above-screen −0.5 覆盖 → crash −1 覆盖），并在**每一步 assert 重建值 = 环境实际 reward**；27,567 步全部一致。

事先定义（看结果之前）：
- `pipes_cleared` = 鸟的左边缘越过 pipe 右边缘的次数（full-body clearance），与沿用的 midpoint `game_score` 区分。
- Failure taxonomy：`first_pipe_*`（cleared=0 时撞）/ `later_pipe_*`，后缀 upper / lower / ground；truncated 单列。

## 结果（文件：`checkpoint_table.csv`、`episode_taxonomy.csv`、`passages.csv`、`summary.json`）

### 1. 死因：全部是撞 pipe，大多数死在第一根

| Failure | 次数 / 360 |
|---|---:|
| first_pipe_upper | 179 |
| first_pipe_lower | 83 |
| later_pipe_upper | 63 |
| later_pipe_lower | 35 |
| ground / truncated / above-screen | 0 / 0 / 0 |

73%（262/360）的 episodes 一根 pipe 都没完整通过。没有任何 ground crash，没有 truncation（3,000-step cap 在这一批 policies 下尚未构成 censoring）。

### 2. 每个 policy 的撞击方向高度一致（固定的高度偏差）

多数 checkpoint 的死亡几乎全在同一侧，例如 fixed100_seed0 在 fixed100 下 100% 撞 lower pipe（死亡时鸟中心平均在 gap center 下方 +54.6 px），fixed100_seed1、random90_110_seed1/seed2 100% 撞 upper pipe（约 −42 至 −56 px）。gap 半宽是 50 px，所以这些 policy 不是“偶尔偏一点”，而是**以大致固定的高度飞向 pipe**，不随 gap 位置调整。
这是观察到的 pattern；原因（observation 不足、budget 不够或 reward 导致）尚未经过 controlled intervention 验证。

### 3. Midpoint scoring 会高估表现

random90_110_seed0：`game_score` 平均 1.0，但 `pipes_cleared` = 0（20/20 局都是先得 1 分，再在同一根 pipe 里撞 lower pipe）。也就是说，沿用的 midpoint-crossing score 会把“进了 pipe 但没出来”记成 1 分。之前报告的 seed means 中至少这个 checkpoint 的分数并不代表真正通过了 pipe。

### 4. Proximity penalty 主导 reward

360 局的 reward 加总（按 case 分解）：

| Case | 贡献 |
|---|---:|
| alive (+0.1) | +2498.2 |
| score (+1) | +264 |
| proximity (−1) | **−1961** |
| crash (−1) | −360 |

Proximity 的总负贡献是 scoring 正贡献的约 7.4 倍。逐根 pipe 统计（186 次成功通过）：

| 通过时的 gap | n | 平均 proximity 步数 | 零 penalty 比例 |
|---|---:|---:|---:|
| 80–89 | 17 | 8.6 | 0% |
| 90–99 | 23 | 4.6 | 0% |
| 100（fixed100 condition） | 91 | 3.5 | 0% |
| 100–109 | 18 | 3.0 | 0% |
| 110–119 | 31 | 1.8 | 16% |
| 120 | 6 | 0.3 | 67% |

即：在现有 reward 下，**成功通过一根 gap≈100 的 pipe，平均要付出约 −3.5 的 proximity penalty，而通过只得 +1**。对 policy 来说，“通过 pipe”这件事的即时 reward 往往是净负的。

### 5. 这些 penalty 是 policy 没学好，还是结构性的？（feasibility search）

`feasibility_min_proximity_v1.py` 用真实环境做穷举的 dynamic-programming search：pipe 的移动与 action 无关，所以只需枚举鸟的状态 (y, vel, rot, flapped)，求“完整通过第一根 pipe 最少要吃几步 proximity −1”。结果见 `../feasibility_v1_20261118/feasibility.json`：

| Gap | 完整通过第一根 pipe 至少需要的 proximity 步数（seeds 220000–220002） |
|---:|---|
| 80 | 4–6（找到的最优路径是 6；剪枝下只能确定 >3） |
| 90 | 3 |
| 100 | 1 |
| 110 | 0 |
| 120 | 0 |

精确性说明：搜索只剪掉“成本 > 当步最优+3”的状态，因此结果 ≤3 时是精确最小值；>3 时只给出上界。三个 seeds 结果相同。

**结论：在原 reward 下，gap ≤100 时，任何 policy（包括完美 policy）都不可能零 penalty 地通过 pipe；gap 越小，结构性 penalty 越多。** 所以 random gap 训练时，小 gap 的 pipe 除了更难，还附带更多“不可避免的负 reward”——gap 和 reward 在当前环境里是纠缠在一起的。这一点在解释 randomization 结果时必须考虑。

### 6. 对照：你 6 月的历史 checkpoints（`../historical_eval_v1_20261118/`）

同一环境（gap=100）、同一 episode seeds、同一 logger，只评估不训练：

取消 3,000-step 上限后（cap 20,000，无任何 truncation；`../historical_eval_uncapped_v1_20261118/`）：

| Checkpoint | 训练 steps | deterministic mean / median / max | stochastic mean / median / max |
|---|---:|---|---|
| flappy_model | 100k | 30.9 / 21 / 96 | 10.5 / 9 / 38 |
| flappy_model_300k | 300k | 26.0 / 18.5 / 80 | 10.7 / 9 / 34 |
| flappy_model_top_penalty_300k（Exp B，top −2） | 300k | 44.7 / 10.5 / **228** | 21.7 / 17.5 / 112 |
| flappy_model_obstacle_penalty_300k（Exp C，prox −1 = 当前 vendor reward） | 300k | 14.6 / 9 / 77 | 9.5 / 8 / 26 |
| gap_retry fixed100（3 seeds 平均，cap 3,000） | 65k | 1.2 | — |

3,000-step cap 会截断高分（约 79 根封顶），所以 `historical_eval_v1_20261118/`（cap 3,000）里 top_penalty 的分数被低估了。Exp B model 的分布极不均匀：20 局中 8 局 ≤2 分，但最高 228 分，与你邮件中“最好 132、但不稳定”的描述一致（当时用 stochastic 评估）。

- 即使是 **reward 与当前 vendor 相同**的 obstacle_penalty_300k，也能平均通过约 14 根 pipe；而 gap_retry 的 65k-step models 平均约 1 根。
- 历史 models 的训练 seed 未记录、每个条件只有 1 个 model，不能当成 matched comparison；但差距是一个数量级，最直接的解释候选是 **training budget 不足**（65k vs 100k–300k）。这仍是假设，需要 controlled budget pilot 验证。
- 你 6 月邮件里报告的分数（如 43、132）当时用的是 `test_flappy.py` 的默认 `model.predict(obs)`，即 **stochastic** action；gap_retry 用的是 deterministic。两者不能直接比较。

### 7. 环境版本的来源（重要）

你 6 月邮件记录的原始 reward 是 proximity −0.5；Experiment C 把它改成 −1，并观察到表现变差。当前冻结的 vendor 代码 proximity 是 −1、top 是 −0.5，也就是说 **之前一轮（2026 年 11 月初）的所有 gap 实验都建立在 Experiment C 的 reward 上**，而不是原始 baseline reward。这不是错误，但论文里必须写清楚。

## 解读边界

- 全部是 development data；3 个 training seeds；没有统计检验。
- 以上是描述性 diagnostics，尚未证明 proximity penalty **导致**了 weak learning。要支持因果说法，必须做下面的 controlled pilot。

## 建议的下一个 controlled pilot（尚未运行）

**首选：Budget diagnostic。** 历史对照显示 budget 可能是最大的单一因素，而且很便宜（65k steps 约 44 秒，300k 约 4 分钟/model）。

- Hypothesis：在 reward、observation、PPO 超参数都不变时，把训练从 65,536 增加到 ~300k steps，fixed100 与 random80_120 两组的 `pipes_cleared` 都会显著提高；只有在 baseline 学会之后，比较 fixed vs random 才有意义。
- 改变的变量：training steps（同一 run 中保存 65k/100k/200k/300k 中间 checkpoints）。
- 保持不变：当前 vendor reward（prox −1）、LiDAR、gap 设定、PPO 默认超参数。
- Groups：fixed100、random80_120；seeds 0–4。
- 评估：新的 development panel；`pipes_cleared` 为主，deterministic 与 stochastic 分开报告。

**其次：Reward diagnostic（在 budget 足够之后再做）：proximity penalty 是否阻碍学会通过 pipe。**

- Hypothesis：在其他条件不变时，降低 proximity penalty（−1 → 0）会提高 fixed100 训练下的 `pipes_cleared`。
- 改变的唯一变量：proximity case 的 reward 值（新的 versioned environment module，例如 `random_gap_env_v2.py`，vendor 不动）。
- 保持不变：PPO 超参数、LiDAR observation、gap=100、65,536 steps、seeds、episode cap、score/alive/crash/above-screen reward。
- Groups：prox−1（现状，重新训练以保证 matched）vs prox0；可选 prox−0.1 作为中间档。
- Seeds：0–4（每组 5 个），保存中间 checkpoints。
- 评估：新的 development panel，**主要 metric 用 `pipes_cleared`**（同时报告 game_score、return、failure taxonomy），所有 seeds 都报告，包括 0。
- 注意：去掉 penalty 后 return 的尺度会变，所以不同组之间只比较 game metric，不比较 return。
