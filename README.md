# Flappy Bird under Randomness: PPO, Execution Noise, and an Exact Performance Bound

Supervised research project (LAH 358Q, The University of Texas at Austin), supervised by Professor Haiqing Xu.

This project studies how reinforcement learning agents trained with **Proximal Policy Optimization (PPO)** behave when Flappy Bird is made more realistic by adding randomness. Randomness comes from two sources:

1. **Random vertical opening:** the gap between the upper and lower pipe is drawn independently for every pipe pair from 80–120 px. The original game always uses 100 px.
2. **Execution noise (sticky actions):** at every step, with probability *p*, the game repeats the previously executed action instead of the chosen one. This models imperfect human or hardware input, following Machado et al. (2018).

The project also derives and computes an **exact upper bound** on how far *any* policy can fly. This separates limits imposed by the environment from limits of the learning agent.

## Main results

The metric is the number of pipes **fully** cleared, averaged over 20 fixed evaluation episodes per training seed. Every training seed is reported.

| Experiment | Result |
|---|---|
| **E1: random opening** (5 seeds per group) | Training on random openings is better in the random environment (14.2 vs. 10.3 pipes) but worse in the original game (23.0 vs. 48.5): a trade-off between specialization and robustness. |
| **E2: random horizontal spacing added** | No benefit at this training budget (6–7 pipes in the doubly random environment). The real game's spacing is fixed, so later experiments drop it. |
| **Improving the agent** (random opening) | 11.5 → 16.4 (1M steps) → 22.7 (stack 4 frames) → 49.9 (linear learning-rate decay) → **92.1** (LR decay + 3M steps). The LR-decay effect replicated on new seeds (71.9 vs. 16.3). Curriculum learning, an entropy bonus, a larger network, and longer training with a constant LR did not help. |
| **Original game** (40,000-step evaluation cap) | The best agents average **700–770 pipes**; about half of the episodes reach the cap (~1,061 pipes). |
| **Random opening + 5% sticky actions** | All agents drop to about **20 pipes**. Training with the noise (20.1 vs. 18.6) and a modified reward (20.0) barely help. |
| **Exact upper bound** (dynamic programming) | With 5% sticky actions, an optimal policy that knows all future pipes could clear on average about **425 of the first 500 pipes** (57% chance of clearing all 500). The gap to PPO (~20) shows that the bottleneck is the agent, not the environment. |
| Round 5 (in progress) | Testing whether adding the previously executed action to the observation, or training for 10M steps, closes the gap. |

Detailed records for each experiment are in the `README_结果.md` / `README_E*_结果.md` file of its results folder. Each record states what was changed, what randomness was added, the results, and whether the change helped. The design of every round was written down in `PLAN.md` **before** training.

## Method highlights

- **Verified environment.** New environments subclass the original package without modifying it. With randomness turned off, the environment matches the original game step by step (22,437 steps: identical observations, rewards, and termination flags). Separate seeded random-number streams ensure that the fixed and random environments see the same sequence of opening heights.
- **Formulation.** The game is written as a parameterized family of MDPs. Sticky actions become an augmented-state MDP, and the LiDAR observation without velocity makes the problem a POMDP.
- **Exact bound.** Pipe motion does not depend on the bird, so for each pipe sequence the maximal probability of surviving *k* pipes is computed by backward induction over the integer state space (height, velocity, previous action). By an information-relaxation argument, this bounds every policy. The simulator was checked against the real environment with zero mismatches.
- **Evaluation discipline.** Each plan is written before training. One factor is changed at a time, with matched seeds and training budgets. The metric is full-body pipe clearance (stricter than the game's midpoint score). All seeds are shown, and no best seed is selected.

## Repository layout

```
project/
  env_v3.py                     environment factory: random gap, spacing, sticky actions, frame stack, reward variants
  random_gap_env.py             random vertical opening (separate RNG stream)
  random_spacing_env.py         random horizontal spacing
  gap_spacing_env.py            both
  original_reward_env.py        restores the original 0.4.0 reward on top of the frozen package
  train_v3.py                   PPO training (checkpoints, LR decay, curriculum, metadata + source hashes)
  eval_v3.py, eval_spacing_pilot.py   evaluation (full-body clearance, configurable step cap)
  summarize_v3.py               seed-level summaries
  theoretical_bound_sticky_v1.py      exact upper bound under sticky actions (dynamic programming)
  feasibility_*.py              exact reachability checks for the narrowest openings / largest jumps
  diagnose_policies_v1.py, summarize_diagnostics_v1.py, evaluate_historical_v1.py   diagnostics of earlier models
  test_*.py                     environment tests (equivalence with the original, RNG separation, sticky actions)
  vendor/flappy_bird_gymnasium  frozen copy of flappy-bird-gymnasium 0.4.0 (MIT license, see LICENSE file)
  results/<experiment>/         PLAN.md, results record, raw evaluation CSVs, summaries, training progress
historical_reference/original_package/   original 0.4.0 environment file and its diff to the frozen copy
```

Trained model files and checkpoints are not included in this repository, to keep it small.

## Reproducing

```bash
pip install -r requirements.txt
cd project
python test_env_v3_sticky.py
python train_v3.py --out results/demo --spec gap80_120+sticky5+stack4 --seed 0 --steps 3000000 --lr-linear
python eval_v3.py --run results/demo --label gap80_120+sticky5+stack4_seed0 --cap 40000 --conditions gap80_120+sticky5 gap100_100
python summarize_v3.py results/demo
python theoretical_bound_sticky_v1.py --out results/bound_demo --validate
```

Recorded versions: Python 3.11, Gymnasium 1.2.3, Stable-Baselines3 2.8.0, PyTorch 2.12.0. A clean installation on another machine has not been tested.

## References

Software this project builds on:
- flappy-bird-gymnasium (v0.4.0), G. Nogueira and M. Kubovcik, MIT License: https://github.com/markub3327/flappy-bird-gymnasium
- Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3 (Raffin et al., *JMLR* 22(268), 2021)
- Gymnasium: https://github.com/Farama-Foundation/Gymnasium (Towers et al., arXiv:2407.17032, 2024)

Literature:
- Schulman et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
- Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation. ICLR.
- Mnih et al. (2015). Human-level control through deep reinforcement learning. *Nature* 518.
- Machado et al. (2018). Revisiting the Arcade Learning Environment. *JAIR* 61.
- Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World. IROS.
- Cobbe et al. (2020). Leveraging Procedural Generation to Benchmark Reinforcement Learning. ICML.
- Kaelbling, Littman & Cassandra (1998). Planning and Acting in Partially Observable Stochastic Domains. *Artificial Intelligence* 101.
- Brown, Smith & Sun (2010). Information Relaxations and Duality in Stochastic Dynamic Programs. *Operations Research* 58(4).
- Henderson et al. (2018). Deep Reinforcement Learning That Matters. AAAI.
- Sutton & Barto (2018). *Reinforcement Learning: An Introduction*, 2nd ed.

## Status and limitations

This is ongoing research. Each condition uses 3–5 training seeds. All evaluations use one development panel of 20 episodes, and no untouched final test set has been used yet. The upper bound assumes knowledge of all future pipes and may be loose.
