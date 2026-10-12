<div align="center">
    <div style="margin-bottom: 20px">
        <h1>
            SRL: </br>
            Survival Reinforcement Learning </br>
            Toward Scalable Self-Supervised RL
        </h1>
        <h2>
            <a href="https://arxiv.org/abs/2605.31273">
                ArXiv
            </a>
        </h2>
    </div>
</div>

**Survival Reinforcement Learning (SRL)** is an online, classification-based alternative to TD and contrastive goal-conditioned reinforcement learning.
SRL casts goal-reaching as an event-time prediction problem: reaching a goal is a *survival event*, a truncated episode is *right-censored*, and a discrete-time hazard function is fit by maximum likelihood on both — no Bellman backups and no target networks. The goal-conditioned value is recovered in closed form from the hazard as a discounted sum of survival probabilities. 

SRL extends [Survival Value Learning (SVL)](https://arxiv.org/abs/2604.17551) by modeling what happens *after* the goal is reached.
Minimizing the first-hitting time alone rewards fly-by hits and can induce "bang-bang" control on complex dynamical systems; SRL instead conditions the critic on a goal sequence of length `agent.m` and only declares an event when the agent stays within the goal region for the whole sequence.
Maximizing dwell time this way makes the agent stabilize at the goal rather than merely touch it.
On the [JaxGCRL](https://github.com/MichalBortkiewicz/JaxGCRL) locomotion, navigation and manipulation benchmarks, scaled SRL matches state-of-the-art contrastive RL on manipulation and outperforms it by 2× to 8× on stable, long-horizon locomotion tasks.

# Installation

We manage the project with [uv](https://docs.astral.sh/uv/). From the repo root:

```shell
uv sync
```

This creates `.venv/` and installs all dependencies pinned in `uv.lock` (including the right JAX + CUDA wheels on Linux).

# Usage

Configuration is managed with [Hydra](https://hydra.cc/). The entry point is `srl/main.py`, defaults live in `srl/config/`, and all knobs can be overridden from the command line.

```shell
# Default run (ant-U4-maze)
uv run -m srl.main

# Single goal (m = 1)
uv run -m srl.main \
    agent.m=1 \
    env.env_id=ant_u4_maze \
    architecture.critic_network_depth=8 \
    architecture.actor_network_depth=8 \
    agent.discount=0.999 \
    seed=0

# Dwell-time critic (m = 32) on humanoid locomotion task
uv run -m srl.main \
    agent.m=32 \
    env.env_id=humanoid \
    architecture.critic_network_depth=16 \
    architecture.actor_network_depth=16 \
    agent.discount=0.999 \
    seed=0
```


## Multiple seeds / sweeps

Use [Hydra multi-run](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/) (`-m`) to dispatch several runs:

```shell
# Three seeds
uv run -m srl.main -m \
    agent.m=32 env.env_id=humanoid seed=0,1,2
```


# Citing SRL

```bibtex
@inproceedings{tiofacksurvival,
  title={Survival Reinforcement Learning: Toward Scalable Self-Supervised RL},
  author={Tiofack, Franki Nguimatsia and Schramm, Fabian and Le Hellard, Theotime and Carpentier, Justin},
  booktitle={Nineteenth European Workshop on Reinforcement Learning}
}
```
