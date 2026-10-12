# coggrid

[![CI](https://github.com/johnschwarcz/coggrid/actions/workflows/ci.yml/badge.svg)](https://github.com/johnschwarcz/coggrid/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/coggrid)](https://pypi.org/project/coggrid/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/johnschwarcz/coggrid/blob/main/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2603.27134-b31b1b)](https://arxiv.org/abs/2603.27134)

A stationary POMDP for studying **compositional generalization in latent space** —
the environment and ideal-observer baselines from
[arXiv:2603.27134](https://arxiv.org/abs/2603.27134).

> **Sequential Bayesian inference is represented as navigating a latent space.**

An example episode:
<p align="center">
  <img src="https://raw.githubusercontent.com/johnschwarcz/coggrid/main/docs/images/episode_animation_1.gif" alt="An episode playing back: joint and naive posteriors, their difference, and the evidence stream" width="100%">
</p>

<p align="center"><em>
Top: the optimal and naive observers' posteriors and their difference. Bottom: A stream of observations.
</em></p>


---

## Contrasting Optimal and Naive Bayes

<p align="center">
  <img src="https://raw.githubusercontent.com/johnschwarcz/coggrid/main/docs/images/factorization_cost.png" alt="Factorization regret predicts the naive observer's failures, and its own confidence does not" width="100%">
</p>

The naive observer is not just noisier — Factorization regret decouples confidence and accuracy.

---

## How the environment works

```python
from coggrid import CogGridConfig, World
from coggrid.viz import animate_interaction_phases, plot_interaction_phases

world = World(CogGridConfig(n_vars=500, n_contexts=2, seed=0))
batch = world.sample_episodes(2000)

plot_interaction_phases(world, batch, episode=0, channels=(0,))
animate_interaction_phases(world, batch, episode=0)
```

Each episode:

1. **Contexts.** `n_contexts` latent variables are drawn from a pool of `n_vars`.
   One is designated the **goal**.
2. **Realizations.** Each "active" variable in the context takes one of `n_realizations` discrete
   values.
3. **Likelihood.** Active variables interact through key/query embeddings which map to an `n_realizations`^`n_contexts` likelihood.
4. **Observations.** `n_steps` i.i.d. observations sampled stochastically from the likelihood / prob. of observing 1 aka 1 - prob. of observing 0.

### What's in a batch

`sample_episodes` returns an `EpisodeBatch`:

| Field | Shape | Meaning |
| --- | --- | --- |
| `cfg` | — | The `CogGridConfig` the episodes were drawn from. |
| `split` | — | `"train"` or `"held_out"` — which variable pool the contexts came from. |
| `ctx_inds` | `(batch_size, n_contexts)` | Indices of the active variables, sorted. |
| `goal_ind` | `(batch_size,)` | Which column of `ctx_inds` is the goal. |
| `ctx_vals` | `(batch_size, n_contexts)` | The realization each active variable took. |
| `goal_val` | `(batch_size,)` | The goal variable's realization — the quantity to infer. |
| `observations` | `(batch_size, n_steps, n_observations)` | The boolean observation stream. |
| `joint_likelihood` | `(batch_size, n_observations, n_realizations, ...)` | `P(observation = 1)` given each joint realization — what the joint observer uses. `None` with `expose_likelihood=False`. |
| `marginal_likelihood` | `(batch_size, n_observations, n_contexts, n_realizations)` | The same for each variable's realizations after marginalizing over all others — what the naive observer uses. `None` with `expose_likelihood=False`. |
| `p_obs__vals` | `(batch_size, n_observations)` | `P(observation = 1)` given `ctx_vals` (read `__` as "given"): `joint_likelihood` at `ctx_vals`, what actually generated `observations`. |
| `interactions` | `(batch_size, n_observations, k)` | Interaction phases `z`, one per ordered pair of active variables: `k = n_contexts * (n_contexts - 1)`, or 1 with a single context. |

A learning agent should see only `ctx_inds`, `goal_ind` and `observations` (and optionally `interactions` if they are not to be learned).

`batch.select(index)` returns a sub-batch, `batch.save(path)` / `EpisodeBatch.load(path)` round-trip it through a compressed `.npz`.

### Interactions shift the joint likelihood in latent space

The inner product of one variable's key with the other's query sets a *phase* that translates an "XOR"-like pattern.

<p align="center">
  <img src="https://raw.githubusercontent.com/johnschwarcz/coggrid/main/docs/images/interaction_phases.png" alt="Variable embeddings, the inner products they produce, and the standard likelihood pattern those phases translate" width="100%">
</p>


Reading left to right:
1. **Embeddings.** Each latent variable carries a **key** and a **query** vector
   per channel. The angle between one variable's key and the other's query gives an interaction phase `z`. 
2. **Phase.**  Each phase shifts a standard sinusoid by `−2π · likelihood_freq · z`. 
3. **The standard pattern**, Sinusoids are expanded to a repeating pattern through an outer product.
4. **The selected pattern**  A specific pair of phases defines a specific region of the pattern.

### Impact of rotating the embeddings
Turning the keys sweeps the phases, and every panel to the right follows:

<p align="center">
  <img src="https://raw.githubusercontent.com/johnschwarcz/coggrid/main/docs/images/interaction_phases_animated.gif" alt="Turning a variable's key sweeps its interaction phase, sliding the selected region across the standard pattern" width="100%">
</p>

Note: This is meant to demonstrate how embeddings impact the likelihood. Embeddings are never actually rotated.

### What a single observation actually says

```python
from coggrid.viz import plot_evidence_likelihood

plot_evidence_likelihood(batch, episode=0)
```

The agent sees a **vector** of
`n_observations`  at once, and its likelihood is the product of the
per-channel likelihoods.

<p align="center">
  <img src="https://raw.githubusercontent.com/johnschwarcz/coggrid/main/docs/images/evidence_likelihood.png" alt="Per-channel likelihood tables, and the posterior induced by every possible observation vector" width="100%">
</p>


The top row is the per-channel joint likelihoods.
The grid below is every possible observation vector and the belief update it induces.


## Quickstart

```python
from coggrid import CogGridConfig, World, run_observers

world = World(CogGridConfig(n_vars=500, n_contexts=2, seed=0))
batch = world.sample_episodes(2000)
traces = run_observers(batch)

traces["joint"].final()   # {'accuracy': ..., 'p_correct': ..., 'mse': ...}
traces["naive"].final()   # the same three, for the factorized observer
```

### How the pieces fit together

| Piece | What it is |
| --- | --- |
| `CogGridConfig` | Every parameter, validated and immutable. |
| `World` | A config plus the variables' key/query embeddings, drawn once. `world.sample_episodes(n)` draws `n` complete episodes at once. |
| `EpisodeBatch` | What `sample_episodes` returns: the episodes as arrays — which variables were active, their values, every step's observations, and the likelihood tables. |
| `run_observers` | The ideal observers: take an `EpisodeBatch`, return each observer's beliefs and scores at every step. |
| `CogGridEnv` | The same episodes served one step at a time, for Gymnasium agents. It calls `world.sample_episodes()` internally and hands out one episode per `reset()`. |

To train a network, or to run the observers, work with batches from `world.sample_episodes`. `CogGridEnv` is only for agents written against the Gymnasium step-by-step interface.

## Install

```bash
pip install coggrid          # core
pip install "coggrid[viz]"   # + matplotlib for the figures
pip install "coggrid[torch]" # + optional CUDA acceleration of the joint observer
```

Python 3.10+. No compiled extensions, no GPU needed, any OS.

From source, for development:

```bash
git clone https://github.com/johnschwarcz/coggrid
cd coggrid
pip install -e ".[dev]"
pytest
```

### Every knob

`CogGridConfig` holds the whole specification of a world. 

| Field | Default | Meaning |
| --- | --- | --- |
| `n_vars` | 500 | Size of the latent-variable pool episodes draw from. |
| `n_contexts` | 2 | Variables active per episode. The joint table is `n_realizations ** n_contexts` wide, so this drives memory. |
| `n_realizations` | 10 | Discrete values each active variable can take. |
| `n_observations` | 5 | Binary observation channels. |
| `n_steps` | 30 | Observation samples per episode — the horizon. |
| `embedding_dim` | 30 | Length of each key/query vector. Must be ≥ `n_observations`, since they are orthogonalized across channels. |
| `likelihood_temp` | 2.0 | Amplitude of the sinusoids. Higher pushes the likelihood towards 0/1, making single observations more informative. |
| `likelihood_freq` | 1.0 | Number of periods in each sinusoid. Higher partitions the realization axis more finely. |
| `batch_size` | 1000 | Episodes per `sample_episodes` call by default, and per `CogGridEnv` refill. |
| `n_held_out_vars` | `None` | Size of the held-out pool. `None` means `n_vars // 10`. Held-out variables are `range(n_held_out_vars)`. |
| `allow_repeated_vars` | `False` | Whether one episode may activate the same variable twice. Two slots holding one variable can't be told apart, so a goal among them would be unidentifiable. When `False`, `n_contexts` may not exceed the smaller of the two pools. |
| `seed` | `None` | Seed for the default RNG. `None` means a fresh world each run. |

### The two baselines

| Observer | Uses | Meaning |
| --- | --- | --- |
| `joint` | `batch.joint_likelihood` — the likelihood of every joint realization, interactions intact | Bayes-optimal. The ceiling. |
| `naive` | `batch.marginal_likelihood` — interactions averaged away | Exactly the joint observer with the interactions removed, and nothing else changed. |

Both run in log space with a cumulative sum rather than a per-step
multiply-and-renormalize, so neither underflows at long horizons (tested to
`n_steps=4000`).

```python
from coggrid import CogGridConfig, World, run_observers
from coggrid import factorization_regret, disentanglement

world = World(CogGridConfig(n_vars=500, n_contexts=2, seed=0))
batch = world.sample_episodes(2000)
traces = run_observers(batch)

factorization_regret(traces["joint"], traces["naive"])   # (batch_size, n_steps)
disentanglement(traces["joint"], traces["naive"], batch) # (batch_size, n_steps)
```

**Factorization regret** — `D_KL(B_joint ‖ B_naive)` 
([§3.1](https://arxiv.org/abs/2603.27134)) — measures the cost of factorizing inference of interacting latent variables.

**Dis-entanglement** ([§B.3](https://arxiv.org/abs/2603.27134)) quantifies the 'history-dependence' of an observation, 
by measuring the divergence between the naive marginal likelihood `p(o_t | r)` 
and the optimal, history-conditioned one `p(o_t | r, o_1:t-1)`. 
It is zero when marginal belief dynamics are Markovian.


### Train / held-out splits

```python
world.sample_episodes(1000, split="train")      # <=1 novel variable, goal always familiar
world.sample_episodes(1000, split="held_out")   # every active variable novel
```

`"train"` is the condition under which a subset of the variables are never the
goal and never co-occur. `"held_out"` is the condition where no variable in the
episode has been a goal, nor co-occurred, during training. This is specifically relevant for training and evaluating networks.
With `n_contexts=1` there is no split: an episode's only variable is its goal, so
training covers every variable, held-out ones included.

---

## Gymnasium interface

```python
from coggrid import CogGridEnv

env = CogGridEnv(seed=0)
obs, info = env.reset()

for _ in range(env.cfg.n_steps):
    obs, reward, terminated, truncated, info = env.step(my_agent(obs))
```

**obs** — a dict; `ctx_inds` and `goal_ind` are named as in `EpisodeBatch`:

| Key | Shape | Meaning |
| --- | --- | --- |
| `observation` | `(n_observations,)` int8 | Fresh binary sample each step (the terminal step repeats the last one). |
| `ctx_inds` | `(n_contexts,)` int64 | Which latent variables are active. Constant within an episode. |
| `goal_ind` | scalar int | Which column of `ctx_inds` you are scored on. |

`ctx_inds` holds *indices*, not embeddings — learning an embedding per index
from experience is the point of the task. At test time none of them has been a
goal, nor co-occurred, during training (see [Train / held-out splits](#train--held-out-splits)).

**info** — `step`, `split`, and the episode's `goal_val`, `ctx_vals` and `p_obs__vals`, named as in `EpisodeBatch`;
`correct` after `step()`; `joint_likelihood` and `marginal_likelihood` only with `expose_likelihood=True`.

**Action** — `Discrete(n_realizations)`: the agent's current guess (MAP) of the goal
variable's value.

**Reward** — 1.0 for a correct guess, only on the last step (`reward_mode="terminal"`, the default) or on every step (`reward_mode="dense"`).

Every episode runs `n_steps` steps and ends with `terminated=True`. `truncated` is part of the Gymnasium API and is always `False`.

`coggrid.env.register()` adds `CogGrid-v0` to the Gymnasium registry.

`CogGridEnv` takes:

| Argument | Default | Meaning |
| --- | --- | --- |
| `cfg` | `None` | The world specification, per the table above. `None` builds a default `CogGridConfig()`. |
| `world` | `None` | Draw from an existing `World` instead of building one, so several environments share embeddings and therefore the same split. Overrides `cfg`. |
| `split` | `"held_out"` | Which variable pool episodes come from — see [Train / held-out splits](#train--held-out-splits). |
| `reward_mode` | `"terminal"` | `"terminal"` scores only the last step, `"dense"` every step. |
| `expose_likelihood` | `False` | Put the joint and marginal likelihood tables in `info` — what an ideal-observer baseline needs, and what a learning agent must not see. When `False`, the tables are never built (see [Scaling up](#scaling-up)). |
| `seed` | `None` | Seeds the episode stream, and the embeddings when the env builds its own world. `None` falls back to `cfg.seed`. |

### Batched rollouts

There is no vector environment. Batching is what `World` already does:

```python
rollout = world.sample_episodes(4096)      # every episode at once
rollout.observations                       # (4096, n_steps, n_observations)
```

`sample_episodes` builds the whole batch at once, vectorized, which is the cheap path. A gym wrapper around it would only add
a step-by-step interface on top of data that is already complete.

---

## Visualization

The `plot_*` functions return a `Figure` and the `animate_*` functions an animation; nothing calls `show()`.

```python
from coggrid.viz import summary_figure

figures = summary_figure(batch, traces)   # -> 4 figures, narrowing in scope
```

- **`plot_episode`** — one episode's likelihood surface beside its belief traces.
- **`plot_performance`** — batch-averaged curves, one column per metric
  (accuracy, `p_correct`, MSE, factorization regret), with individual episode
  trajectories behind each mean.
- **`plot_regret_analysis`** — whether regret explains the gap.
- **`plot_belief_shape`** — the posteriors behind the accuracy numbers: the mean
  final belief with every episode aligned on its true value, and the
  distribution of `P(true value)` across episodes at each timestep.

The last two are omitted when `n_contexts == 1`, where the two observers coincide
and there is nothing to factorize away.

Every panel is also available on its own — `plot_likelihood`, `plot_trial`,
`plot_regret`, `plot_relative_accuracy`, `plot_regret_vs_accuracy`,
`plot_map_agreement`, `plot_belief_profile`, `plot_confidence_density` — and each
takes `ax` (or `axes`, or a `subplot_spec`) so you can compose your own layout.

### Watching an episode

`animate_episode` is the `render()` of this package. In a notebook, make it the
last expression in a cell and it plays — no ffmpeg, no `to_jshtml`, no backend to
configure:

```python
from coggrid.viz import animate_episode

animate_episode(batch, traces)                            # plays inline
animate_episode(batch, traces).save("docs/images/episode")   # or write a GIF
animate_episode(batch, traces).to_html()                  # or a scrub player
```

Panels are laid out in rows — realization grids on top, anything that evolves
over time underneath:

| Row | Panels |
| --- | --- |
| `GRID_PANELS` | `joint`, `naive (factorized)`, and `joint − naive` — where factorizing moves probability mass. Each marks the truth in green, `joint` and `naive` ring their current mode in white, and the axes name which variable is the **goal** (green) and which is **context** (orange). |
| `TRACE_PANELS` | the evidence stream, revealed up to the current step |

`animate_episode` takes `episode` (which episode to play, default 0), `fps`
(playback speed, default 6), and two arguments that decide what is drawn:

- **`extended=True`** adds the goal variable's marginal belief and two panels
  relating factorization regret to dis-entanglement, one accumulating and one per step.
- **`panels`** — a list of rows, each row a list of panel functions, replacing the
  default layout entirely. This is the extension point below.

**Adding your own panel.** A panel draws its furniture into an axis and returns
an updater called with the timestep, so extending the animation never means
editing the library:

```python
from coggrid.viz import GRID_PANELS, TRACE_PANELS, animate_episode

def p_correct_panel(ax, view):
    (line,) = ax.plot([], [])
    p_correct = view.traces["joint"].p_correct[view.episode]
    ax.set(xlim=(0, view.n_steps - 1), ylim=(0, 1), title="P(true value)")

    def update(t):
        line.set_data(range(t + 1), p_correct[: t + 1])

    return update

animate_episode(batch, traces, panels=[GRID_PANELS, [*TRACE_PANELS, p_correct_panel]])
```

`view` is an `EpisodeView` — the batch, traces, episode index and palette, plus
the derived arrays panels keep needing (`joint_grid`, `naive_grid`,
`observations`, `ctx_vals`, `goal_regret`, `disentanglement`).

```bash
python examples/04_figures.py --out docs/images/     # the four static figures
python examples/05_animation.py --out docs/images/   # an episode played back, as a GIF
```

---

## Customizing the generative model

The embeddings, which set how latent variables relate to each other, are swappable: pass your own function to `World`; you never edit the library.

```python
def codebook_embeddings(cfg, rng):
    """An EmbeddingSource: (cfg, rng) -> (keys, queries)."""
    ...

world = World(cfg, embeddings=codebook_embeddings)
```

`EmbeddingSource`, the signature, is at the top of [`src/coggrid/world.py`](https://github.com/johnschwarcz/coggrid/blob/main/src/coggrid/world.py); [`examples/03_customize.py`](https://github.com/johnschwarcz/coggrid/blob/main/examples/03_customize.py) is a complete example.

---

## Memory

The joint likelihood is `batch_size x n_observations x n_realizations ** n_contexts`.
That last term grows fast, so check before you run:

```python
>>> print(CogGridConfig(n_contexts=4, n_realizations=20).memory_report(1000))
1000 episodes x 5 channels x 20^4 realizations
  joint likelihood    : 6.0 GiB
  joint log posterior : 35.8 GiB
  peak (approx)       : 41.7 GiB
```

A `ResourceWarning` fires when `sample_episodes` builds a joint likelihood above 2 GiB.
The observers' log posterior is not checked; `keep_log_posterior=False` holds only a chunk of it at a time.

---

## Scaling up

An episode needs only `p_obs__vals`, the probabilities at the realization that
occurred; the `n_realizations ** n_contexts` tables are for the ideal observers.
`expose_likelihood=False` skips them and gives bit-for-bit the same episodes at `O(n_contexts²)` each:
on one thread, 1000 take milliseconds at 2 contexts and under a second at 16. The
observers and figures rebuild the tables they need, a few episodes at a time.

```python
from coggrid import CogGridConfig, World, run_observers

many = World(CogGridConfig(n_vars=500, n_contexts=16, seed=0))
episodes = many.sample_episodes(1000, expose_likelihood=False)   # 10^16-wide tables: skipped

world = World(CogGridConfig(n_vars=500, n_contexts=4, seed=0))
traces = run_observers(world.sample_episodes(1000, expose_likelihood=False))
```

With many contexts the likelihood saturates. At the default
`likelihood_temp=2.0`, about 0.6% of `p_obs__vals` are exactly 0 or 1 at 8 contexts,
20% at 12, 55% at 16 and 99% at 32. Lowering `likelihood_temp` removes the
saturation but not the drift: the mean logit grows with the number of pairs, so
at 16 contexts even `likelihood_temp=0.6` leaves 89% of observations reading 1.

With `pip install "coggrid[torch]"` the joint observer can run on a CUDA GPU
(not Apple's MPS, which has no float64), only when asked; coggrid itself never
needs torch.

```python
from coggrid.torch import joint_belief

episodes = world.sample_episodes(1000, expose_likelihood=False)
traces = run_observers(episodes, device="cuda", keep_log_posterior=False)
belief, goal_belief = joint_belief(episodes, device="cuda")   # float32 tensors
```

At 4 contexts, the joint observer takes 2.6 s for 1000 episodes of 30 steps in numpy on
one thread, and 94 ms in float64 or 55 ms in float32 on an RTX 3090 Ti. Float64 agrees with
numpy to about `1e-12`, float32 to about `1e-4`.

---

## Layout

```
src/coggrid/
├── config.py       CogGridConfig — every tunable, validated, immutable
├── sampling.py     the random draws: contexts, realizations, observations
├── likelihood.py   how realizations set P(obs = 1): embeddings, sinusoids, tables
├── world.py        World (config + embeddings) and EpisodeBatch
├── observers.py    ideal-observer baselines, factorization regret,
│                   dis-entanglement
├── env.py          CogGridEnv, the Gymnasium environment
├── torch.py        optional CUDA acceleration of the joint observer
└── viz/            plotting and animation
examples/           five runnable scripts
docs/make_assets.py regenerates the images in this README
tests/              behavioral tests, plus the docstring examples
```

Numerical notes worth knowing:

- **Randomness** is an explicit `numpy.random.Generator` throughout — no global
  `np.random` — so any batch is reproducible from a seed.
- **Inference** accumulates log-likelihoods with a cumulative sum rather than
  multiplying and renormalizing per step, so it does not underflow at long
  horizons.
- **Per-step belief updates** are recovered by differencing *log* beliefs.
  Beliefs routinely fall below `1e-12` once an observer is confident, and
  dividing two such numbers loses most of the significant digits.
- **Likelihoods** are clipped away from 0 and 1 before taking logs, and the sigmoid is
  overflow-safe.

---

## Related repositories

| Repository | Contents |
| --- | --- |
| **this one** (`coggrid`) | The environment and ideal-observer baselines. No networks; torch is optional. |
| [`CognitiveGridworld`](https://github.com/johnschwarcz/CognitiveGridworld) | The reference implementation from the paper: environment *and* trained networks, together, as published. |

Use **this** repo if you want the task. Use the paper repo to reproduce published
results.

---

## Citation

```bibtex
@article{schwarcz2026factorization,
  title  = {Factorization Regret mediates compositional generalization in latent space},
  author = {Schwarcz, John},
  year   = {2026},
  eprint = {2603.27134},
  archivePrefix = {arXiv}
}
```

## License

MIT — see [LICENSE](https://github.com/johnschwarcz/coggrid/blob/main/LICENSE).
