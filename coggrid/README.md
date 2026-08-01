# coggrid

A stationary POMDP for studying **compositional generalisation in latent space** —
packaged as a standalone environment you can drop into your own training loop.

> **Despite the name, there is no spatial navigation.** The "grid" is the table
> of observation rates over the joint realisations of the active latent states.
> The agent infers a latent value from accumulated evidence rather than moving
> through a space; actions do not affect the observation stream.

Latent states interact *pairwise* to produce observation statistics, so the
observation distribution does not factorise over them. An agent that represents
the latent space as independent factors is therefore provably lossy, and the
size of that loss is the quantity this environment is built to measure.

```python
from coggrid import CogGridConfig, World, run_observers

world = World(CogGridConfig(n_states=500, n_contexts=2, seed=0))
batch = world.sample_episodes(2000)
traces = run_observers(batch)

traces["joint"].final()   # {'accuracy': 0.774, 'p_correct': 0.690, 'mse': 0.361}
traces["naive"].final()   # {'accuracy': 0.451, 'p_correct': 0.398, 'mse': 5.312}
```

That gap — 0.774 against 0.451 — is the cost of a factorised representation.

---

## Related repositories

| Repository | Contents |
| --- | --- |
| **this one** (`coggrid`) | The environment and ideal-observer baselines. No networks, no torch. |
| [`CognitiveGridworld`](https://github.com/johnschwarcz/CognitiveGridworld) | The reference implementation from [arXiv:2603.27134](https://arxiv.org/abs/2603.27134): environment *and* trained networks, together, as published. |
| *(planned)* `coggrid-models` | Customisable network architectures, depending on this package. |

Use **this** repo if you want the task. Use the **paper** repo to reproduce
published results.

---

## Install

```bash
pip install coggrid            # environment only — numpy
pip install 'coggrid[viz]'     # + matplotlib for the figures
pip install 'coggrid[gym]'     # + gymnasium registration
pip install 'coggrid[all]'
```

Python 3.10+. Pure numpy, no compiled extensions, no GPU, any OS.

Development install:

```bash
git clone https://github.com/johnschwarcz/coggrid
cd coggrid
pip install -e '.[dev]'
pytest
```

The convention used throughout the docs:

```python
import coggrid as cgw
```

---

## How the environment works

Each episode:

1. **Contexts.** `n_contexts` latent states are drawn from a pool of `n_states`.
   One of them is designated the **goal**.
2. **Realisations.** Each active state takes one of `n_realizations` discrete
   values. The goal state's value is the quantity to be inferred.
3. **Likelihood.** Every *pair* of active states contributes an interaction
   potential, built from their key/query embeddings. The potentials are summed
   and squashed, giving a Bernoulli rate for each of `n_observations` binary
   channels — one rate per hypothetical joint realisation.
4. **Observations.** The agent sees `n_steps` i.i.d. samples from those rates.

Stationary means the rate never changes within an episode. The difficulty is not
tracking a moving target; it is that the mapping from realisations to rates is
non-separable, so evidence about one state is only interpretable given the
others.

### The two baselines

| Observer | Uses | Meaning |
| --- | --- | --- |
| `joint` | `batch.rates` — the full interaction table | Bayes-optimal. The ceiling. |
| `naive` | `batch.marginal_rates` — interactions averaged away | Exactly the joint observer with the interactions removed, and nothing else changed. |

`factorization_regret(joint, naive)` is the symmetric KL between their
posteriors over the goal state.

### Train / held-out splits

```python
world.sample_episodes(1000, split="train")      # <=1 novel state, goal always familiar
world.sample_episodes(1000, split="held_out")   # every active state novel
```

`"train"` is the compositional condition: a novel state appears alongside
familiar ones, and the agent is scored on a familiar one. `"held_out"` is the
harder case where nothing in the episode has been seen.

---

## Gymnasium interface

```python
from coggrid import CogGridEnv

env = CogGridEnv(seed=0)
obs, info = env.reset()

for _ in range(env.cfg.n_steps):
    obs, reward, terminated, truncated, info = env.step(my_agent(obs))
```

**Observation** — a dict:

| Key | Shape | Meaning |
| --- | --- | --- |
| `observation` | `(n_observations,)` int8 | Fresh binary sample each step. |
| `active_states` | `(n_contexts,)` int64 | Which latent states are active. Constant within an episode. |
| `goal_context` | scalar int | Which column of `active_states` you are scored on. |

`active_states` holds *indices*, not embeddings — learning an embedding per index
from experience is the point of the task. At test time those indices have never
been seen.

**Action** — `Discrete(n_realizations)`: your current guess at the goal state's
value. Actions do not affect the observation stream; the agent is a decoder, not
a controller. Guessing every step is what makes the accuracy-versus-time curves
well defined.

**Reward** — 1.0 for a correct guess, every step (`reward_mode="dense"`) or only
on the last step (`reward_mode="terminal"`).

Episodes end with `terminated=True`, not `truncated=True`: the horizon *is* the
task ("decode from exactly `n_steps` samples"), not an externally imposed time
limit, so there is no value left to bootstrap.

For batched rollouts:

```python
from coggrid import CogGridVectorEnv
venv = CogGridVectorEnv(n_envs=256, seed=0)
```

This is the interface that matches how the environment is actually cheap to run —
the joint likelihood is one vectorised einsum over the batch.

With gymnasium installed, `coggrid.env.register()` adds `CogGrid-v0` to the
registry.

---

## Customising the generative model

Five stages are swappable. Pass a function to `World`; you never edit the
library.

```python
def codebook_embeddings(cfg, rng):
    """An EmbeddingSource: (cfg, rng) -> (keys, queries)."""
    ...

world = World(cfg, embeddings=codebook_embeddings)
```

| Argument | Protocol | Replace it to change |
| --- | --- | --- |
| `embeddings` | `EmbeddingSource` | How latent states relate to each other |
| `contexts` | `ContextSampler` | Which states are active; the split structure |
| `likelihood` | `LikelihoodModel` | The *form* of the interaction (e.g. add a 3-way term) |
| `realizations` | `RealizationSampler` | The prior over latent values |
| `observations` | `ObservationModel` | Non-stationary, correlated or continuous observations |

Signatures are in `coggrid/components.py`. Because each returns its outputs
rather than mutating shared state, a replacement can be unit-tested on its own.

---

## Visualisation

Every plotting function takes data, returns a `Figure`, and never calls `show()`.

```python
from coggrid.viz import plot_likelihood, plot_trial, plot_performance

plot_likelihood(batch, episode=0)     # is the rate surface separable?
plot_trial(batch, traces, episode=0)  # evidence + both posteriors over time
plot_performance(traces)              # mean +/- SD learning curves
```

`summary_figure(batch, traces)` returns all four at once.

```bash
python examples/04_figures.py --out figures/
```

---

## Memory

The joint likelihood is `n_episodes x n_observations x n_realizations ** n_contexts`.
That last term grows fast, so check before you run:

```python
>>> print(CogGridConfig(n_contexts=4, n_realizations=20).memory_report(1000))
1000 episodes x 5 channels x 20^4 realisations
  joint likelihood : 6.0 GiB
  ...
```

A `ResourceWarning` fires automatically above 2 GiB.

---

## Layout

```
src/coggrid/
├── config.py       CogGridConfig — every tunable, validated, immutable
├── generative.py   the generative model as pure functions
├── world.py        World (config + embeddings) and EpisodeBatch
├── observers.py    ideal-observer baselines and metrics
├── env.py          Gymnasium single and vector environments
├── components.py   protocols for the five swappable stages
├── spaces.py       gymnasium spaces, with shims when it isn't installed
└── viz/            plotting
examples/           four runnable scripts
tests/              62 tests, including parity with the reference implementation
```

---

## Relationship to the reference implementation

This is a refactor of the code in
[`CognitiveGridworld`](https://github.com/johnschwarcz/CognitiveGridworld), not a
rewrite. That implementation was driven with fixed inputs to dump reference
arrays; the same inputs through this package reproduce every likelihood, belief
and metric to floating-point precision (max |Δ| ≈ 1e-14). See
`tests/test_parity.py`.

Behaviour that deliberately changed:

- **Randomness.** Global `np.random` → explicit `numpy.random.Generator`, so
  results are reproducible from a seed. Random *draws* therefore differ from the
  original; everything downstream of them does not.
- **Inference.** Step-by-step multiply-and-renormalise → cumulative sum in log
  space. Mathematically identical, but does not underflow at long horizons
  (tested to `n_steps=4000`).
- **Numerics.** Overflow-safe sigmoid; rates clipped away from 0 and 1 before
  taking logs.

Migrating an existing script: see [`MIGRATION.md`](MIGRATION.md).
`CogGridConfig.from_legacy(**old_kwargs)` translates the original keyword names
directly.

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

MIT.
