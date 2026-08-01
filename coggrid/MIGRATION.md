# Migrating from `CognitiveGridworld` to `coggrid`

The original `CognitiveGridworld` object did everything: it preprocessed the
environment, generated embeddings, sampled episodes, ran the Bayesian observers
and drew plots — all from `__init__` — leaving results scattered across ~40
attributes on `self`. This package splits those responsibilities apart.

## The shortest possible version

```python
# before
from main import CognitiveGridworld
self = CognitiveGridworld(episodes=1000, state_num=500, batch_num=8000,
                          step_num=30, obs_num=5, ctx_num=2, mode=None,
                          show_plots=True)
accuracy = self.joint_acc.mean(0)

# after
from coggrid import CogGridConfig, World, run_observers
from coggrid.viz import summary_figure

cfg = CogGridConfig(n_states=500, n_contexts=2, n_observations=5,
                      n_steps=30, seed=0)
batch = World(cfg).sample_episodes(8000)
traces = run_observers(batch)
accuracy = traces["joint"].mean_curves()["accuracy"]

for fig in summary_figure(batch, traces):
    fig.savefig(...)
```

Keeping your old keyword dict:

```python
cfg = CogGridConfig.from_legacy(state_num=500, ctx_num=2, obs_num=5,
                                  batch_num=8000, hid_dim=1000, cuda=0)
# network-only keywords are dropped with a warning; renamed ones are translated
```

## Constructor keywords

| Original | Now | Note |
| --- | --- | --- |
| `state_num` | `CogGridConfig.n_states` | |
| `ctx_num` | `n_contexts` | |
| `realization_num` | `n_realizations` | |
| `obs_num` | `n_observations` | |
| `step_num` | `n_steps` | |
| `batch_num` | `n_episodes` | also the argument to `sample_episodes()` |
| `KQ_dim` | `embedding_dim` | now validated as `>= n_observations` |
| `test_states` | `n_held_out_states` | |
| `subsample_states` | `subsample_states` | |
| `likelihood_temp` | `likelihood_temp` | |
| `likelihood_freq` | `likelihood_freq` | |
| `test_set` (implicit) | `split="held_out"` / `"train"` | an explicit argument, not a mutable flag |
| — | `seed` | **new**; the original had no seeding at all |
| — | `allow_repeated_states` | **new**; the original always allowed repeats |
| `show_plots`, `plot_every`, `checkpoint_every`, `showtime` | — | plotting is no longer triggered from the simulator |
| `episodes`, `hid_dim`, `mode`, `training`, `reservoir`, `learn_embeddings`, `*_LR`, `*_ent_bonus`, `cuda`, `early_stopping`, `skip_training_analyses`, `load_env`, `save_env` | — | network/training only; not in this package |

## Result attributes

Everything that used to live on `self` is now on an `EpisodeBatch` or a
`BeliefTrace`.

| Original `self.…` | Now |
| --- | --- |
| `ctx_inds` | `batch.ctx_inds` |
| `ctx_vals` | `batch.ctx_vals` |
| `goal_ind` | `batch.goal_ind` |
| `goal_value` | `batch.goal_value` |
| `joint_likelihood` | `batch.rates` |
| `naive_likelihood` | `batch.marginal_rates` |
| `joint_Z` | `batch.interactions` |
| `pobs__joint` | `batch.true_rates` |
| `obs_flat` | `batch.observations` |
| `joint_belief` | `traces["joint"].belief` |
| `naive_belief` | `traces["naive"].belief` |
| `joint_goal_belief` | `traces["joint"].goal_belief` |
| `joint_acc` / `naive_acc` | `traces[…].accuracy` |
| `joint_TP` / `naive_TP` | `traces[…].p_correct` |
| `joint_mse` / `naive_mse` | `traces[…].mse` |
| `joint_est` / `naive_est` | `traces[…].estimate` |
| `SII` | `factorization_regret(traces["joint"], traces["naive"])` |
| `agent_accs`, `agent_beliefs`, … | iterate `traces.values()` |

Renames worth knowing: `TP` → `p_correct` (it was the posterior probability of
the true value, not a true-positive rate), and `SII` → `factorization_regret`.

## Methods

| Original | Now |
| --- | --- |
| `preprocess_env()` | gone — derived quantities are `CogGridConfig` properties |
| `EC_gen_state_embeddings()` | `World(cfg)` (once, at construction) |
| `EC_gen_context()` | `generative.sample_contexts()` |
| `EC_gen_likelihoods()` | `generative.joint_likelihood()` + `marginal_likelihood()` |
| `EC_gen_realizations()` | `generative.sample_realizations()` |
| `EC_gen_observations()` | `generative.sample_observations()` |
| `run_generators()` | `world.sample_episodes()` |
| `run_inference()` | `run_observers(batch)` |
| `plot_likelihood()` | `viz.plot_likelihood(batch, episode=…)` |
| `plot_trial()` | `viz.plot_trial(batch, traces, episode=…)` |
| `plot_bayes_perf()` | `viz.plot_performance(traces)` |
| `main_plotters()` | `viz.summary_figure(batch, traces)` |

## Customisation

The `custom_*` methods in `main/env/Env_Customization.py` — which returned `False` by
default and required editing a file inside the library — are replaced by five
protocols in `coggrid/components.py`. Pass a function to `World`.

```python
# before: edit main/env/Env_Customization.py in place
def custom_state_embeddings(self, using_custom=True):
    self.all_K = ...      # assign to self, correct shape not stated anywhere
    self.all_Q = ...
    return using_custom

# after: your own file, nothing in the library changes
def my_embeddings(cfg, rng):
    """-> (keys, queries), each (n_states, n_observations, embedding_dim)"""
    return keys, queries

world = World(cfg, embeddings=my_embeddings)
```

Mapping: `custom_state_embeddings` → `embeddings`, `custom_context` →
`contexts`, `custom_likelihoods` → `likelihood`, `custom_realizations` →
`realizations`, `custom_gen_observations` → `observations`.

## Things that will bite you

- **Results will not match run-for-run.** The original used global `np.random`
  with no seed. Same-seed reproducibility is new; identical output to an old
  unseeded run is impossible. The *deterministic* pipeline is unchanged — see
  `tests/test_parity.py`.
- **`n_episodes` is an argument, not a mutable attribute.** Sample a new batch
  instead of resizing the world.
- **`EpisodeBatch` is frozen.** Use `.select(...)` for subsets and
  `dataclasses.replace` for variants.
- **Nothing plots itself.** `show_plots=True` has no equivalent; call a `viz`
  function and do what you like with the returned `Figure`.
- **`batch.rates` is the big one.** Check `cfg.memory_report()` before raising
  `n_contexts` or `n_realizations`.
- **The old env-only path (`mode=None`) still required PyTorch**, because
  `Env_control_manager` sat in the inheritance chain and imported the model.
  That is fixed: this package's only dependency is numpy.
