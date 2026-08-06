"""coggrid — a stationary POMDP for studying compositional generalization
in latent space.

The "grid" is the table of observation rates over the joint realizations of the
active latent variables.

Quick start
-----------
>>> from coggrid import CogGridConfig, World, run_observers
>>> world = World(CogGridConfig(n_vars=200, n_contexts=2, seed=0))
>>> batch = world.sample_episodes(256)
>>> traces = run_observers(batch)
>>> traces["joint"].final()["accuracy"] > traces["naive"].final()["accuracy"]
True

Gymnasium interface
-------------------
>>> from coggrid import CogGridEnv
>>> env = CogGridEnv(seed=0)
>>> obs, info = env.reset()
>>> obs, reward, terminated, truncated, info = env.step(0)

Layout
------
``config``       :class:`CogGridConfig` — every tunable, validated, immutable.
``generative``   The generative model as pure functions.
``world``        :class:`World`, :class:`EpisodeBatch`, and the five swappable
                 generative-stage signatures.
``observers``    Ideal-observer baselines and metrics.
``env``          Gymnasium single and vector environments.
``viz``          Plotting. Every function returns a figure; none call ``show()``.
"""

from __future__ import annotations

from .config import CogGridConfig
from .env import CogGridEnv
from .observers import (
    BeliefTrace,
    disentanglement,
    factorization_regret,
    joint_observer,
    naive_observer,
    run_observers,
    score_belief,
)
from .world import (
    ContextSampler,
    EmbeddingSource,
    EpisodeBatch,
    LikelihoodModel,
    ObservationModel,
    RealizationSampler,
    World,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # configuration
    "CogGridConfig",
    # environment
    "World",
    "EpisodeBatch",
    "CogGridEnv",
    # observers
    "BeliefTrace",
    "joint_observer",
    "naive_observer",
    "run_observers",
    "score_belief",
    "factorization_regret",
    "disentanglement",
    # extension points
    "EmbeddingSource",
    "ContextSampler",
    "LikelihoodModel",
    "RealizationSampler",
    "ObservationModel",
]
