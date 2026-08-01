"""coggrid — a stationary POMDP for studying compositional generalisation
in latent space.

Despite the name, there is no spatial navigation. The "grid" is the table of
observation rates over the joint realisations of the active latent states; the
agent infers a latent value from accumulated evidence rather than moving
through a space.

Quick start
-----------
>>> from coggrid import CogGridConfig, World, run_observers
>>> world = World(CogGridConfig(n_states=200, n_contexts=2, seed=0))
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
``world``        :class:`World` and :class:`EpisodeBatch`.
``observers``    Ideal-observer baselines and metrics.
``env``          Gymnasium-compatible single and vector environments.
``components``   Protocols for the five swappable generative stages.
``viz``          Plotting. Every function returns a figure; none call ``show()``.
"""

from __future__ import annotations

from .components import (
    ContextSampler,
    EmbeddingSource,
    LikelihoodModel,
    ObservationModel,
    RealizationSampler,
)
from .config import CogGridConfig
from .env import CogGridEnv, CogGridVectorEnv
from .observers import (
    BeliefTrace,
    factorization_regret,
    joint_observer,
    naive_observer,
    run_observers,
    score_belief,
)
from .world import EpisodeBatch, World

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # configuration
    "CogGridConfig",
    # environment
    "World",
    "EpisodeBatch",
    "CogGridEnv",
    "CogGridVectorEnv",
    # observers
    "BeliefTrace",
    "joint_observer",
    "naive_observer",
    "run_observers",
    "score_belief",
    "factorization_regret",
    # extension points
    "EmbeddingSource",
    "ContextSampler",
    "LikelihoodModel",
    "RealizationSampler",
    "ObservationModel",
]
