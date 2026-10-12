"""A Gymnasium environment that serves CogGrid episodes one step at a time.

* **Observation** — a dict: ``observation``, an ``(n_observations,)`` int8 binary vector, a fresh sample each step (the terminal step repeats the last one);
  ``ctx_inds``, the ``(n_contexts,)`` indices of the active latent variables
  (an agent is expected to *learn* an embedding per index; training never makes a held-out index the goal and shows at most one per episode, unless ``n_contexts=1``);
  and ``goal_ind``, which column of ``ctx_inds`` the agent is scored on.
  ``ctx_inds`` and ``goal_ind`` are named as in :class:`~coggrid.EpisodeBatch`.
* **Info** — ``step``, ``split``, and the episode's ``goal_val``, ``ctx_vals`` and ``p_obs__vals`` (read ``__`` as "given"), named as in :class:`~coggrid.EpisodeBatch`;
  ``correct`` after ``step()``; ``joint_likelihood`` and ``marginal_likelihood`` only with ``expose_likelihood=True``.
* **Action** — ``Discrete(n_realizations)``: the agent's current guess at the goal variable's value.
  Actions do not influence the observations; the agent is a decoder, not a controller.
* **Reward** — 1.0 for a correct guess: on the last step only (``reward_mode="terminal"``, the default), or on every step (``"dense"``).

Every episode runs for ``n_steps`` steps and ends with ``terminated=True``; ``truncated``, which the Gymnasium API requires, is always ``False``.
:class:`CogGridEnv` draws ``cfg.batch_size`` episodes at a time from its :class:`~coggrid.World` and serves them one per ``reset()``.
If you want the batch directly, call :meth:`~coggrid.World.sample_episodes`.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary

from .config import CogGridConfig
from .sampling import Split
from .world import EpisodeBatch, World

__all__ = ["CogGridEnv", "register"]

RewardMode = Literal["dense", "terminal"]


class CogGridEnv(Env):
    """Single-episode Gymnasium environment.

    Parameters
    ----------
    cfg: A :class:`CogGridConfig`, or ``None`` for the defaults.
    world: An existing :class:`World` to draw from, so that several environments share embeddings. If given, ``cfg`` is ignored.
    split: Which variable pool episodes are drawn from, ``"held_out"`` or ``"train"``.
    reward_mode: ``"terminal"`` (reward on the last step only) or ``"dense"`` (every step).
    expose_likelihood: If ``True``, ``info`` carries the current episode's joint and marginal likelihood tables.
        Off by default, and then the tables are never built.
    seed: Seed for the episode stream, and for the embeddings when no ``world`` is given.
        ``None`` falls back to ``cfg.seed``.

    >>> env = CogGridEnv(seed=0)
    >>> obs, info = env.reset()
    >>> sorted(obs)
    ['ctx_inds', 'goal_ind', 'observation']
    >>> for _ in range(env.cfg.n_steps):
    ...     obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    >>> terminated
    True
    """

    def __init__(self, cfg: CogGridConfig | None = None, *, world: World | None = None, split: Split = "held_out", reward_mode: RewardMode = "terminal", expose_likelihood: bool = False, seed: int | None = None) -> None:
        if reward_mode not in ("dense", "terminal"):
            raise ValueError(f"reward_mode must be 'dense' or 'terminal', got {reward_mode!r}")
        if world is None:
            cfg = CogGridConfig() if cfg is None else cfg
            world = World(cfg if seed is None else cfg.replace(seed=seed))
        self.world = world
        self.cfg = cfg = world.cfg
        self.split: Split = split
        self.reward_mode: RewardMode = reward_mode
        self.expose_likelihood = expose_likelihood

        self.observation_space = Dict({
            "observation": MultiBinary(cfg.n_observations),
            "ctx_inds": Box(low=0, high=cfg.n_vars - 1, shape=(cfg.n_contexts,), dtype=np.int64),
            "goal_ind": Discrete(cfg.n_contexts),
        })
        self.action_space = Discrete(cfg.n_realizations)

        self._stream = np.random.default_rng(cfg.seed if seed is None else seed)
        self._buffer: EpisodeBatch | None = None
        self._cursor = 0
        self._episode: EpisodeBatch | None = None
        self._t = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start the next episode. Returns ``(observation, info)``. ``options`` is accepted for the Gymnasium API and ignored."""
        if seed is not None:
            seed_seq = np.random.SeedSequence(seed)  # validates the seed before anything changes
            self._stream = np.random.default_rng(seed_seq)
            # Gymnasium's np_random, from a spawned child: reproducible from the seed, yet not a replay of the episodes' draws.
            self.np_random = np.random.default_rng(seed_seq.spawn(1)[0])
            self._buffer = None
        if self._buffer is None or self._cursor >= len(self._buffer):
            # The same episodes either way: the tables are built only to be exposed.
            self._buffer = self.world.sample_episodes(split=self.split, rng=self._stream, expose_likelihood=self.expose_likelihood)
            self._cursor = 0
        self._episode = self._buffer.select(self._cursor)
        self._cursor += 1
        self._t = 0
        return self._observe(), self._info()

    def step(self, action: int | np.integer) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Submit a guess at the goal variable's value and advance one step.

        Returns Gymnasium's ``(observation, reward, terminated, truncated, info)``.
        ``truncated`` is always ``False``, since every episode runs its full ``n_steps``.
        """
        if self._episode is None:
            raise RuntimeError("call reset() before step()")
        action = int(action)
        if not 0 <= action < self.cfg.n_realizations:
            raise ValueError(f"action {action} outside Discrete({self.cfg.n_realizations})")
        correct = action == int(self._episode.goal_val[0])
        self._t += 1
        terminated = self._t >= self.cfg.n_steps
        reward = float(correct) if terminated or self.reward_mode == "dense" else 0.0
        info = self._info()
        info["correct"] = correct
        return self._observe(), reward, terminated, False, info

    def _observe(self) -> dict[str, Any]:
        assert self._episode is not None
        e = self._episode
        # After the last step there is no new sample to reveal: repeat the last one, so the dict always matches observation_space.
        t = min(self._t, self.cfg.n_steps - 1)
        return {"observation": e.observations[0, t].astype(np.int8), "ctx_inds": e.ctx_inds[0].astype(np.int64), "goal_ind": int(e.goal_ind[0])}

    def _info(self) -> dict[str, Any]:
        assert self._episode is not None
        e = self._episode
        info: dict[str, Any] = {"step": self._t, "split": self.split, "goal_val": int(e.goal_val[0]), "ctx_vals": e.ctx_vals[0].copy(), "p_obs__vals": e.p_obs__vals[0].copy()}
        if self.expose_likelihood:
            assert e.joint_likelihood is not None and e.marginal_likelihood is not None
            info["joint_likelihood"] = e.joint_likelihood[0].copy()
            info["marginal_likelihood"] = e.marginal_likelihood[0].copy()
        return info

    @property
    def episode(self) -> EpisodeBatch | None:
        """The current episode as a one-element :class:`EpisodeBatch`: handy for running an ideal observer on exactly the episode the agent saw."""
        return self._episode


def register() -> None:
    """Add ``CogGrid-v0`` to the Gymnasium registry.

    >>> import gymnasium as gym
    >>> from coggrid.env import register
    >>> register()
    >>> env = gym.make("CogGrid-v0")
    """
    from gymnasium.envs.registration import register as gym_register

    gym_register(id="CogGrid-v0", entry_point="coggrid.env:CogGridEnv", max_episode_steps=None)  # the env terminates on its own
