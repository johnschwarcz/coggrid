"""Gymnasium-style environments.

The environment presents one episode at a time:

* **Observation** — a dict with three keys:

  ``observation``
      ``(n_observations,)`` int8 binary vector, one fresh sample per step.
  ``active_states``
      ``(n_contexts,)`` int indices of the latent states active this episode.
      Constant within an episode. An agent is expected to *learn* an embedding
      per index, which is where "generalisation from experience" comes in — at
      test time these indices have never been seen.
  ``goal_context``
      Which column of ``active_states`` the agent is scored on. Constant within
      an episode.

* **Action** — ``Discrete(n_realizations)``: the agent's current guess at the
  goal state's value. Actions do **not** influence the observation stream; the
  world is stationary and the agent is a decoder, not a controller. Reporting a
  guess every step is what makes the accuracy-versus-time curves in the paper
  well defined.

* **Reward** — 1.0 for a correct guess. ``reward_mode="dense"`` (default) pays
  out every step; ``"terminal"`` pays out only on the last step.

Episodes end with ``terminated=True`` rather than ``truncated=True``. The
horizon is not an externally imposed time limit — it *is* the task, namely "how
well can you decode from exactly ``n_steps`` samples of evidence" — so there is
no value left to bootstrap from.

Because the generative model is cheap in batch and expensive one episode at a
time, :class:`CogGridEnv` samples a buffer of episodes and serves
them one by one. Use :class:`CogGridVectorEnv` when you want the
batch directly.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from .config import CogGridConfig
from .generative import Split
from .spaces import Box, Dict, Discrete, EnvBase, MultiBinary
from .world import EpisodeBatch, World

__all__ = ["CogGridEnv", "CogGridVectorEnv", "register"]

RewardMode = Literal["dense", "terminal"]


def _observation_space(cfg: CogGridConfig) -> Dict:
    return Dict(
        {
            "observation": MultiBinary(cfg.n_observations),
            "active_states": Box(
                low=0,
                high=cfg.n_states - 1,
                shape=(cfg.n_contexts,),
                dtype=np.int64,
            ),
            "goal_context": Discrete(cfg.n_contexts),
        }
    )


class CogGridEnv(EnvBase):
    """Single-episode Gymnasium environment.

    Parameters
    ----------
    config:
        A :class:`CogGridConfig`, or ``None`` for the defaults.
    world:
        An existing :class:`World` to draw from. Pass one to share embeddings
        (and therefore the train/held-out split) across several environments.
        If given, ``config`` is ignored.
    split:
        Which state pool episodes are drawn from — ``"held_out"`` or ``"train"``.
    reward_mode:
        ``"dense"`` or ``"terminal"``.
    buffer_size:
        How many episodes to generate per internal refill. Larger is faster but
        uses more memory; see :meth:`CogGridConfig.memory_report`.
    expose_likelihood:
        If ``True``, ``info`` carries the full joint and marginal rate tables for
        the current episode. This is what an ideal-observer baseline needs. Off
        by default because the arrays are large and an agent should not see them.
    seed:
        Seed for the world's embeddings and the episode stream.

    Examples
    --------
    >>> env = CogGridEnv(seed=0)
    >>> obs, info = env.reset()
    >>> sorted(obs)
    ['active_states', 'goal_context', 'observation']
    >>> total = 0.0
    >>> for _ in range(env.cfg.n_steps):
    ...     obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    ...     total += reward
    >>> terminated
    True
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        config: CogGridConfig | None = None,
        *,
        world: World | None = None,
        split: Split = "held_out",
        reward_mode: RewardMode = "dense",
        buffer_size: int = 256,
        expose_likelihood: bool = False,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        if reward_mode not in ("dense", "terminal"):
            raise ValueError(
                f"reward_mode must be 'dense' or 'terminal', got {reward_mode!r}"
            )
        if buffer_size < 1:
            raise ValueError(f"buffer_size must be >= 1, got {buffer_size}")

        if world is None:
            cfg = config if config is not None else CogGridConfig()
            if seed is not None:
                cfg = cfg.replace(seed=seed)
            world = World(cfg)
        self.world = world
        self.cfg = world.cfg

        self.split: Split = split
        self.reward_mode: RewardMode = reward_mode
        self.buffer_size = int(buffer_size)
        self.expose_likelihood = bool(expose_likelihood)
        self.render_mode = render_mode

        self.observation_space = _observation_space(self.cfg)
        self.action_space = Discrete(self.cfg.n_realizations)

        self._stream = np.random.default_rng(
            self.cfg.seed if seed is None else seed
        )
        self._buffer: EpisodeBatch | None = None
        self._cursor = 0
        self._episode: EpisodeBatch | None = None
        self._t = 0
        self._last_action: int | None = None

    # ------------------------------------------------------------------ buffer
    def _next_episode(self) -> EpisodeBatch:
        if self._buffer is None or self._cursor >= len(self._buffer):
            self._buffer = self.world.sample_episodes(
                self.buffer_size, split=self.split, rng=self._stream
            )
            self._cursor = 0
        episode = self._buffer.select(self._cursor)
        self._cursor += 1
        return episode

    # ---------------------------------------------------------------- gym api
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start a new episode. Returns ``(observation, info)``."""
        if seed is not None:
            self._stream = np.random.default_rng(seed)
            self._buffer = None
        if options:
            split = options.get("split")
            if split is not None:
                self.split = split
                self._buffer = None

        self._episode = self._next_episode()
        self._t = 0
        self._last_action = None
        return self._observe(), self._info()

    def step(
        self, action: int | np.integer
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Submit a guess at the goal state's value and advance one step."""
        if self._episode is None:
            raise RuntimeError("call reset() before step()")

        action = int(action)
        if not 0 <= action < self.cfg.n_realizations:
            raise ValueError(
                f"action {action} outside Discrete({self.cfg.n_realizations})"
            )
        self._last_action = action

        correct = action == int(self._episode.goal_value[0])
        self._t += 1
        terminated = self._t >= self.cfg.n_steps

        if self.reward_mode == "dense":
            reward = float(correct)
        else:
            reward = float(correct) if terminated else 0.0

        # On the final step there is no further observation to reveal; repeat the
        # last one so the returned dict always matches observation_space.
        observation = self._observe(clamp=True)
        info = self._info()
        info["correct"] = correct
        return observation, reward, terminated, False, info

    def render(self) -> str | None:
        """One-line text summary of the current step."""
        if self.render_mode != "ansi" or self._episode is None:
            return None
        e = self._episode
        bits = "".join("1" if b else "0" for b in self._current_bits())
        guess = "-" if self._last_action is None else str(self._last_action)
        return (
            f"t={self._t:>3}/{self.cfg.n_steps}  obs={bits}  "
            f"states={e.ctx_inds[0].tolist()}  goal_ctx={int(e.goal_ind[0])}  "
            f"truth={int(e.goal_value[0])}  guess={guess}"
        )

    # ---------------------------------------------------------------- internals
    def _current_bits(self, clamp: bool = True) -> np.ndarray:
        assert self._episode is not None
        t = min(self._t, self.cfg.n_steps - 1) if clamp else self._t
        return self._episode.observations[0, t]

    def _observe(self, clamp: bool = True) -> dict[str, Any]:
        assert self._episode is not None
        return {
            "observation": self._current_bits(clamp).astype(np.int8),
            "active_states": self._episode.ctx_inds[0].astype(np.int64),
            "goal_context": int(self._episode.goal_ind[0]),
        }

    def _info(self) -> dict[str, Any]:
        assert self._episode is not None
        e = self._episode
        info: dict[str, Any] = {
            "step": self._t,
            "split": self.split,
            # Ground truth: for analysis and for scoring, never as agent input.
            "goal_value": int(e.goal_value[0]),
            "context_values": e.ctx_vals[0].copy(),
            "true_rates": e.true_rates[0].copy(),
        }
        if self.expose_likelihood:
            info["rates"] = e.rates[0]
            info["marginal_rates"] = e.marginal_rates[0]
        return info

    # ------------------------------------------------------------------ extras
    @property
    def episode(self) -> EpisodeBatch | None:
        """The current episode as a one-element :class:`EpisodeBatch`.

        Handy for running an ideal observer on exactly the episode the agent saw.
        """
        return self._episode


class CogGridVectorEnv(EnvBase):
    """Batched environment: ``n_envs`` synchronised episodes.

    Same semantics as :class:`CogGridEnv`, but every array carries a
    leading ``n_envs`` axis and all sub-episodes reset together. Since the
    horizon is fixed and identical across episodes, autoreset is unnecessary and
    the whole batch simply terminates at once.

    This is the interface that matches how the environment is actually cheap to
    run: the joint likelihood is one vectorised einsum over the batch.

    Examples
    --------
    >>> venv = CogGridVectorEnv(n_envs=64, seed=0)
    >>> obs, info = venv.reset()
    >>> obs["observation"].shape
    (64, 5)
    >>> actions = np.zeros(64, dtype=int)
    >>> obs, rewards, terminated, truncated, info = venv.step(actions)
    >>> rewards.shape
    (64,)
    """

    def __init__(
        self,
        n_envs: int = 64,
        config: CogGridConfig | None = None,
        *,
        world: World | None = None,
        split: Split = "held_out",
        reward_mode: RewardMode = "dense",
        seed: int | None = None,
    ) -> None:
        if n_envs < 1:
            raise ValueError(f"n_envs must be >= 1, got {n_envs}")
        if world is None:
            cfg = config if config is not None else CogGridConfig()
            if seed is not None:
                cfg = cfg.replace(seed=seed)
            world = World(cfg)

        self.world = world
        self.cfg = world.cfg
        self.n_envs = int(n_envs)
        self.split: Split = split
        self.reward_mode: RewardMode = reward_mode

        self.single_observation_space = _observation_space(self.cfg)
        self.single_action_space = Discrete(self.cfg.n_realizations)
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space

        self._stream = np.random.default_rng(
            self.cfg.seed if seed is None else seed
        )
        self._batch: EpisodeBatch | None = None
        self._t = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Draw ``n_envs`` fresh episodes."""
        if seed is not None:
            self._stream = np.random.default_rng(seed)
        if options and options.get("split") is not None:
            self.split = options["split"]

        self._batch = self.world.sample_episodes(
            self.n_envs, split=self.split, rng=self._stream
        )
        self._t = 0
        return self._observe(), self._info()

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Submit one guess per sub-episode."""
        if self._batch is None:
            raise RuntimeError("call reset() before step()")

        actions = np.asarray(actions).reshape(-1)
        if actions.size != self.n_envs:
            raise ValueError(
                f"expected {self.n_envs} actions, got {actions.size}"
            )
        if ((actions < 0) | (actions >= self.cfg.n_realizations)).any():
            raise ValueError(
                f"actions outside Discrete({self.cfg.n_realizations})"
            )

        correct = actions == self._batch.goal_value
        self._t += 1
        done = self._t >= self.cfg.n_steps

        if self.reward_mode == "dense":
            rewards = correct.astype(float)
        else:
            rewards = correct.astype(float) if done else np.zeros(self.n_envs)

        terminated = np.full(self.n_envs, done)
        truncated = np.zeros(self.n_envs, dtype=bool)
        info = self._info()
        info["correct"] = correct
        return self._observe(), rewards, terminated, truncated, info

    def _observe(self) -> dict[str, Any]:
        assert self._batch is not None
        t = min(self._t, self.cfg.n_steps - 1)
        return {
            "observation": self._batch.observations[:, t].astype(np.int8),
            "active_states": self._batch.ctx_inds.astype(np.int64),
            "goal_context": self._batch.goal_ind.astype(np.int64),
        }

    def _info(self) -> dict[str, Any]:
        assert self._batch is not None
        return {
            "step": self._t,
            "split": self.split,
            "goal_value": self._batch.goal_value.copy(),
            "context_values": self._batch.ctx_vals.copy(),
        }

    @property
    def batch(self) -> EpisodeBatch | None:
        """The current batch, for running ideal observers on the same episodes."""
        return self._batch


def register() -> None:
    """Register ``CogGrid-v0`` with Gymnasium, if it is installed.

    >>> import gymnasium as gym          # doctest: +SKIP
    >>> from coggrid.env import register  # doctest: +SKIP
    >>> register()                        # doctest: +SKIP
    >>> env = gym.make("CogGrid-v0")       # doctest: +SKIP
    """
    try:
        from gymnasium.envs.registration import register as gym_register
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "gymnasium is not installed; install with "
            "pip install 'coggrid[gym]'"
        ) from exc

    gym_register(
        id="CogGrid-v0",
        entry_point="coggrid.env:CogGridEnv",
        max_episode_steps=None,  # the env terminates on its own
    )
