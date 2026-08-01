"""Behavioural tests: the properties the environment is supposed to have.

``test_parity.py`` pins the numbers to the original implementation. This file
pins the *contracts* — the things that should stay true even if the generative
model is later changed on purpose.
"""

from __future__ import annotations

import numpy as np
import pytest

from coggrid import (
    CogGridEnv,
    CogGridVectorEnv,
    CogGridConfig,
    World,
    factorization_regret,
    run_observers,
    score_belief,
)

SMALL = CogGridConfig(
    n_states=80, n_held_out_states=8, n_contexts=2, n_realizations=6,
    n_observations=4, n_steps=15, embedding_dim=10, seed=0,
)


@pytest.fixture(scope="module")
def world() -> World:
    return World(SMALL)


@pytest.fixture(scope="module")
def batch(world: World):
    return world.sample_episodes(200)


# ------------------------------------------------------------------- config
class TestConfig:
    def test_defaults_are_valid(self):
        assert CogGridConfig().n_held_out_states == 50

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_contexts": 0},
            {"n_states": -1},
            {"embedding_dim": 2, "n_observations": 5},
            {"n_states": 5},                       # held-out split rounds to 0
            {"likelihood_temp": 0.0},
            {"n_states": 20, "n_held_out_states": 20},
            {"n_states": 20, "n_contexts": 30, "allow_repeated_states": False},
        ],
    )
    def test_rejects_bad_configs(self, kwargs):
        with pytest.raises(ValueError):
            CogGridConfig(**kwargs)

    def test_is_immutable(self):
        cfg = CogGridConfig()
        with pytest.raises(Exception):
            cfg.n_states = 10  # type: ignore[misc]

    def test_replace_revalidates(self):
        with pytest.raises(ValueError):
            CogGridConfig().replace(n_contexts=0)

    def test_from_legacy_translates_and_warns(self):
        with pytest.warns(UserWarning, match="removed network"):
            cfg = CogGridConfig.from_legacy(
                state_num=200, ctx_num=3, obs_num=4, batch_num=64,
                hid_dim=1000, mode=None, cuda=0,
            )
        assert (cfg.n_states, cfg.n_contexts, cfg.n_observations) == (200, 3, 4)
        assert cfg.n_episodes == 64

    def test_from_legacy_rejects_unknown(self):
        with pytest.raises(TypeError, match="Unrecognised"):
            CogGridConfig.from_legacy(not_a_real_option=1)

    def test_memory_report_mentions_the_blowup(self):
        report = CogGridConfig(n_contexts=4, n_realizations=20).memory_report(1000)
        assert "20^4" in report


# ------------------------------------------------------------- generative
class TestGenerativeModel:
    def test_shapes(self, batch):
        cfg = batch.cfg
        assert batch.rates.shape == (200, cfg.n_observations, *cfg.realization_shape)
        assert batch.marginal_rates.shape == (200, cfg.n_observations, cfg.n_contexts, cfg.n_realizations)
        assert batch.observations.shape == (200, cfg.n_steps, cfg.n_observations)
        assert batch.observations.dtype == bool

    def test_rates_are_probabilities(self, batch):
        assert (batch.rates > 0).all() and (batch.rates < 1).all()

    def test_embeddings_orthonormal_across_channels(self, world):
        gram = np.einsum("sod,sed->soe", world.keys, world.keys)
        eye = np.eye(world.cfg.n_observations)
        assert np.allclose(gram, eye, atol=1e-10)

    def test_marginal_is_a_marginal_of_the_joint(self, batch):
        direct = batch.rates.mean(axis=3)  # average out context 1
        assert np.allclose(batch.marginal_rates[:, :, 0, :], direct)

    def test_observation_rate_matches_realised_cell(self, batch):
        for b in range(10):
            cell = batch.rates[(b, slice(None), *batch.ctx_vals[b])]
            assert np.allclose(cell, batch.true_rates[b])

    def test_observation_frequency_matches_rate(self, world):
        """Empirical Bernoulli frequency should track the true rate."""
        batch = world.sample_episodes(400, rng=1)
        observed = batch.observations.mean(axis=1)
        assert np.abs(observed.mean(0) - batch.true_rates.mean(0)).max() < 0.05


# ------------------------------------------------------------------ splits
class TestSplits:
    def test_held_out_uses_only_novel_states(self, world):
        batch = world.sample_episodes(300, split="held_out", rng=2)
        assert (batch.ctx_inds < world.cfg.n_held_out_states).all()

    def test_train_allows_at_most_one_novel_state(self, world):
        batch = world.sample_episodes(300, split="train", rng=3)
        novel = batch.ctx_inds < world.cfg.n_held_out_states
        assert novel.sum(axis=1).max() <= 1

    def test_train_goal_is_never_novel(self, world):
        batch = world.sample_episodes(300, split="train", rng=4)
        goal_state = np.take_along_axis(
            batch.ctx_inds, batch.goal_ind[:, None], axis=1
        ).squeeze(1)
        assert (goal_state >= world.cfg.n_held_out_states).all()

    def test_distinct_states_when_requested(self):
        world = World(SMALL.replace(allow_repeated_states=False))
        batch = world.sample_episodes(200)
        assert all(len(set(row)) == len(row) for row in batch.ctx_inds)

    def test_rejects_unknown_split(self, world):
        with pytest.raises(ValueError, match="split"):
            world.sample_episodes(4, split="nonsense")  # type: ignore[arg-type]


# --------------------------------------------------------------- observers
class TestObservers:
    def test_beliefs_are_distributions(self, batch):
        for trace in run_observers(batch).values():
            assert np.allclose(trace.belief.sum(-1), 1.0)
            assert (trace.belief >= 0).all()

    def test_joint_dominates_naive(self, batch):
        traces = run_observers(batch)
        assert traces["joint"].final()["p_correct"] > traces["naive"].final()["p_correct"]
        assert traces["joint"].final()["mse"] < traces["naive"].final()["mse"]

    def test_evidence_accumulates(self, batch):
        curve = run_observers(batch)["joint"].mean_curves()["p_correct"]
        assert curve[-1] > curve[0]

    def test_identical_when_single_context(self):
        """With one active state there is no interaction to lose."""
        world = World(SMALL.replace(n_contexts=1))
        traces = run_observers(world.sample_episodes(100))
        assert np.allclose(traces["joint"].belief, traces["naive"].belief, atol=1e-12)

    def test_regret_is_non_negative(self, batch):
        traces = run_observers(batch)
        assert (factorization_regret(traces["joint"], traces["naive"]) >= 0).all()

    def test_score_belief_rejects_wrong_shape(self, batch):
        with pytest.raises(ValueError, match="belief must have shape"):
            score_belief(np.zeros((200, 15, 99, 99)), batch)

    def test_no_underflow_at_long_horizons(self):
        """The log-space rewrite must survive horizons the original could not."""
        world = World(SMALL.replace(n_steps=4000))
        traces = run_observers(world.sample_episodes(16))
        for trace in traces.values():
            assert np.isfinite(trace.belief).all()
            assert np.allclose(trace.belief.sum(-1), 1.0)


# ----------------------------------------------------------- reproducibility
class TestReproducibility:
    def test_same_seed_same_batch(self):
        a = World(SMALL).sample_episodes(20, rng=99)
        b = World(SMALL).sample_episodes(20, rng=99)
        assert np.array_equal(a.observations, b.observations)
        assert np.array_equal(a.ctx_inds, b.ctx_inds)

    def test_different_seed_different_batch(self):
        a = World(SMALL).sample_episodes(20, rng=1)
        b = World(SMALL).sample_episodes(20, rng=2)
        assert not np.array_equal(a.observations, b.observations)

    def test_roundtrip_through_disk(self, batch, tmp_path):
        from coggrid.world import EpisodeBatch

        restored = EpisodeBatch.load(batch.save(tmp_path / "batch"))
        assert restored.cfg == batch.cfg
        assert np.array_equal(restored.observations, batch.observations)
        assert np.allclose(restored.rates, batch.rates)

    def test_select_preserves_content(self, batch):
        one = batch.select(3)
        assert len(one) == 1
        assert np.array_equal(one.observations[0], batch.observations[3])


# ---------------------------------------------------------------- gym api
class TestGymEnv:
    def test_reset_step_contract(self):
        env = CogGridEnv(SMALL, seed=0)
        obs, info = env.reset()
        assert env.observation_space.contains(obs)
        steps = 0
        terminated = False
        while not terminated:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            steps += 1
            assert env.observation_space.contains(obs)
            assert not truncated
        assert steps == SMALL.n_steps

    def test_step_before_reset_raises(self):
        with pytest.raises(RuntimeError, match="reset"):
            CogGridEnv(SMALL).step(0)

    def test_rejects_out_of_range_action(self):
        env = CogGridEnv(SMALL, seed=0)
        env.reset()
        with pytest.raises(ValueError, match="Discrete"):
            env.step(SMALL.n_realizations)

    def test_perfect_agent_scores_one(self):
        env = CogGridEnv(SMALL, seed=0)
        _, info = env.reset()
        total = sum(env.step(info["goal_value"])[1] for _ in range(SMALL.n_steps))
        assert total == pytest.approx(SMALL.n_steps)

    def test_terminal_reward_mode(self):
        env = CogGridEnv(SMALL, reward_mode="terminal", seed=0)
        _, info = env.reset()
        rewards = [env.step(info["goal_value"])[1] for _ in range(SMALL.n_steps)]
        assert rewards[:-1] == [0.0] * (SMALL.n_steps - 1)
        assert rewards[-1] == 1.0

    def test_seeded_reset_is_reproducible(self):
        a = CogGridEnv(SMALL).reset(seed=5)[0]
        b = CogGridEnv(SMALL).reset(seed=5)[0]
        assert np.array_equal(a["observation"], b["observation"])
        assert np.array_equal(a["active_states"], b["active_states"])

    def test_vector_env_shapes(self):
        venv = CogGridVectorEnv(n_envs=16, config=SMALL, seed=0)
        obs, _ = venv.reset()
        assert obs["observation"].shape == (16, SMALL.n_observations)
        assert obs["active_states"].shape == (16, SMALL.n_contexts)
        _, rewards, terminated, truncated, _ = venv.step(np.zeros(16, dtype=int))
        assert rewards.shape == (16,)
        assert not terminated.any() and not truncated.any()

    def test_vector_env_terminates_together(self):
        venv = CogGridVectorEnv(n_envs=8, config=SMALL, seed=0)
        venv.reset()
        for _ in range(SMALL.n_steps):
            _, _, terminated, _, _ = venv.step(np.zeros(8, dtype=int))
        assert terminated.all()

    def test_env_episode_is_scoreable(self):
        """The episode an agent saw can be handed straight to an observer."""
        env = CogGridEnv(SMALL, seed=0)
        env.reset()
        traces = run_observers(env.episode)
        assert traces["joint"].belief.shape[0] == 1
