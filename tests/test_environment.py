"""Behavioral tests: the properties the environment is supposed to have.

These pin the *contracts* — the things that should stay true even if the
generative model is later changed on purpose.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from coggrid import (
    CogGridConfig,
    CogGridEnv,
    World,
    disentanglement,
    factorization_regret,
    run_observers,
    score_belief,
)

SMALL = CogGridConfig(
    n_vars=80, n_held_out_vars=8, n_contexts=2, n_realizations=6,
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
        assert CogGridConfig().n_held_out_vars == 50

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_contexts": 0},
            {"n_vars": -1},
            {"embedding_dim": 2, "n_observations": 5},
            {"n_vars": 5},                       # held-out split rounds to 0
            {"likelihood_temp": 0.0},
            {"n_vars": 20, "n_held_out_vars": 20},
            {"n_vars": 20, "n_contexts": 30, "allow_repeated_vars": False},
        ],
    )
    def test_rejects_bad_configs(self, kwargs):
        with pytest.raises(ValueError):
            CogGridConfig(**kwargs)

    def test_is_immutable(self):
        cfg = CogGridConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.n_vars = 10  # type: ignore[misc]

    def test_replace_revalidates(self):
        with pytest.raises(ValueError):
            CogGridConfig().replace(n_contexts=0)

    def test_held_out_default_resolves_to_an_int(self):
        """``None`` is resolved during validation, so readers never see it."""
        assert CogGridConfig(n_vars=200).n_held_out_vars == 20
        assert CogGridConfig(n_vars=200).n_train_vars == 180

    def test_memory_report_mentions_the_blowup(self):
        report = CogGridConfig(n_contexts=4, n_realizations=20).memory_report(1000)
        assert "20^4" in report


# ------------------------------------------------------------- generative
class TestGenerativeModel:
    def test_shapes(self, batch):
        cfg = batch.cfg
        assert batch.rates.shape == (200, cfg.n_observations, *cfg.realization_shape)
        assert batch.marginal_rates.shape == (
            200, cfg.n_observations, cfg.n_contexts, cfg.n_realizations
        )
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

    def test_observation_rate_matches_realized_cell(self, batch):
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
    def test_held_out_uses_only_novel_vars(self, world):
        batch = world.sample_episodes(300, split="held_out", rng=2)
        assert (batch.ctx_inds < world.cfg.n_held_out_vars).all()

    def test_train_allows_at_most_one_novel_var(self, world):
        batch = world.sample_episodes(300, split="train", rng=3)
        novel = batch.ctx_inds < world.cfg.n_held_out_vars
        assert novel.sum(axis=1).max() <= 1

    def test_train_goal_is_never_novel(self, world):
        batch = world.sample_episodes(300, split="train", rng=4)
        goal_var = np.take_along_axis(
            batch.ctx_inds, batch.goal_ind[:, None], axis=1
        ).squeeze(1)
        assert (goal_var >= world.cfg.n_held_out_vars).all()

    def test_distinct_vars_when_requested(self):
        world = World(SMALL.replace(allow_repeated_vars=False))
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
        """With one active variable there is no interaction to lose."""
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
        """Log-space accumulation must stay finite at horizons that would underflow."""
        world = World(SMALL.replace(n_steps=4000))
        traces = run_observers(world.sample_episodes(16))
        for trace in traces.values():
            assert np.isfinite(trace.belief).all()
            assert np.allclose(trace.belief.sum(-1), 1.0)


class TestPhaseStructure:
    """Interaction strength enters the likelihood only as a phase.

    The README and ``plot_interaction_phases`` both claim every rate table is
    one standard pattern, translated. These pin that down.
    """

    @staticmethod
    def _waveform(cfg, n_samples=4001):
        """The standard potential waveform, read off as the phase sweeps a turn."""
        from coggrid.generative import realization_potential, value_profile

        turn = np.linspace(0.0, 1.0, n_samples)[:-1]
        return turn, realization_potential(turn, value_profile(cfg), cfg.n_roll)[:, 0]

    def test_phase_is_linear_in_interaction_strength(self):
        """Phase advances by ``-2*pi*likelihood_freq*z``, with fixed amplitude."""
        from coggrid.generative import realization_potential, value_profile

        for freq in (1.0, 2.0):
            cfg = CogGridConfig(n_vars=60, n_realizations=10,
                                likelihood_freq=freq, seed=0)
            profile = value_profile(cfg)
            r = np.arange(cfg.n_realizations)
            basis = np.exp(-2j * np.pi * freq * r / cfg.n_roll)

            zs = np.linspace(-0.45, 0.45, 31)
            comp = np.array([
                basis @ realization_potential(np.array(z), profile, cfg.n_roll)
                for z in zs
            ])
            slope, intercept = np.polyfit(zs, np.unwrap(np.angle(comp)), 1)
            assert slope == pytest.approx(-2 * np.pi * freq, rel=0.02)
            # A pure phase shift leaves the amplitude alone.
            assert np.ptp(np.abs(comp)) / np.abs(comp).mean() < 0.15
            residual = np.abs(np.unwrap(np.angle(comp)) - (slope * zs + intercept))
            assert residual.max() < 0.1

    def test_every_rate_table_is_the_same_pattern_translated(self):
        """Rebuild each table from one waveform plus its two strengths."""
        from coggrid.generative import sigmoid

        cfg = CogGridConfig(n_vars=200, n_contexts=2, n_realizations=10, seed=4)
        batch = World(cfg).sample_episodes(6)
        turn, waveform = self._waveform(cfg)
        r = np.arange(cfg.n_realizations)

        def wave(position):
            return np.interp(np.mod(position, cfg.n_roll) / cfg.n_roll,
                             turn, waveform, period=1.0)

        worst = 0.0
        for e in range(batch.n_episodes):
            for c in range(cfg.n_observations):
                z_ij, z_ji = batch.interactions[e, c, 0], batch.interactions[e, c, 1]
                rebuilt = sigmoid(np.outer(wave(z_ij * cfg.n_roll - r),
                                           wave(z_ji * cfg.n_roll - r)))
                worst = max(worst, np.abs(rebuilt - batch.rates[e, c]).max())
        assert worst < 1e-5


# ------------------------------------------------------------ dis-entanglement
class TestDisentanglement:
    """The §B.3 metric, independently of anything that draws it."""

    @staticmethod
    def _episodes(n_contexts):
        batch = World(SMALL.replace(n_contexts=n_contexts)).sample_episodes(8)
        return batch, run_observers(batch)

    def test_is_the_jeffreys_divergence(self):
        """Jeffreys is the *sum* of both KL directions, not their average."""
        from coggrid.observers import _log_marginal_belief, _normalized_update

        batch, traces = self._episodes(2)
        conditioned = _normalized_update(_log_marginal_belief(traces["joint"], 2))
        naive = _normalized_update(_log_marginal_belief(traces["naive"], 2))
        forward = (conditioned * np.log(conditioned / naive)).sum(-1)
        reverse = (naive * np.log(naive / conditioned)).sum(-1)

        values = disentanglement(
            traces["joint"], traces["naive"], batch, per_variable=True
        )
        assert np.allclose(values, forward + reverse)
        assert values.max() > 0

    def test_naive_update_equals_the_evidence_likelihood(self):
        """Independent derivation: no beliefs, straight from the rate table.

        The naive observer's per-step update *is* the marginal likelihood of that
        step's evidence. Computing it both ways pins the whole update
        construction, and catches the underflow that belief ratios hit once an
        observer becomes confident.
        """
        from coggrid.observers import (
            RATE_FLOOR,
            _log_marginal_belief,
            _normalized_update,
        )

        for n_contexts in (1, 2, 3):
            batch, traces = self._episodes(n_contexts)
            via_belief = _normalized_update(
                _log_marginal_belief(traces["naive"], n_contexts)
            )
            rates = np.clip(batch.marginal_rates, RATE_FLOOR, 1 - RATE_FLOOR)
            obs = batch.observations.astype(float)
            log_lik = np.einsum("bto,bocr->btcr", obs, np.log(rates)) + np.einsum(
                "bto,bocr->btcr", 1 - obs, np.log1p(-rates)
            )
            direct = np.exp(log_lik - log_lik.max(-1, keepdims=True))
            direct /= direct.sum(-1, keepdims=True)
            assert np.allclose(via_belief, direct, atol=1e-12)

    def test_joint_update_differs_from_the_marginal_likelihood(self):
        """The joint observer reads the same evidence differently — that is the point."""
        from coggrid.observers import _log_marginal_belief, _normalized_update

        _, traces = self._episodes(2)
        joint = _normalized_update(_log_marginal_belief(traces["joint"], 2))
        naive = _normalized_update(_log_marginal_belief(traces["naive"], 2))
        assert np.abs(joint - naive).max() > 1e-3

    def test_vanishes_when_there_is_nothing_to_factorize(self):
        batch, traces = self._episodes(1)
        values = disentanglement(traces["joint"], traces["naive"], batch)
        assert values.shape == (8, SMALL.n_steps)
        assert np.allclose(values, 0.0, atol=1e-9)

    @pytest.mark.parametrize("n_contexts", [1, 2, 3])
    def test_goal_slice_matches_the_per_variable_form(self, n_contexts):
        batch, traces = self._episodes(n_contexts)
        per_var = disentanglement(
            traces["joint"], traces["naive"], batch, per_variable=True
        )
        goal = disentanglement(traces["joint"], traces["naive"], batch)
        expected = per_var[np.arange(batch.n_episodes), :, batch.goal_ind]
        assert np.array_equal(goal, expected)


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
        assert np.array_equal(a["active_vars"], b["active_vars"])

    def test_env_episode_is_scoreable(self):
        """The episode an agent saw can be handed straight to an observer."""
        env = CogGridEnv(SMALL, seed=0)
        env.reset()
        traces = run_observers(env.episode)
        assert traces["joint"].belief.shape[0] == 1
