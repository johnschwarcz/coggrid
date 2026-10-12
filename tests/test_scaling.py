"""Sampling without the likelihood tables, and the observers that rebuild them."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from coggrid import (
    CogGridConfig,
    CogGridEnv,
    EpisodeBatch,
    World,
    disentanglement,
    likelihood,
    run_observers,
)
from coggrid.observers import LIKELIHOOD_FLOOR

SMALL = CogGridConfig(
    n_vars=80, n_held_out_vars=8, n_contexts=2, n_realizations=6,
    n_observations=4, n_steps=15, embedding_dim=10, seed=0,
)

FIELDS = ("ctx_inds", "goal_ind", "ctx_vals", "goal_val", "p_obs__vals",
          "observations", "interactions")


def _pair(cfg: CogGridConfig, split: str = "held_out"):
    """The same episodes, sampled with and without tables."""
    world = World(cfg)
    full = world.sample_episodes(24, rng=7, split=split)
    return full, world.sample_episodes(24, rng=7, split=split, expose_likelihood=False)


def _textbook(batch: EpisodeBatch) -> np.ndarray:
    """Per-variable beliefs by step-by-step Bayes over the joint grid."""
    p = np.clip(batch.joint_likelihood, LIKELIHOOD_FLOOR, 1 - LIKELIHOOD_FLOOR)
    grid = tuple(range(1, batch.cfg.n_contexts + 1))
    log_post, beliefs = np.zeros((batch.batch_size, *p.shape[2:])), []
    for t in range(batch.n_steps):
        o = batch.observations[:, t].astype(float)
        log_post = log_post + np.einsum("bo,bo...->b...", o, np.log(p))
        log_post = log_post + np.einsum("bo,bo...->b...", 1 - o, np.log1p(-p))
        post = np.exp(log_post - log_post.max(axis=grid, keepdims=True))
        post /= post.sum(axis=grid, keepdims=True)
        beliefs.append(np.stack(
            [post.sum(axis=tuple(a for a in grid if a != c)) for c in grid], axis=1
        ))
    return np.stack(beliefs, axis=1)


@pytest.mark.parametrize("split", ["held_out", "train"])
@pytest.mark.parametrize(
    "changes",
    [
        {"n_contexts": 1},
        {"n_contexts": 2},
        {"n_contexts": 3},
        {"n_contexts": 3, "allow_repeated_vars": True},
        {"n_contexts": 3, "n_observations": 1, "embedding_dim": 1},
    ],
)
def test_episodes_are_bit_for_bit_the_same(changes, split):
    full, lite = _pair(SMALL.replace(**changes), split)
    for name in FIELDS:
        assert np.array_equal(getattr(full, name), getattr(lite, name)), name
    assert lite.joint_likelihood is None and lite.marginal_likelihood is None
    rebuilt = likelihood.joint_likelihood(full.cfg, lite.interactions)
    assert np.array_equal(rebuilt, full.joint_likelihood)


def test_the_world_stream_moves_on_as_with_tables():
    a, b = World(SMALL), World(SMALL)
    a.sample_episodes(10)
    b.sample_episodes(10, expose_likelihood=False)
    assert a.rng.bit_generator.state == b.rng.bit_generator.state


def test_many_contexts():
    cfg = SMALL.replace(n_vars=400, n_held_out_vars=100, n_contexts=48)
    batch = World(cfg).sample_episodes(8, expose_likelihood=False)
    assert batch.interactions.shape == (8, cfg.n_observations, 48 * 47)


@pytest.mark.parametrize("n_contexts", [1, 2, 3])
def test_observers_rebuild_the_tables(n_contexts, monkeypatch):
    """Textbook Bayes, and the same traces without tables, whatever the chunking."""
    full, lite = _pair(SMALL.replace(n_contexts=n_contexts))
    traces = run_observers(full)
    textbook = _textbook(full)
    np.testing.assert_allclose(traces["joint"].belief, textbook, rtol=0, atol=1e-12)
    monkeypatch.setattr(likelihood, "_CHUNK_CELLS", 1)  # one episode, and one pair, at a time
    for name, trace in run_observers(lite).items():
        assert np.array_equal(trace.belief, traces[name].belief), name
        assert np.array_equal(trace.log_posterior, traces[name].log_posterior), name


def test_dropping_the_log_posterior():
    full, _ = _pair(SMALL.replace(n_contexts=3))
    kept, lean = run_observers(full), run_observers(full, keep_log_posterior=False)
    assert lean["joint"].log_posterior is None
    assert np.array_equal(kept["joint"].belief, lean["joint"].belief)
    with pytest.raises(ValueError, match="keep_log_posterior"):
        disentanglement(lean["joint"], lean["naive"], full)


def test_save_and_load_without_tables(tmp_path):
    _, lite = _pair(SMALL)
    back = EpisodeBatch.load(lite.save(tmp_path / "lite"))
    assert back.joint_likelihood is None and back.select(slice(0, 3)).marginal_likelihood is None
    assert np.array_equal(back.observations, lite.observations)


def test_the_env_builds_tables_only_to_expose_them():
    plain, exposing = CogGridEnv(SMALL, seed=0), CogGridEnv(SMALL, expose_likelihood=True, seed=0)
    for _ in range(3):
        plain.reset()
        info = exposing.reset()[1]
        a, b = plain.episode, exposing.episode
        assert a.joint_likelihood is None and b.joint_likelihood is not None
        assert np.array_equal(a.observations, b.observations)
        assert np.array_equal(info["joint_likelihood"], b.joint_likelihood[0])


def test_the_core_never_imports_torch():
    """`pip install coggrid` must stay numpy-only, whatever else is installed."""
    code = (
        "import sys\n"
        "from coggrid import CogGridConfig, World, run_observers\n"
        "b = World(CogGridConfig(n_vars=60, seed=0)).sample_episodes(4, expose_likelihood=False)\n"
        "run_observers(b)\n"
        "assert 'torch' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
