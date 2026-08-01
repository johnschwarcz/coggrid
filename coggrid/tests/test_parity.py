"""Numerical parity with the original implementation.

``tests/reference.npz`` was produced by driving the original ``main/`` code
(env-only path, PyTorch stubbed out) with fixed inputs. This test feeds the
*same* embeddings, contexts, realisations and observations through the
refactored package and requires the likelihoods, beliefs and metrics to match.

Sampling is deliberately excluded: the refactor moved from the legacy global
``np.random`` to explicit ``Generator`` objects, so random *draws* differ by
construction. Everything downstream of the draws must not.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from coggrid import CogGridConfig
from coggrid.generative import (
    joint_likelihood,
    marginal_likelihood,
    observation_rates,
    value_profile,
)
from coggrid.observers import (
    factorization_regret,
    joint_observer,
    naive_observer,
)
from coggrid.world import EpisodeBatch

REFERENCE = pathlib.Path(__file__).parent / "reference.npz"

CFG = CogGridConfig(
    n_states=40,
    n_held_out_states=4,
    n_observations=3,
    n_contexts=2,
    n_realizations=6,
    embedding_dim=8,
    n_episodes=32,
    n_steps=12,
    likelihood_temp=2.0,
    likelihood_freq=1.0,
)


@pytest.fixture(scope="module")
def ref():
    if not REFERENCE.exists():  # pragma: no cover
        pytest.skip(f"missing golden reference: {REFERENCE}")
    with np.load(REFERENCE) as data:
        return {k: data[k] for k in data.files}


@pytest.fixture(scope="module")
def batch(ref) -> EpisodeBatch:
    """Rebuild an EpisodeBatch from the reference's *inputs* only."""
    rates, interactions = joint_likelihood(
        CFG, ref["all_K"], ref["all_Q"], ref["ctx_inds"]
    )
    ctx_vals = ref["ctx_vals"]
    return EpisodeBatch(
        cfg=CFG,
        split="held_out",
        ctx_inds=ref["ctx_inds"],
        goal_ind=ref["goal_ind"],
        ctx_vals=ctx_vals,
        goal_value=ctx_vals[np.arange(len(ctx_vals)), ref["goal_ind"]],
        rates=rates,
        marginal_rates=marginal_likelihood(rates, CFG.n_contexts),
        true_rates=observation_rates(rates, ctx_vals),
        observations=ref["obs_flat"],
        interactions=interactions,
    )


# --------------------------------------------------------------------- pieces
def test_value_profile(ref):
    assert np.allclose(value_profile(CFG), ref["roll_V_range"])


def test_joint_likelihood(ref, batch):
    assert batch.rates.shape == ref["joint_likelihood"].shape
    assert np.allclose(batch.rates, ref["joint_likelihood"])


def test_interaction_strengths(ref, batch):
    assert np.allclose(batch.interactions, ref["joint_Z"])


def test_marginal_likelihood(ref, batch):
    assert np.allclose(batch.marginal_rates, ref["naive_likelihood"])


def test_observation_rates(ref, batch):
    assert np.allclose(batch.true_rates, ref["pobs"])


def test_rates_are_not_saturated(batch):
    """Guards the log-space inference: clipping must be a no-op here."""
    assert batch.rates.min() > 1e-6
    assert batch.rates.max() < 1 - 1e-6


# ------------------------------------------------------------------ observers
@pytest.mark.parametrize("name", ["joint", "naive"])
def test_beliefs(ref, batch, name):
    trace = joint_observer(batch) if name == "joint" else naive_observer(batch)
    assert np.allclose(trace.belief, ref[f"{name}_belief"], atol=1e-10)


@pytest.mark.parametrize("name", ["joint", "naive"])
@pytest.mark.parametrize("metric", ["acc", "TP", "mse", "est"])
def test_metrics(ref, batch, name, metric):
    trace = joint_observer(batch) if name == "joint" else naive_observer(batch)
    mine = {
        "acc": trace.accuracy,
        "TP": trace.p_correct,
        "mse": trace.mse,
        "est": trace.estimate,
    }[metric]
    assert np.allclose(mine, ref[f"{name}_{metric}"], atol=1e-10)


def test_factorization_regret(ref, batch):
    regret = factorization_regret(joint_observer(batch), naive_observer(batch))
    assert np.allclose(regret, ref["SII"], atol=1e-9)


def test_joint_beats_naive(batch):
    """The headline property of the environment, asserted rather than assumed."""
    joint = joint_observer(batch).final()
    naive = naive_observer(batch).final()
    assert joint["p_correct"] > naive["p_correct"]
