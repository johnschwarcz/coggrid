"""The optional PyTorch path: the joint observer, on a device.

Skipped without torch, which coggrid never needs. Runs on the CPU build too, and
on the GPU as well when CUDA is available.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from coggrid import CogGridConfig, World, joint_observer  # noqa: E402
from coggrid.torch import joint_belief  # noqa: E402

SMALL = CogGridConfig(
    n_vars=80, n_held_out_vars=8, n_contexts=3, n_realizations=5,
    n_observations=4, n_steps=25, embedding_dim=10, seed=0,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(scope="module", params=[True, False], ids=["tables", "table-free"])
def batch(request):
    return World(SMALL).sample_episodes(40, rng=1, expose_likelihood=request.param)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("keep", [True, False])
def test_device_traces_match_numpy(batch, device, keep):
    expected, got = joint_observer(batch), joint_observer(
        batch, keep_log_posterior=keep, device=device
    )
    np.testing.assert_allclose(got.belief, expected.belief, rtol=0, atol=1e-12)
    assert np.array_equal(got.accuracy, expected.accuracy)
    # A log belief near 0 carries the rounding of the larger log posterior it
    # comes from, hence the atol.
    np.testing.assert_allclose(
        got.log_goal_belief, expected.log_goal_belief, rtol=1e-12, atol=1e-12
    )
    if keep:
        np.testing.assert_allclose(got.log_posterior, expected.log_posterior, rtol=1e-12)
    else:
        assert got.log_posterior is None


@pytest.mark.parametrize("device", DEVICES)
def test_log_goal_belief_where_beliefs_underflow(device, monkeypatch):
    """At long horizons the goal belief underflows to 0, and its log must not.

    Three of these episodes come near underflow and one does not, which the
    observers compute two ways: here with all four in one chunk, then one per chunk.
    """
    world = World(SMALL.replace(n_steps=1000))
    long = world.sample_episodes(4, rng=2, expose_likelihood=False)
    expected = joint_observer(long, keep_log_posterior=False)
    deep = expected.goal_belief.min(axis=(1, 2)) < 1e-250
    assert deep.any() and not deep.all()
    assert (expected.goal_belief == 0).any()
    chunked = joint_observer(long, keep_log_posterior=False, device=device)
    monkeypatch.setattr("coggrid.torch._DEVICE_CHUNK_CELLS", 1)  # an episode at a time
    for got in (chunked, joint_observer(long, keep_log_posterior=False, device=device)):
        assert np.isfinite(got.log_goal_belief).all()
        np.testing.assert_allclose(
            got.log_goal_belief, expected.log_goal_belief, rtol=1e-12, atol=1e-10
        )


@pytest.mark.parametrize("device", DEVICES)
def test_beliefs_for_training(batch, device):
    expected = joint_observer(batch)
    belief, goal = joint_belief(batch, device=device)  # float32 by default
    assert belief.dtype == torch.float32 and belief.device.type == device
    np.testing.assert_allclose(belief.cpu().numpy(), expected.belief, atol=1e-4)
    np.testing.assert_allclose(goal.cpu().numpy(), expected.goal_belief, atol=1e-4)
    # Reduced-precision autocast, common in training loops, must not touch them.
    autocast = torch.bfloat16 if device == "cpu" else torch.float16
    with torch.autocast(device, dtype=autocast):
        belief, _ = joint_belief(batch, device=device, dtype=torch.float64)
    np.testing.assert_allclose(belief.cpu().numpy(), expected.belief, rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="float32 or torch.float64"):
        joint_belief(batch, device=device, dtype=torch.float16)
