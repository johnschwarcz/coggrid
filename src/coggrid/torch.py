"""CUDA acceleration of the joint observer: optional, and only when asked for.

coggrid never needs PyTorch, and nothing imports it unless you call into this module, directly or with ``device=`` on an observer (``pip install "coggrid[torch]"``).
Only the deterministic, exponential part moves to the device, building each chunk's likelihood table and running Bayes over it,
so a seed gives the same episodes with or without a GPU.
Matrix products run in float64, so neither TF32 nor autocast can touch the result.

For training against the Bayes-optimal posterior, keep it on the device::

    from coggrid.torch import joint_belief

    batch = world.sample_episodes(1024, expose_likelihood=False)
    belief, goal_belief = joint_belief(batch, device="cuda")   # float32 tensors
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import numpy as np

from . import likelihood
from .observers import LIKELIHOOD_FLOOR
from .world import EpisodeBatch

__all__ = ["joint_belief"]

#: Numbers one chunk of episodes may hold on the device, roughly (512 MiB of float64).
#: Far more than ``likelihood._CHUNK_CELLS``, since a GPU is only efficient on big batches of work; like it, this changes speed and memory, never results.
_DEVICE_CHUNK_CELLS = 1 << 26


def joint_belief(batch: EpisodeBatch, *, device: Any, dtype: Any = None) -> tuple[Any, Any]:
    """The joint observer's ``(belief, goal_belief)``, as tensors on ``device``, shaped as in :class:`~coggrid.BeliefTrace`.

    ``dtype`` is ``torch.float32`` (the default, accurate to about ``1e-4`` at 30 steps) or ``torch.float64`` (about ``1e-12``).
    """
    torch = _torch()
    dtype = torch.float32 if dtype is None else dtype
    if dtype not in (torch.float32, torch.float64):
        raise ValueError(f"dtype must be torch.float32 or torch.float64, got {dtype}")
    cfg = batch.cfg
    belief = torch.empty((batch.batch_size, batch.n_steps, cfg.n_contexts, cfg.n_realizations), dtype=dtype, device=device)
    for eps, log_post in _log_posteriors(batch, torch.device(device), dtype):
        belief[eps] = _marginals(log_post.to(dtype), cfg.realization_shape)
    episodes = torch.arange(batch.batch_size, device=device)
    return belief, belief[episodes, :, torch.as_tensor(batch.goal_ind, device=device)]


def _joint_arrays(batch: EpisodeBatch, device: Any, keep_log_posterior: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """``(belief, log_posterior)`` in numpy float64, for ``joint_observer(device=)``."""
    torch = _torch()
    cfg = batch.cfg
    shape = (batch.batch_size, batch.n_steps, *cfg.realization_shape)
    belief = np.empty((*shape[:2], cfg.n_contexts, cfg.n_realizations))
    log_posterior = np.empty(shape) if keep_log_posterior else None
    for eps, log_post in _log_posteriors(batch, torch.device(device), torch.float64):
        belief[eps] = _marginals(log_post, cfg.realization_shape).cpu().numpy()
        if log_posterior is not None:
            log_posterior[eps] = log_post.reshape(-1, *shape[1:]).cpu().numpy()
    return belief, log_posterior


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on the install
        raise ImportError('coggrid.torch needs PyTorch: pip install "coggrid[torch]"') from error
    return torch


def _log_posteriors(batch: EpisodeBatch, device: Any, dtype: Any) -> Iterator[tuple[slice, Any]]:
    """Per chunk of episodes, ``(eps, log_posterior)``: float64 ``(n, n_steps, n_hyp)``.
    A rebuilt table is built in ``dtype``: float64 transcendentals are slow on most GPUs, and float32 results need no more."""
    torch = _torch()
    cfg = batch.cfg
    n_obs, n_hyp = cfg.n_observations, math.prod(cfg.realization_shape)
    obs = torch.as_tensor(batch.observations, device=device, dtype=torch.float64)
    ones = obs.cumsum(1)
    seen = torch.arange(1, batch.n_steps + 1, device=device, dtype=torch.float64)
    counts = torch.cat([ones, seen[:, None] - ones], dim=-1)  # (B, T, 2 n_obs): running counts of ones and zeros
    sinusoids = likelihood.get_sinusoids(cfg)
    limit = math.log((1 - LIKELIHOOD_FLOOR) / LIKELIHOOD_FLOOR)  # clipping likelihoods to [floor, 1 - floor] is clamping logits to +-limit
    step = max(1, _DEVICE_CHUNK_CELLS // (n_hyp * (2 * batch.n_steps + 4 * n_obs)))
    with torch.no_grad(), torch.autocast(device.type, enabled=False):
        for lo in range(0, batch.batch_size, step):
            eps = slice(lo, lo + step)
            if batch.joint_likelihood is not None:  # stored tables: upload them
                table = torch.as_tensor(batch.joint_likelihood[eps], device=device)
                p = table.reshape(-1, n_obs, n_hyp).clamp(LIKELIHOOD_FLOOR, 1 - LIKELIHOOD_FLOOR)
                log_p = torch.cat([p.log(), (-p).log1p()], dim=1)
            else:  # rebuilt from the interactions, as likelihood.joint_likelihood does
                x = _logits(batch.interactions[eps], sinusoids, cfg, device).to(dtype).clamp(-limit, limit)
                log_p = torch.cat([torch.nn.functional.logsigmoid(x), torch.nn.functional.logsigmoid(-x)], dim=1).to(torch.float64)
            yield eps, torch.bmm(counts[eps], log_p)


def _logits(interactions: np.ndarray, sinusoids: np.ndarray, cfg: Any, device: Any) -> Any:
    """``(n, n_obs, n_hyp)`` logits of the chunk's joint table, on ``device``."""
    torch = _torch()
    # (n, k, n_obs, R), as in likelihood.joint_likelihood
    outer_vecs = torch.as_tensor(likelihood.soft_roll(interactions.transpose(0, 2, 1), sinusoids), device=device)
    n, n_ctx = outer_vecs.shape[0], cfg.n_contexts
    logits = torch.zeros((n, cfg.n_observations, *cfg.realization_shape), dtype=torch.float64, device=device)

    def on_grid(slot: int, c: int) -> Any:
        shape = [1] * n_ctx
        shape[c] = cfg.n_realizations
        return outer_vecs[:, slot].reshape(n, cfg.n_observations, *shape)

    likelihood._add_pairwise(logits, n_ctx, on_grid)
    return logits.reshape(n, cfg.n_observations, -1)


def _marginals(log_post: Any, grid: tuple[int, ...]) -> Any:
    """``(n, T, n_hyp)`` log posterior to ``(n, T, n_contexts, n_realizations)`` beliefs."""
    torch = _torch()
    rest = (log_post - log_post.amax(dim=-1, keepdim=True)).exp()  # unnormalized
    sums = []
    for _ in grid[1:]:  # peel one variable off at a time: two full passes in all
        level = rest.reshape(*rest.shape[:2], grid[0], -1)
        sums.append(level.sum(dim=-1))
        rest = level.sum(dim=2)
    belief = torch.stack([*sums, rest], dim=2)
    return belief / belief[:, :, :1].sum(dim=-1, keepdim=True)
