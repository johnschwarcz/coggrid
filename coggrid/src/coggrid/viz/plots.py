"""Figures for inspecting the environment and its observers.

Every function here follows the same three rules, which the original did not:

1. **It takes data, not ``self``.** The argument is an
   :class:`~coggrid.world.EpisodeBatch` or a
   :class:`~coggrid.observers.BeliefTrace` — never a live simulator.
   You can plot a batch loaded from disk six months later.
2. **It returns the figure and never calls ``show()``.** The caller decides
   whether to display, save, or embed in a subplot grid. That is what makes the
   same function usable from a script, a notebook and a CI artefact job.
3. **It accepts ``ax`` (or ``axes``) so it can be composed.** Passing existing
   axes draws into them instead of making a new figure.

Import this module only if you want plots; the environment itself has no
matplotlib dependency.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from ..observers import BeliefTrace, factorization_regret
from ..world import EpisodeBatch
from .style import PALETTE, Palette, label_axes, transparent_cmap

__all__ = [
    "plot_likelihood",
    "plot_trial",
    "plot_performance",
    "plot_regret",
    "summary_figure",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _pair_slice(rates: np.ndarray, i: int, j: int, n_contexts: int) -> np.ndarray:
    """Marginalise a joint rate table down to the ``(i, j)`` realisation plane."""
    other = tuple(i_ for i_ in range(n_contexts) if i_ not in (i, j))
    out = rates.mean(axis=tuple(a for a in other)) if other else rates
    # After averaging, axis order among the kept contexts is preserved.
    return out if i < j else out.T


def _context_pairs(n_contexts: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n_contexts) for j in range(i + 1, n_contexts)]


# --------------------------------------------------------------------------- #
# 1. the likelihood
# --------------------------------------------------------------------------- #
def plot_likelihood(
    batch: EpisodeBatch,
    episode: int = 0,
    channel: int = 0,
    *,
    show_marginal: bool = True,
    axes: Sequence[plt.Axes] | None = None,
    palette: Palette = PALETTE,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Show ``P(observation = 1)`` as a function of the active states' values.

    One panel per pair of active states, plus (optionally) one panel holding the
    marginal — factorised — version of the same thing. The square outline marks
    the realisation that actually occurred; green means that state is the goal.

    Reading it: if the joint panel has structure that survives collapsing to the
    marginal panel, a factorised observer loses nothing. Where the joint panel is
    diagonal, checkered or otherwise non-separable, it does.

    Parameters
    ----------
    batch:
        Any :class:`EpisodeBatch`.
    episode:
        Which episode in the batch to show.
    channel:
        Which observation channel.
    show_marginal:
        Append a panel of ``batch.marginal_rates`` for comparison.
    axes:
        Optional pre-made axes; must be at least as long as the panel count.
    """
    cfg = batch.cfg
    rates = batch.rates[episode, channel]
    ctx_vals = batch.ctx_vals[episode]
    goal_ctx = int(batch.goal_ind[episode])
    pairs = _context_pairs(cfg.n_contexts)
    n_panels = max(1, len(pairs)) + int(show_marginal)

    if axes is None:
        figsize = figsize or (3.2 * n_panels + 1.0, 3.6)
        fig, axes_arr = plt.subplots(1, n_panels, figsize=figsize, squeeze=False)
        axes = list(axes_arr.ravel())
    else:
        fig = axes[0].get_figure()

    common = dict(vmin=0.0, vmax=1.0, cmap="magma", origin="lower")

    if not pairs:  # n_contexts == 1: a single strip over realisations
        ax = axes[0]
        image = ax.imshow(rates[None, :], aspect="auto", **common)
        ax.add_patch(
            Rectangle(
                (ctx_vals[0] - 0.5, -0.5), 1, 1,
                edgecolor=palette.context(0, goal_ctx), facecolor="none", linewidth=2.5,
            )
        )
        ax.set_yticks([])
        label_axes(ax, xlabel="realisation of state 0", title="joint rate")
        fig.colorbar(image, ax=ax, fraction=0.046)
    else:
        for ax, (i, j) in zip(axes, pairs):
            plane = _pair_slice(rates, i, j, cfg.n_contexts)
            image = ax.imshow(plane.T, aspect="equal", **common)
            ax.add_patch(
                Rectangle(
                    (ctx_vals[i] - 0.5, ctx_vals[j] - 0.5), 1, 1,
                    edgecolor=palette.goal if goal_ctx in (i, j) else palette.other,
                    facecolor="none", linewidth=2.5,
                )
            )
            label_axes(
                ax,
                xlabel=f"state {i} (idx {batch.ctx_inds[episode, i]})",
                ylabel=f"state {j} (idx {batch.ctx_inds[episode, j]})",
                title=f"joint: states {i}x{j}",
            )
            fig.colorbar(image, ax=ax, fraction=0.046)

    if show_marginal:
        ax = axes[len(pairs) if pairs else 1]
        marginal = batch.marginal_rates[episode, channel]  # (n_contexts, n_real)
        image = ax.imshow(marginal, aspect="auto", **common)
        for c in range(cfg.n_contexts):
            ax.add_patch(
                Rectangle(
                    (ctx_vals[c] - 0.5, c - 0.5), 1, 1,
                    edgecolor=palette.context(c, goal_ctx),
                    facecolor="none", linewidth=2.5,
                )
            )
        ax.set_yticks(range(cfg.n_contexts))
        ax.set_yticklabels([f"state {c}" for c in range(cfg.n_contexts)])
        label_axes(ax, xlabel="realisation", title="marginal (factorised)")
        fig.colorbar(image, ax=ax, fraction=0.046)

    fig.suptitle(
        f"observation channel {channel}, episode {episode} "
        f"({batch.split} split) — green outline = goal state",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 2. a single trial
# --------------------------------------------------------------------------- #
def plot_trial(
    batch: EpisodeBatch,
    traces: Mapping[str, BeliefTrace] | Iterable[BeliefTrace],
    episode: int = 0,
    *,
    palette: Palette = PALETTE,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Observation stream on top, then one belief panel per active state.

    The top strip is the raw evidence the observers saw. Each panel below shows
    both observers' posteriors over that state's value as they accumulate, with a
    dashed line at the truth. The goal state's panel is outlined in green.

    This is the figure to look at when an aggregate curve surprises you: it shows
    *why* a particular episode was easy or hard.
    """
    if isinstance(traces, Mapping):
        trace_list = list(traces.values())
    else:
        trace_list = list(traces)

    cfg = batch.cfg
    n_steps = batch.n_steps
    goal_ctx = int(batch.goal_ind[episode])

    figsize = figsize or (9.0, 1.6 + 1.5 * cfg.n_contexts)
    fig, axes = plt.subplots(
        cfg.n_contexts + 1, 1,
        figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [0.7] + [1.0] * cfg.n_contexts},
    )
    axes = np.atleast_1d(axes)

    # --- observations -------------------------------------------------------
    ax = axes[0]
    ax.imshow(
        batch.observations[episode].T,
        aspect="auto", cmap="Greys", vmin=0, vmax=1,
        origin="lower", interpolation="nearest",
        extent=(-0.5, n_steps - 0.5, -0.5, cfg.n_observations - 0.5),
    )
    ax.set_yticks(range(cfg.n_observations))
    rate_text = ", ".join(f"{p:.2f}" for p in batch.true_rates[episode])
    label_axes(ax, ylabel="channel", title=f"observations   (true rates: {rate_text})")

    # --- beliefs ------------------------------------------------------------
    steps = np.arange(n_steps)
    for c in range(cfg.n_contexts):
        ax = axes[c + 1]
        for trace in trace_list:
            colour = palette.for_observer(trace.name)
            ax.imshow(
                trace.belief[episode, :, c, :].T,
                aspect="auto", origin="lower", vmin=0, vmax=1,
                cmap=transparent_cmap(colour, f"cg_{trace.name}"),
                interpolation="nearest",
                extent=(-0.5, n_steps - 0.5, -0.5, cfg.n_realizations - 0.5),
            )
            ax.plot(
                steps, trace.estimate[episode, :, c],
                color=colour, linewidth=1.8, label=f"{trace.name} mean",
            )

        truth = int(batch.ctx_vals[episode, c])
        ax.axhline(truth, ls="--", lw=2.0, color=palette.context(c, goal_ctx), zorder=5)
        ax.set_ylim(-0.5, cfg.n_realizations - 0.5)
        is_goal = c == goal_ctx
        label_axes(
            ax,
            ylabel="realisation",
            title=(
                f"state {c} (idx {batch.ctx_inds[episode, c]}), truth={truth}"
                + ("   <- GOAL" if is_goal else "")
            ),
        )
        if is_goal:
            for spine in ax.spines.values():
                spine.set_edgecolor(palette.goal)
                spine.set_linewidth(2.5)
        if c == 0:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    axes[-1].set_xlabel("timestep")
    axes[-1].set_xlim(-0.5, n_steps - 0.5)
    fig.suptitle(f"episode {episode} ({batch.split} split)", fontsize=12)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 3. aggregate performance
# --------------------------------------------------------------------------- #
def plot_performance(
    traces: Mapping[str, BeliefTrace] | Iterable[BeliefTrace],
    *,
    n_samples: int = 60,
    metrics: Sequence[str] = ("accuracy", "p_correct", "mse"),
    axes: Sequence[plt.Axes] | None = None,
    palette: Palette = PALETTE,
    figsize: tuple[float, float] | None = None,
    rng: np.random.Generator | None = None,
) -> Figure:
    """Mean +/- SD over episodes for each metric, with individual episodes behind.

    The gap between the ``joint`` and ``naive`` curves is the headline result: it
    is what a factorised latent representation costs on this environment.
    """
    trace_list = list(traces.values()) if isinstance(traces, Mapping) else list(traces)
    if not trace_list:
        raise ValueError("no traces to plot")

    rng = rng or np.random.default_rng(0)
    n_episodes, n_steps = trace_list[0].accuracy.shape
    steps = np.arange(n_steps)
    picks = rng.choice(n_episodes, size=min(n_samples, n_episodes), replace=False)

    if axes is None:
        figsize = figsize or (4.0 * len(metrics), 3.6)
        fig, axes_arr = plt.subplots(1, len(metrics), figsize=figsize, squeeze=False)
        axes = list(axes_arr.ravel())
    else:
        fig = axes[0].get_figure()

    pretty = {
        "accuracy": "P(mode correct)",
        "p_correct": "P(true value)",
        "mse": "squared error",
    }

    for ax, metric in zip(axes, metrics):
        for trace in trace_list:
            data = getattr(trace, metric)
            colour = palette.for_observer(trace.name)
            mean, sd = data.mean(0), data.std(0)
            ax.plot(steps, data[picks].T, color=colour, alpha=0.06, lw=0.8, zorder=-1)
            ax.fill_between(steps, mean - sd, mean + sd, color=colour, alpha=0.18, lw=0)
            ax.plot(steps, mean, color=colour, lw=2.6, label=trace.name, zorder=3)

        label_axes(
            ax, xlabel="timestep",
            ylabel=pretty.get(metric, metric),
            title=pretty.get(metric, metric),
        )
        ax.set_xlim(0, n_steps - 1)
        if metric in ("accuracy", "p_correct"):
            ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25, color=palette.grid)
        ax.legend(fontsize=9)

    fig.suptitle(f"observer performance ({n_episodes} episodes)", fontsize=12)
    fig.tight_layout()
    return fig


def plot_regret(
    joint: BeliefTrace,
    naive: BeliefTrace,
    *,
    ax: plt.Axes | None = None,
    palette: Palette = PALETTE,
    figsize: tuple[float, float] = (5.0, 3.6),
) -> Figure:
    """Divergence between the optimal and factorised posteriors, over time.

    Rising means the two observers are progressively disagreeing about the goal
    state as evidence accumulates — the factorised one is not merely slower, it
    converges somewhere else.
    """
    regret = factorization_regret(joint, naive)
    steps = np.arange(regret.shape[1])

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    mean, sd = regret.mean(0), regret.std(0)
    ax.fill_between(steps, mean - sd, mean + sd, color=palette.joint, alpha=0.2, lw=0)
    ax.plot(steps, mean, color=palette.joint, lw=2.6)
    ax.axhline(0.0, color=palette.truth, lw=0.8, ls=":")
    label_axes(
        ax, xlabel="timestep", ylabel="symmetric KL (nats)",
        title="factorization regret",
    )
    ax.set_xlim(0, regret.shape[1] - 1)
    ax.grid(alpha=0.25, color=palette.grid)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 4. everything at once
# --------------------------------------------------------------------------- #
def summary_figure(
    batch: EpisodeBatch,
    traces: Mapping[str, BeliefTrace],
    *,
    episode: int = 0,
    palette: Palette = PALETTE,
) -> list[Figure]:
    """Build the standard diagnostic set: likelihood, one trial, performance, regret.

    Returns the figures rather than showing them, so a caller can save them all:

    >>> for i, fig in enumerate(summary_figure(batch, traces)):  # doctest: +SKIP
    ...     fig.savefig(f"figure_{i}.png", dpi=150)
    """
    figures = [
        plot_likelihood(batch, episode=episode, palette=palette),
        plot_trial(batch, traces, episode=episode, palette=palette),
        plot_performance(traces, palette=palette),
    ]
    if "joint" in traces and "naive" in traces:
        figures.append(plot_regret(traces["joint"], traces["naive"], palette=palette))
    return figures
