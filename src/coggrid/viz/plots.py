"""Figures for inspecting the environment and its observers.

Every function here follows the same three rules:

1. **It takes data, not a live simulator.** The argument is an
   :class:`~coggrid.world.EpisodeBatch` or a
   :class:`~coggrid.observers.BeliefTrace`, so you can plot a batch loaded from
   disk six months later. The one exception is
   :func:`plot_interaction_phases`, which also needs the
   :class:`~coggrid.World`: it draws the variable embeddings, and those live on
   the world rather than on any batch drawn from it.
2. **It returns the figure and never calls ``show()``.** The caller decides
   whether to display, save, or embed in a subplot grid. That is what makes the
   same function usable from a script, a notebook and a CI artifact job.
3. **It accepts ``ax`` (or ``axes``) so it can be composed.** Passing existing
   axes draws into them instead of making a new figure.

Import this module only if you want plots; the environment itself has no
matplotlib dependency.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .. import generative as gen
from ..observers import BeliefTrace, factorization_regret
from ..world import EpisodeBatch
from .style import PALETTE, Palette, label_axes, transparent_cmap

__all__ = [
    "display_pairs",
    "MAX_PAIRS",
    "plot_likelihood",
    "plot_interaction_phases",
    "plot_evidence_likelihood",
    "plot_trial",
    "plot_episode",
    "plot_performance",
    "plot_regret",
    "plot_relative_accuracy",
    "plot_regret_vs_accuracy",
    "plot_map_agreement",
    "plot_belief_profile",
    "plot_confidence_density",
    "plot_regret_analysis",
    "plot_belief_shape",
    "summary_figure",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _pair_slice(rates: np.ndarray, i: int, j: int, n_contexts: int) -> np.ndarray:
    """Marginalise a joint rate table down to the ``(i, j)`` realization plane."""
    other = tuple(i_ for i_ in range(n_contexts) if i_ not in (i, j))
    out = rates.mean(axis=tuple(a for a in other)) if other else rates
    # After averaging, axis order among the kept contexts is preserved.
    return out if i < j else out.T


def _context_pairs(n_contexts: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n_contexts) for j in range(i + 1, n_contexts)]


#: How many pairwise panels a figure will draw before it stops adding more.
MAX_PAIRS = 3


def display_pairs(
    n_contexts: int, goal_context: int = 0, max_pairs: int = MAX_PAIRS
) -> list[tuple[int, int]]:
    """Which variable pairs are worth drawing, in priority order.

    Pairs grow as ``C(C-1)/2``, so at four active variables a figure showing all
    of them is already unreadable. Pairs involving the **goal** come first —
    that is the variable being scored, so those are the interactions that decide
    the episode — and the list is capped. Returns ``[]`` for a single variable,
    where there is no pair to show at all.
    """
    if n_contexts < 2:
        return []
    goal_first = [
        (min(goal_context, c), max(goal_context, c))
        for c in range(n_contexts)
        if c != goal_context
    ]
    rest = [p for p in _context_pairs(n_contexts) if goal_context not in p]
    return (goal_first + rest)[:max_pairs]


def _n_joint_panels(
    n_contexts: int, goal_context: int = 0, max_pairs: int = MAX_PAIRS
) -> int:
    """Joint panels needed: one per shown pair, or one strip when there is none."""
    return max(1, len(display_pairs(n_contexts, goal_context, max_pairs)))


def _wants_marginal(n_contexts: int, show_marginal: bool) -> bool:
    """Whether a marginal strip carries any information.

    With one active variable there is nothing to marginalize over, so the strip
    would be a pixel-for-pixel copy of the joint panel above it.
    """
    return show_marginal and n_contexts > 1


#: Rates are probabilities, so every heatmap is pinned to this range. Fixing it
#: is what lets the figures drop colorbars: the scale never varies between
#: panels, episodes or runs, so it is stated once in the title instead.
RATE_SCALE = dict(vmin=0.0, vmax=1.0, cmap="magma", origin="lower")


def _observer_pair(
    traces: Mapping[str, BeliefTrace] | Iterable[BeliefTrace],
) -> tuple[BeliefTrace, BeliefTrace]:
    """Pull ``(joint, naive)`` out of whatever container the caller passed."""
    items = list(traces.values()) if isinstance(traces, Mapping) else list(traces)
    by_name = {t.name: t for t in items}
    missing = {"joint", "naive"} - set(by_name)
    if missing:
        raise ValueError(
            f"this figure compares the two observers and is missing {sorted(missing)}; "
            "pass the output of run_observers()"
        )
    return by_name["joint"], by_name["naive"]


def _center_on_truth(
    final_belief: np.ndarray, truth: np.ndarray, n_realizations: int
) -> np.ndarray:
    """Roll each episode's posterior so the true value sits at the middle index.

    Averaging raw posteriors washes out their shape, because every episode has a
    different true value. Aligning them first is what makes "how much mass lands
    one step away from the truth" a meaningful question.
    """
    shift = n_realizations // 2 - np.asarray(truth, dtype=int)
    cols = (np.arange(n_realizations)[None, :] - shift[:, None]) % n_realizations
    return np.take_along_axis(final_belief, cols, axis=1)


def _joint_cell(
    fig: Figure, cell: Any, show_marginal: bool
) -> tuple[plt.Axes, plt.Axes | None, plt.Axes | None]:
    """Split one gridspec cell into a joint panel plus top/right marginal strips."""
    if not show_marginal:
        return fig.add_subplot(cell), None, None
    gs = cell.subgridspec(
        2, 2, width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.04, hspace=0.04
    )
    ax_joint = fig.add_subplot(gs[1, 0])
    return (
        ax_joint,
        fig.add_subplot(gs[0, 0], sharex=ax_joint),
        fig.add_subplot(gs[1, 1], sharey=ax_joint),
    )


def _draw_marginal(
    ax: plt.Axes, profile: np.ndarray, color: str, truth: int, vertical: bool
) -> None:
    """One variable's factorized rate, drawn against the edge of its joint panel."""
    grid = np.arange(profile.size)
    if vertical:
        ax.fill_betweenx(grid, profile, color=color, alpha=0.35, lw=0)
        ax.plot(profile, grid, color=color, lw=1.6)
        ax.axhline(truth, color=color, ls="--", lw=1.4)
        ax.set_xlim(0.0, 1.0)
    else:
        ax.fill_between(grid, profile, color=color, alpha=0.35, lw=0)
        ax.plot(grid, profile, color=color, lw=1.6)
        ax.axvline(truth, color=color, ls="--", lw=1.4)
        ax.set_ylim(0.0, 1.0)
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


# --------------------------------------------------------------------------- #
# 1. the likelihood
# --------------------------------------------------------------------------- #
def plot_likelihood(
    batch: EpisodeBatch,
    episode: int = 0,
    channel: int = 0,
    *,
    show_marginal: bool = True,
    max_pairs: int = MAX_PAIRS,
    subplot_spec: Any | None = None,
    fig: Figure | None = None,
    palette: Palette = PALETTE,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Show ``P(observation = 1)`` as a function of the active variables' values.

    One panel per pair of active variables, with each variable's *factorized*
    rate drawn as a profile against the top and right edges — the same numbers a
    naive observer sees in ``batch.marginal_rates``. The outline marks the
    realization that actually occurred; green means that variable is the goal,
    and the dashed line on each profile marks the same truth.

    Reading it: if the joint surface is the outer product of the two edge
    profiles, a factorized observer loses nothing. Where the surface is diagonal,
    checkered or otherwise non-separable, it does — and the edge profiles are
    exactly what it collapses to.

    There is no colorbar. Rates are probabilities and every panel is pinned to
    ``[0, 1]`` on ``magma`` (dark = 0, light = 1), so the scale is stated in the
    title rather than redrawn beside each panel.

    Parameters
    ----------
    batch:
        Any :class:`EpisodeBatch`.
    episode:
        Which episode in the batch to show.
    channel:
        Which observation channel.
    show_marginal:
        Draw the factorized profiles along the panel edges.
    subplot_spec, fig:
        Draw into a cell of an existing figure instead of making a new one. A
        ``SubplotSpec`` rather than axes, because each panel is itself a small
        grid (joint plus two marginals); see :func:`plot_episode`.
    """
    cfg = batch.cfg
    rates = batch.rates[episode, channel]
    marginal = batch.marginal_rates[episode, channel]  # (n_contexts, n_realizations)
    ctx_vals = batch.ctx_vals[episode]
    goal_ctx = int(batch.goal_ind[episode])
    pairs = display_pairs(cfg.n_contexts, goal_ctx, max_pairs)
    n_joint = max(1, len(pairs))
    show_marginal = _wants_marginal(cfg.n_contexts, show_marginal)

    owns_figure = subplot_spec is None
    if owns_figure:
        figsize = figsize or (3.5 * n_joint + 0.6, 3.9)
        fig = plt.figure(figsize=figsize, layout="constrained")
        cells = fig.add_gridspec(1, n_joint, wspace=0.3)
    elif fig is None:
        raise ValueError("pass the parent `fig` alongside `subplot_spec`")
    else:
        cells = subplot_spec.subgridspec(1, n_joint, wspace=0.3)

    if not pairs:  # n_contexts == 1: a single strip over realizations
        ax = fig.add_subplot(cells[0, 0])
        ax.imshow(rates[None, :], aspect="auto", **RATE_SCALE)
        ax.add_patch(
            Rectangle(
                (ctx_vals[0] - 0.5, -0.5), 1, 1,
                edgecolor=palette.context(0, goal_ctx), facecolor="none", linewidth=2.5,
            )
        )
        ax.set_yticks([])
        label_axes(ax, xlabel="realization of var 0", title="joint rate")
    else:
        for cell, (i, j) in zip(cells, pairs, strict=False):
            ax_j, ax_top, ax_right = _joint_cell(fig, cell, show_marginal)
            plane = _pair_slice(rates, i, j, cfg.n_contexts)
            ax_j.imshow(plane.T, aspect="auto", **RATE_SCALE)
            ax_j.add_patch(
                Rectangle(
                    (ctx_vals[i] - 0.5, ctx_vals[j] - 0.5), 1, 1,
                    edgecolor=palette.goal if goal_ctx in (i, j) else palette.other,
                    facecolor="none", linewidth=2.5,
                )
            )
            label_axes(
                ax_j,
                xlabel=f"var {i} (idx {batch.ctx_inds[episode, i]})",
                ylabel=f"var {j} (idx {batch.ctx_inds[episode, j]})",
            )
            # The title rides on the top marginal when there is one, so it does
            # not land in the middle of the panel stack.
            (ax_top or ax_j).set_title(f"joint: vars {i}x{j}", fontsize=11)

            if show_marginal:
                _draw_marginal(
                    ax_top, marginal[i], palette.context(i, goal_ctx),
                    int(ctx_vals[i]), vertical=False,
                )
                _draw_marginal(
                    ax_right, marginal[j], palette.context(j, goal_ctx),
                    int(ctx_vals[j]), vertical=True,
                )

    if owns_figure:
        fig.suptitle(
            f"observation channel {channel}, episode {episode} "
            f"({batch.split} split) — rate 0 (dark) to 1 (light), "
            "edge profiles = factorized view, green = goal variable",
            fontsize=10,
        )
    return fig


# --------------------------------------------------------------------------- #
# 2. a single trial
# --------------------------------------------------------------------------- #
def plot_trial(
    batch: EpisodeBatch,
    traces: Mapping[str, BeliefTrace] | Iterable[BeliefTrace],
    episode: int = 0,
    *,
    axes: Sequence[plt.Axes] | None = None,
    palette: Palette = PALETTE,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Observation stream on top, then one belief panel per active variable.

    The top strip is the raw evidence the observers saw. Each panel below shows
    both observers' posteriors over that variable's value as they accumulate, with a
    dashed line at the truth. The goal variable's panel is outlined in green.

    This is the figure to look at when an aggregate curve surprises you: it shows
    *why* a particular episode was easy or hard.

    ``axes`` must hold ``n_contexts + 1`` axes stacked top to bottom; pass them
    to draw this as a column of a larger figure (see :func:`plot_episode`).
    """
    trace_list = list(traces.values()) if isinstance(traces, Mapping) else list(traces)

    cfg = batch.cfg
    n_steps = batch.n_steps
    goal_ctx = int(batch.goal_ind[episode])

    owns_figure = axes is None
    if axes is None:
        figsize = figsize or (9.0, 1.6 + 1.5 * cfg.n_contexts)
        fig, axes_arr = plt.subplots(
            cfg.n_contexts + 1, 1,
            figsize=figsize, sharex=True,
            gridspec_kw={"height_ratios": [0.7] + [1.0] * cfg.n_contexts},
        )
        axes = list(np.atleast_1d(axes_arr))
    else:
        axes = list(axes)
        if len(axes) < cfg.n_contexts + 1:
            raise ValueError(
                f"plot_trial needs {cfg.n_contexts + 1} axes "
                f"(observations + one per active variable), got {len(axes)}"
            )
        fig = axes[0].get_figure()

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
    label_axes(ax, ylabel="channel")
    # Two lines at a smaller size: this panel is one column wide when composed
    # into plot_episode, and a single-line rate list overflows it.
    ax.set_title(f"observations\n(true rates: {rate_text})", fontsize=9)

    # --- beliefs ------------------------------------------------------------
    steps = np.arange(n_steps)
    for c in range(cfg.n_contexts):
        ax = axes[c + 1]
        for trace in trace_list:
            color = palette.for_observer(trace.name)
            ax.imshow(
                trace.belief[episode, :, c, :].T,
                aspect="auto", origin="lower", vmin=0, vmax=1,
                cmap=transparent_cmap(color, f"cg_{trace.name}"),
                interpolation="nearest",
                extent=(-0.5, n_steps - 0.5, -0.5, cfg.n_realizations - 0.5),
            )
            ax.plot(
                steps, trace.estimate[episode, :, c],
                color=color, linewidth=1.8, label=f"{trace.name} mean",
            )

        truth = int(batch.ctx_vals[episode, c])
        ax.axhline(truth, ls="--", lw=2.0, color=palette.context(c, goal_ctx), zorder=5)
        ax.set_ylim(-0.5, cfg.n_realizations - 0.5)
        is_goal = c == goal_ctx
        label_axes(
            ax,
            ylabel="realization",
            title=(
                f"var {c} (idx {batch.ctx_inds[episode, c]}), truth={truth}"
                + ("   <- GOAL" if is_goal else "")
            ),
        )
        if is_goal:
            for spine in ax.spines.values():
                spine.set_edgecolor(palette.goal)
                spine.set_linewidth(2.5)
        if c == 0:
            # Compact: this panel is one column wide inside plot_episode.
            ax.legend(
                loc="upper right", fontsize=7, framealpha=0.85,
                labelspacing=0.2, handlelength=1.1, borderpad=0.3,
            )

    # Shared x axis: only the bottom panel carries ticks and a label.
    for ax in axes[: cfg.n_contexts]:
        ax.tick_params(labelbottom=False)
    axes[cfg.n_contexts].set_xlabel("timestep")
    axes[cfg.n_contexts].set_xlim(-0.5, n_steps - 0.5)

    if owns_figure:
        fig.suptitle(f"episode {episode} ({batch.split} split)", fontsize=12)
        fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 3. one episode, end to end
# --------------------------------------------------------------------------- #
def plot_episode(
    batch: EpisodeBatch,
    traces: Mapping[str, BeliefTrace] | Iterable[BeliefTrace],
    episode: int = 0,
    channel: int = 0,
    *,
    show_marginal: bool = True,
    max_pairs: int = MAX_PAIRS,
    palette: Palette = PALETTE,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """:func:`plot_likelihood` and :func:`plot_trial` side by side, one episode.

    The likelihood panels on the left say what *could* have been inferred from
    this episode's rate surface; the trial column on the right says what the two
    observers actually did infer from the samples they saw. Reading them
    together is the point — a large joint/naive gap in the belief panels should
    correspond to visible non-separability in the rate panel.

    The trial column is one panel wide, so it stays legible next to the
    likelihood panels instead of dominating the figure.
    """
    cfg = batch.cfg
    n_joint = _n_joint_panels(cfg.n_contexts, int(batch.goal_ind[episode]), max_pairs)
    n_rows = cfg.n_contexts + 1

    figsize = figsize or (
        3.5 * n_joint + 3.6,
        max(3.8, 1.7 + 1.5 * cfg.n_contexts),
    )
    fig = plt.figure(figsize=figsize, layout="constrained")
    outer = fig.add_gridspec(1, 2, width_ratios=[n_joint, 1], wspace=0.2)

    plot_likelihood(
        batch, episode=episode, channel=channel, show_marginal=show_marginal,
        max_pairs=max_pairs, subplot_spec=outer[0, 0], fig=fig, palette=palette,
    )

    # The trial stack shares one x axis down the whole right-hand column. Only
    # the bottom panel keeps tick labels, so each gap needs to hold a title and
    # nothing else — anything more and the gaps outgrow the panels themselves.
    inner = outer[0, 1].subgridspec(
        n_rows, 1, height_ratios=[0.7] + [1.0] * cfg.n_contexts, hspace=0.12
    )
    trial_axes = [fig.add_subplot(inner[0])]
    trial_axes += [
        fig.add_subplot(inner[r], sharex=trial_axes[0]) for r in range(1, n_rows)
    ]
    plot_trial(batch, traces, episode=episode, axes=trial_axes, palette=palette)

    fig.suptitle(
        f"episode {episode} ({batch.split} split), observation channel {channel}"
        "  —  rate 0 (dark) to 1 (light), green = goal variable",
        fontsize=11,
    )
    return fig


# --------------------------------------------------------------------------- #
# 4. aggregate performance
# --------------------------------------------------------------------------- #
#: y-axis label per metric. ``regret`` is the odd one out: it is computed
#: *between* the two observers rather than read off a single trace.
METRIC_LABELS = {
    "accuracy": "P(mode correct)",
    "p_correct": "P(true value)",
    "mse": "squared error",
    "regret": "symmetric KL (nats)",
}
METRIC_TITLES = {"regret": "factorization regret"}


def _draw_band(
    ax: plt.Axes,
    data: np.ndarray,
    color: str,
    label: str,
    *,
    picks: np.ndarray,
    alpha: float = 0.06,
) -> None:
    """Individual episode trajectories, then mean +/- SD, then the mean on top.

    ``data`` is ``(n_episodes, n_steps)``. The single-episode lines are what stop
    a tidy mean curve from hiding a bimodal or heavy-tailed distribution.

    A per-episode 0/1 indicator (``accuracy``) is the exception: its trajectories
    are vertical sawtooth spanning the whole axis, which hides the very mean they
    are meant to contextualize. For those, the band alone is drawn.
    """
    steps = np.arange(data.shape[1])
    mean, sd = data.mean(0), data.std(0)
    if not np.isin(data[picks], (0.0, 1.0)).all():
        ax.plot(steps, data[picks].T, color=color, alpha=alpha, lw=0.8, zorder=-1)
    ax.fill_between(steps, mean - sd, mean + sd, color=color, alpha=0.18, lw=0)
    ax.plot(steps, mean, color=color, lw=2.6, label=label, zorder=3)


def plot_performance(
    traces: Mapping[str, BeliefTrace] | Iterable[BeliefTrace],
    *,
    n_samples: int = 60,
    metrics: Sequence[str] = ("accuracy", "p_correct", "mse", "regret"),
    axes: Sequence[plt.Axes] | None = None,
    palette: Palette = PALETTE,
    figsize: tuple[float, float] | None = None,
    rng: np.random.Generator | None = None,
) -> Figure:
    """Mean +/- SD over episodes for each metric, with individual episodes behind.

    The gap between the ``joint`` and ``naive`` curves is the headline result: it
    is what a factorized latent representation costs on this environment. The
    final ``regret`` column quantifies that same gap as a divergence between the
    two posteriors, so the whole story sits in one row of panels.

    ``regret`` needs both observers. If only one is supplied the column is
    dropped rather than raising, so this still works on a single trace.
    """
    trace_list = list(traces.values()) if isinstance(traces, Mapping) else list(traces)
    if not trace_list:
        raise ValueError("no traces to plot")

    by_name = {trace.name: trace for trace in trace_list}
    has_both = "joint" in by_name and "naive" in by_name
    columns = [m for m in metrics if m != "regret" or has_both]
    if not columns:
        raise ValueError("no metrics left to plot")

    rng = rng or np.random.default_rng(0)
    n_episodes, n_steps = trace_list[0].accuracy.shape
    picks = rng.choice(n_episodes, size=min(n_samples, n_episodes), replace=False)

    if axes is None:
        figsize = figsize or (4.0 * len(columns), 3.6)
        fig, axes_arr = plt.subplots(1, len(columns), figsize=figsize, squeeze=False)
        axes = list(axes_arr.ravel())
    else:
        fig = axes[0].get_figure()

    for ax, metric in zip(axes, columns, strict=False):
        if metric == "regret":
            # One line per episode here rather than two, so lift the alpha to
            # keep the visual density comparable with the other columns.
            _draw_band(
                ax,
                factorization_regret(by_name["joint"], by_name["naive"]),
                palette.regret, "joint vs naive", picks=picks, alpha=0.10,
            )
            ax.axhline(0.0, color=palette.truth, lw=0.8, ls=":")
        else:
            for trace in trace_list:
                _draw_band(
                    ax, getattr(trace, metric),
                    palette.for_observer(trace.name), trace.name, picks=picks,
                )

        label = METRIC_LABELS.get(metric, metric)
        label_axes(ax, xlabel="timestep", ylabel=label,
                   title=METRIC_TITLES.get(metric, label))
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
    n_samples: int = 60,
    ax: plt.Axes | None = None,
    palette: Palette = PALETTE,
    figsize: tuple[float, float] = (5.0, 3.6),
    rng: np.random.Generator | None = None,
) -> Figure:
    """Divergence between the optimal and factorized posteriors, over time.

    The same panel :func:`plot_performance` draws as its ``regret`` column, on
    its own. Rising means the two observers are progressively disagreeing about
    the goal variable as evidence accumulates — the factorized one is not merely
    slower, it converges somewhere else.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    regret = factorization_regret(joint, naive)
    rng = rng or np.random.default_rng(0)
    picks = rng.choice(
        regret.shape[0], size=min(n_samples, regret.shape[0]), replace=False
    )

    _draw_band(ax, regret, palette.regret, "joint vs naive", picks=picks, alpha=0.10)
    ax.axhline(0.0, color=palette.truth, lw=0.8, ls=":")
    label_axes(
        ax, xlabel="timestep", ylabel=METRIC_LABELS["regret"],
        title=METRIC_TITLES["regret"],
    )
    ax.set_xlim(0, regret.shape[1] - 1)
    ax.grid(alpha=0.25, color=palette.grid)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 5. how the two observers relate
# --------------------------------------------------------------------------- #
def _new_ax(
    ax: plt.Axes | None, figsize: tuple[float, float]
) -> tuple[Figure, plt.Axes, bool]:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        return fig, ax, True
    return ax.get_figure(), ax, False


def plot_relative_accuracy(
    traces: Mapping[str, BeliefTrace] | Iterable[BeliefTrace],
    *,
    metric: str = "accuracy",
    n_markers: int = 4,
    ax: plt.Axes | None = None,
    palette: Palette = PALETTE,
    figsize: tuple[float, float] = (4.4, 4.2),
) -> Figure:
    """Joint against naive, as a trajectory through time rather than two curves.

    Each point is one timestep: naive performance on x, joint on y. The diagonal
    is "factorizing costs nothing", so vertical distance above it *is* the cost,
    and the shape of the path says whether that cost grows or closes as evidence
    arrives. The annotation is the final-step ratio.
    """
    joint, naive = _observer_pair(traces)
    x, y = getattr(naive, metric).mean(0), getattr(joint, metric).mean(0)
    fig, ax, _ = _new_ax(ax, figsize)

    hi = float(max(x.max(), y.max())) * 1.08
    ax.plot([0, hi], [0, hi], color=palette.truth, lw=1.0, ls="-", alpha=0.4, zorder=1)
    ax.plot(x, y, color=palette.regret, lw=2.4, zorder=2)

    picks = np.linspace(0, len(x) - 1, min(n_markers, len(x)), dtype=int)
    ax.scatter(x[picks], y[picks], s=45, facecolors=palette.regret,
               edgecolors="white", linewidths=1.2, zorder=3)
    ax.annotate("start", (x[0], y[0]), xytext=(8, -14),
                textcoords="offset points", fontsize=9)
    if x[-1] > 0:
        ax.annotate(f"×{y[-1] / x[-1]:.1f}", (x[-1], y[-1]), xytext=(-30, 10),
                    textcoords="offset points", fontsize=11, color=palette.regret)

    label = METRIC_LABELS.get(metric, metric)
    label_axes(ax, xlabel=f"naive {label}", ylabel=f"joint {label}",
               title="relative performance")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.grid(alpha=0.25, color=palette.grid)
    return fig


def _pair_rank(i: int, j: int, n_contexts: int) -> int:
    """Index of the ordered pair ``(i, j)`` within ``EpisodeBatch.interactions``.

    ``joint_likelihood`` stores two strengths per unordered pair, in the order
    ``(0,1), (1,0), (0,2), (2,0), ...`` — so the forward direction sits at twice
    the pair's lexicographic rank and the reverse one just after it.
    """
    lo, hi = min(i, j), max(i, j)
    rank = _context_pairs(n_contexts).index((lo, hi))
    return 2 * rank + (0 if i < j else 1)


#: Interaction strengths get their own colour throughout the phase figure, so
#: that variable colours can mean "which variable owns this vector" and nothing
#: else.
PHASE_COLOR = "#7b52ab"


def _standard_waveform(cfg, n_samples: int = 2001) -> np.ndarray:
    """The one potential waveform every variable is a phase-shifted copy of.

    Recovered by reading realization 0 as the interaction strength sweeps a full
    turn, which is the same thing as evaluating the waveform at every phase.
    """
    turn = np.linspace(0.0, 1.0, n_samples)[:-1]
    return gen.realization_potential(turn, gen.value_profile(cfg), cfg.n_roll)[:, 0]


def _wave_at(cfg, waveform: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Evaluate the standard waveform at a continuous circular ``position``."""
    turn = np.linspace(0.0, 1.0, waveform.size + 1)[:-1]
    return np.interp(np.mod(position, cfg.n_roll) / cfg.n_roll, turn, waveform,
                     period=1.0)


def _draw_angle(ax, cosine: float, baseline: float, colours: tuple[str, str],
                tips: tuple[str, str], label: str) -> None:
    """Draw one key/query pair as arrows on a unit circle, ``arccos(cosine)`` apart.

    The two vectors span a plane and are drawn *in that plane*, so the angle
    between the arrows is their true angle in the full embedding space rather
    than a projection of it. Only the absolute orientation is arbitrary, which is
    why each pair gets its own ``baseline``.
    """
    theta = np.arccos(np.clip(cosine, -1.0, 1.0))
    arrows = ((baseline, colours[0], tips[0]), (baseline + theta, colours[1], tips[1]))
    for angle, colour, tip in arrows:
        ax.annotate("", xy=(np.cos(angle), np.sin(angle)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=colour, lw=2.2,
                                    shrinkA=0, shrinkB=0))
        ax.text(1.2 * np.cos(angle), 1.2 * np.sin(angle), tip, color=colour,
                ha="center", va="center", fontsize=11)
    arc = np.linspace(baseline, baseline + theta, 60)
    ax.plot(0.45 * np.cos(arc), 0.45 * np.sin(arc), color=PHASE_COLOR, lw=1.4)
    mid = baseline + theta / 2
    ax.text(0.72 * np.cos(mid), 0.72 * np.sin(mid), label, color=PHASE_COLOR,
            ha="center", va="center", fontsize=10)


def plot_interaction_phases(
    world: Any,
    batch: EpisodeBatch,
    episode: int = 0,
    channels: Sequence[int] = (0, 1),
    pair: tuple[int, int] | None = None,
    *,
    fig: Figure | None = None,
    palette: Palette = PALETTE,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Interactions as **phase shifts of one standard likelihood**.

    Every rate table in the environment is the *same* pattern. A pair of latent
    variables only chooses which part of it you see: the angle between one
    variable's key and the other's query sets a phase, and the phase translates
    the pattern. The shape never changes.

    One row per observation channel, reading left to right:

    1. **Embeddings.** The key and query whose angle matters. Both are unit
       vectors, so the cosine of that angle *is* the interaction strength ``z``.
       Arrow colour says which variable owns the vector; ``z`` is drawn in its
       own colour because it belongs to neither. Both directions appear, and
       they differ — that asymmetry is what stops the joint from being symmetric.
    2. **Phase.** One waveform, read from two starting points. Both variables
       sample the same curve; their strengths only say where to start.
    3. **The standard pattern**, with the window this channel selects. Each pair
       of box edges takes the colour of the variable whose phase range it spans.
       The pattern is periodic, so it is drawn over two turns and the window
       stays a single box — one that runs off an edge continues on the next copy.
    4. **The rate table** that window yields.

    Colour means exactly one thing here: **which variable** an element belongs
    to. Interaction strengths belong to neither, so they get their own colour.
    Nothing marks which realization actually occurred — this figure is about how
    the world is built, not about how one episode turned out.

    Comparing the rows is the point: same waveform, same pattern, different
    angles, different window, different table. Two channels are drawn rather than
    all of them because the embeddings are orthogonalized across channels, so two
    is enough to show the phases move independently.

    This describes the **built-in** likelihood. Passing your own ``likelihood=``
    to :class:`~coggrid.World` replaces the mechanism, and only the last column
    stays meaningful.

    Parameters
    ----------
    world:
        The :class:`~coggrid.World` the batch came from. Needed because the
        embeddings live there rather than on the batch.
    batch:
        A batch drawn from ``world``.
    episode:
        Which episode to illustrate.
    channels:
        Which observation channels to show, one row each.
    pair:
        Which two active variables. Defaults to the first pair involving the goal.
    """
    cfg = batch.cfg
    if cfg.n_contexts < 2:
        raise ValueError(
            "phase modulation needs two active variables to interact; "
            "n_contexts=1 has a single self-interaction and no pair to show"
        )
    channels = [int(c) for c in channels]
    if any(not 0 <= c < cfg.n_observations for c in channels):
        raise ValueError(
            f"channels must lie in [0, {cfg.n_observations}), got {channels}"
        )

    goal = int(batch.goal_ind[episode])
    if pair is None:
        pair = display_pairs(cfg.n_contexts, goal, max_pairs=1)[0]
    i, j = pair
    var_i, var_j = (int(batch.ctx_inds[episode, c]) for c in (i, j))
    colour_i, colour_j = palette.context(i, goal), palette.context(j, goal)

    n_roll, n_r = cfg.n_roll, cfg.n_realizations
    waveform = _standard_waveform(cfg)
    realizations = np.arange(n_r)

    if figsize is None:
        figsize = (15.0, 3.9 * len(channels))
    if fig is None:
        fig = plt.figure(figsize=figsize, layout="constrained")
    axes = fig.subplots(len(channels), 4, squeeze=False)

    # The standard pattern is identical in every row — that is the whole claim.
    # Drawn over two turns so a window that crosses the wrap stays one box.
    fine = np.arange(n_roll * 4) / 4.0
    tile = gen.sigmoid(np.outer(_wave_at(cfg, waveform, fine),
                                _wave_at(cfg, waveform, fine)))
    pattern = np.tile(tile, (2, 2))

    for row, channel in enumerate(channels):
        # Read from the batch, so the panel cannot drift from the strengths the
        # environment actually used.
        z_ij = float(batch.interactions[episode, channel,
                                        _pair_rank(i, j, cfg.n_contexts)])
        z_ji = float(batch.interactions[episode, channel,
                                        _pair_rank(j, i, cfg.n_contexts)])

        # ── 1: the embeddings, and the angle between them
        ax = axes[row][0]
        circle = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(circle), np.sin(circle), color=palette.grid, lw=1.0, ls=":")
        _draw_angle(ax, z_ij, np.pi / 2, (colour_i, colour_j),
                    (f"$K_{i}$", f"$Q_{j}$"), f"$z_{{{i}{j}}}$={z_ij:+.2f}")
        _draw_angle(ax, z_ji, -np.pi / 2, (colour_j, colour_i),
                    (f"$K_{j}$", f"$Q_{i}$"), f"$z_{{{j}{i}}}$={z_ji:+.2f}")
        ax.set_xlim(-1.55, 1.55), ax.set_ylim(-1.55, 1.55)
        ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"channel {channel}: embeddings", fontsize=10)
        ax.set_xlabel(r"$z = \cos\theta$  (unit vectors)", fontsize=8, color="0.4")

        # ── 2: one waveform, two starting phases
        ax = axes[row][1]
        turn = np.arange(n_roll * 8) / 8.0
        ax.plot(turn, _wave_at(cfg, waveform, turn), color="0.4", lw=1.4,
                label="standard waveform")
        for c, other, z, colour, marker in ((i, j, z_ij, colour_i, "o"),
                                            (j, i, z_ji, colour_j, "s")):
            pos = z * n_roll - realizations
            ax.plot(np.mod(pos, n_roll), _wave_at(cfg, waveform, pos), marker,
                    ms=6, color=colour, label=f"var {c}  (phase $z_{{{c}{other}}}$)")
        ax.set_title("one waveform, two phases", fontsize=10)
        ax.set_xlabel("phase (slots)", fontsize=9)
        ax.set_ylabel("potential", fontsize=9)
        ax.grid(alpha=0.25, color=palette.grid)
        ax.legend(fontsize=8, frameon=False, loc="lower right")

        # ── 3: which window of the standard pattern this channel picks
        ax = axes[row][2]
        ax.imshow(pattern.T, origin="lower", cmap="magma", vmin=0.0, vmax=1.0,
                  extent=(0, 2 * n_roll, 0, 2 * n_roll))
        lo_i = np.mod(z_ij * n_roll - (n_r - 1), n_roll)
        lo_j = np.mod(z_ji * n_roll - (n_r - 1), n_roll)
        span = n_r - 1
        # Each pair of edges is drawn in the colour of the variable whose phase
        # range it spans, so the box says which axis is which.
        for offset in (0, span):
            ax.plot([lo_i, lo_i + span], [lo_j + offset] * 2, color=colour_i, lw=2.4)
            ax.plot([lo_i + offset] * 2, [lo_j, lo_j + span], color=colour_j, lw=2.4)
        for edge in (n_roll,):  # where the pattern starts repeating
            ax.axvline(edge, color="white", lw=0.8, alpha=0.45)
            ax.axhline(edge, color="white", lw=0.8, alpha=0.45)
        ax.set_title("the standard pattern (2 turns)", fontsize=10)
        ax.set_xlabel(f"phase of var {i}", fontsize=9)
        ax.set_ylabel(f"phase of var {j}", fontsize=9)

        # ── 4: the rate table an observer sees
        ax = axes[row][3]
        table = _pair_slice(batch.rates[episode, channel], i, j, cfg.n_contexts)
        ax.imshow(table.T, origin="lower", cmap="magma", vmin=0.0, vmax=1.0)
        ax.set_title("the rate table", fontsize=10)
        ax.set_xlabel(f"var {i} realization", fontsize=9)
        ax.set_ylabel(f"var {j} realization", fontsize=9)

    fig.suptitle(
        f"one standard likelihood, translated by the interaction angles  —  "
        f"vars {var_i} and {var_j}, rates dark 0 to light 1",
        fontsize=12,
    )
    return fig


def plot_evidence_likelihood(
    batch: EpisodeBatch,
    episode: int = 0,
    pair: tuple[int, int] | None = None,
    *,
    max_vectors: int = 32,
    fig: Figure | None = None,
    palette: Palette = PALETTE,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """What a *single* observation vector says about the joint realization.

    The top row is the per-channel rate tables — the building blocks. Because
    key and query embeddings are orthonormalized across channels, each one
    carves the realization plane differently, and none of them is a shifted copy
    of another.

    The grid below is the payoff: one panel per possible observation *vector*,
    showing the posterior a single sample induces from a uniform prior. It is the
    product of the channel tables, taking each channel's rate where that bit is
    1 and its complement where the bit is 0 — so ``n_observations`` binary
    channels generate ``2 ** n_observations`` distinct ways of slicing the plane,
    most of them far sharper than any channel alone.

    That richness is the reason the task is solvable at all, and the reason
    factorizing hurts: the informative structure lives in *combinations* of
    channels evaluated jointly across variables, which is exactly what a
    per-variable belief cannot hold.

    Each panel is normalized to its own maximum, so the shape is visible rather
    than the absolute likelihood — which falls off geometrically with the number
    of channels and would otherwise render every panel uniformly dark.

    Parameters
    ----------
    batch:
        Any :class:`~coggrid.world.EpisodeBatch`.
    episode:
        Which episode's rate tables to use.
    pair:
        Which two active variables. Defaults to the first pair involving the goal.
    max_vectors:
        Cap on the number of observation vectors drawn. With more than this many
        possible, an evenly spaced subset is shown.
    """
    from ..observers import RATE_FLOOR

    cfg = batch.cfg
    goal = int(batch.goal_ind[episode])
    if pair is None:
        pairs = display_pairs(cfg.n_contexts, goal, max_pairs=1)
        pair = pairs[0] if pairs else (0, 0)
    i, j = pair

    tables = np.stack([
        np.atleast_2d(_pair_slice(batch.rates[episode, c], i, j, cfg.n_contexts))
        for c in range(cfg.n_observations)
    ])
    tables = np.clip(tables, RATE_FLOOR, 1.0 - RATE_FLOOR)

    n_possible = 2**cfg.n_observations
    chosen = (
        np.arange(n_possible)
        if n_possible <= max_vectors
        else np.unique(np.linspace(0, n_possible - 1, max_vectors).astype(int))
    )
    bits = ((chosen[:, None] >> np.arange(cfg.n_observations)[None, :]) & 1).astype(float)

    # One normalized likelihood surface per chosen vector.
    log_like = np.einsum("vo,oxy->vxy", bits, np.log(tables)) + np.einsum(
        "vo,oxy->vxy", 1.0 - bits, np.log1p(-tables)
    )
    surfaces = np.exp(log_like - log_like.max(axis=(1, 2), keepdims=True))

    n_cols = min(8, len(chosen))
    n_rows = int(np.ceil(len(chosen) / n_cols))
    if figsize is None:
        figsize = (1.55 * max(n_cols, cfg.n_observations), 1.75 + 1.55 * n_rows)
    if fig is None:
        fig = plt.figure(figsize=figsize, layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.25, 1.05 * n_rows])

    channel_axes = top.subplots(1, cfg.n_observations, squeeze=False)[0]
    for c, ax in enumerate(channel_axes):
        ax.imshow(tables[c].T, origin="lower", cmap="magma", vmin=0.0, vmax=1.0)
        ax.set_title(f"channel {c}", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
    top.suptitle(
        f"per-channel rates  P(o = 1 | var {i}, var {j})  —  dark 0 to light 1",
        fontsize=10,
    )

    grid_axes = bottom.subplots(n_rows, n_cols, squeeze=False)
    for slot, ax in enumerate(grid_axes.ravel()):
        if slot >= len(chosen):
            ax.set_visible(False)
            continue
        ax.imshow(surfaces[slot].T, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title("".join(str(int(b)) for b in bits[slot]), fontsize=8,
                     family="monospace")
        ax.set_xticks([]), ax.set_yticks([])
    shown = "all" if n_possible <= max_vectors else f"{len(chosen)} of {n_possible}"
    bottom.suptitle(
        f"posterior from one observation vector ({shown}) — each panel "
        "normalized to its own peak",
        fontsize=10,
    )
    return fig


def _safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson ``r``, or ``nan`` when either side is constant.

    A small batch can easily land every episode on the same accuracy, which makes
    the correlation genuinely undefined. ``np.corrcoef`` divides by a zero
    standard deviation there and warns; returning ``nan`` quietly lets the caller
    decide what to show.
    """
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def plot_regret_vs_accuracy(
    batch: EpisodeBatch,
    traces: Mapping[str, BeliefTrace] | Iterable[BeliefTrace],
    *,
    metric: str = "accuracy",
    n_bins: int = 12,
    min_per_bin: int = 5,
    ax: plt.Axes | None = None,
    palette: Palette = PALETTE,
    figsize: tuple[float, float] = (5.0, 4.2),
) -> Figure:
    """Does factorization regret predict who gets the episode right?

    Episodes are sorted into quantile bins by their final-step
    :func:`~coggrid.factorization_regret`, and each observer's mean performance
    is plotted per bin with a standard error band. The legend carries the
    Pearson correlation over the raw per-episode values.

    The expected result is the point of the environment: regret is strongly
    *negatively* correlated with the naive observer's performance and close to
    uncorrelated with the joint observer's. Regret is not a difficulty measure —
    it is specifically the part of the difficulty that factorizing creates.
    """
    joint, naive = _observer_pair(traces)
    regret = factorization_regret(joint, naive)[:, -1]
    fig, ax, _ = _new_ax(ax, figsize)

    edges = np.unique(np.quantile(regret, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 3:
        raise ValueError(
            "factorization regret is near-constant across episodes, so there is "
            "nothing to bin against. This is exact rather than approximate when "
            "n_contexts == 1: with one active variable the two observers "
            "coincide and the regret is identically zero."
        )
    which = np.clip(np.digitize(regret, edges[1:-1]), 0, edges.size - 2)
    centers = 0.5 * (edges[:-1] + edges[1:])

    for trace in (joint, naive):
        values = getattr(trace, metric)[:, -1]
        keep = [k for k in range(edges.size - 1) if (which == k).sum() >= min_per_bin]
        mean = np.array([values[which == k].mean() for k in keep])
        sem = np.array([
            values[which == k].std() / np.sqrt((which == k).sum()) for k in keep
        ])
        color = palette.for_observer(trace.name)
        r = _safe_correlation(regret, values)
        ax.fill_between(centers[keep], mean - sem, mean + sem,
                        color=color, alpha=0.2, lw=0)
        label = trace.name if np.isnan(r) else f"{trace.name} (r={r:+.2f})"
        ax.plot(centers[keep], mean, color=color, lw=2.2, marker="o", ms=4,
                label=label)

    label_axes(ax, xlabel="factorization regret (nats)",
               ylabel=METRIC_LABELS.get(metric, metric),
               title="regret vs performance")
    ax.grid(alpha=0.25, color=palette.grid)
    ax.legend(fontsize=9)
    return fig


def plot_map_agreement(
    batch: EpisodeBatch,
    traces: Mapping[str, BeliefTrace] | Iterable[BeliefTrace],
    *,
    ax: plt.Axes | None = None,
    palette: Palette = PALETTE,
    figsize: tuple[float, float] = (4.4, 4.2),
) -> Figure:
    """Where the two observers' final answers diverge, not just how often.

    Both axes are *signed error* against the truth, so the origin is "both
    correct" and the diagonal is "both made the same mistake". Mass in the
    column above the origin is episodes the joint observer got right and the
    naive one did not — the population the whole environment is built around.
    """
    joint, naive = _observer_pair(traces)
    fig, ax, _ = _new_ax(ax, figsize)

    n_r = batch.cfg.n_realizations
    half = n_r // 2
    truth = batch.goal_value

    def signed_error(trace: BeliefTrace) -> np.ndarray:
        guess = trace.goal_belief[:, -1].argmax(-1)
        return (guess - truth + half) % n_r - half

    err_j, err_n = signed_error(joint), signed_error(naive)
    edges = np.arange(-half - 0.5, n_r - half + 0.5)
    counts, _, _ = np.histogram2d(err_j, err_n, bins=[edges, edges])

    # Counts span orders of magnitude, so log-scale them — but a log norm masks
    # empty bins, and the default "bad" colour is white, which would read as the
    # *high* end. Paint them the colormap's zero instead.
    cmap = plt.get_cmap("magma")
    cmap = cmap.with_extremes(bad=cmap(0.0))
    extent = (edges[0], edges[-1], edges[0], edges[-1])
    ax.imshow(counts.T, origin="lower", extent=extent, aspect="auto",
              cmap=cmap, norm="log" if counts.max() > 20 else None)
    ax.plot([edges[0], edges[-1]], [edges[0], edges[-1]],
            color="white", ls="--", lw=1.0, alpha=0.5)
    ax.axhline(0, color=palette.goal, lw=1.0, alpha=0.7)
    ax.axvline(0, color=palette.goal, lw=1.0, alpha=0.7)

    agree = float((err_j == err_n).mean())
    label_axes(ax, xlabel="joint error", ylabel="naive error",
               title=f"final answers (agree {agree:.0%})")
    return fig


def plot_belief_profile(
    batch: EpisodeBatch,
    traces: Mapping[str, BeliefTrace] | Iterable[BeliefTrace],
    *,
    ax: plt.Axes | None = None,
    palette: Palette = PALETTE,
    figsize: tuple[float, float] = (4.6, 4.0),
) -> Figure:
    """Mean final posterior, with every episode aligned so the truth is at zero.

    Accuracy only reports whether the mode landed on the truth. This reports the
    *shape* around it: a lower peak with fatter shoulders is an observer that is
    merely less certain, while mass displaced to one side is one that has
    converged somewhere wrong.
    """
    items = list(traces.values()) if isinstance(traces, Mapping) else list(traces)
    fig, ax, _ = _new_ax(ax, figsize)

    n_r = batch.cfg.n_realizations
    offsets = np.arange(n_r) - n_r // 2
    for trace in items:
        centered = _center_on_truth(trace.goal_belief[:, -1], batch.goal_value, n_r)
        color = palette.for_observer(trace.name)
        ax.plot(offsets, centered.mean(0), color=color, lw=2.4,
                marker="o", ms=4, label=trace.name)

    ax.axhline(1.0 / n_r, color=palette.truth, ls="--", lw=1.0, alpha=0.6)
    ax.axvline(0, color=palette.goal, lw=1.2, alpha=0.7)
    ax.text(offsets[-1], 1.0 / n_r, " chance", va="bottom", ha="right", fontsize=8)
    label_axes(ax, xlabel="offset from true value", ylabel="posterior mass",
               title="final belief, aligned on truth")
    ax.grid(alpha=0.25, color=palette.grid)
    ax.legend(fontsize=9)
    return fig


def plot_confidence_density(
    trace: BeliefTrace,
    *,
    bins: int = 50,
    n_realizations: int | None = None,
    ax: plt.Axes | None = None,
    palette: Palette = PALETTE,
    figsize: tuple[float, float] = (4.6, 4.0),
) -> Figure:
    """Distribution of ``p_correct`` across episodes, one curve per timestep.

    Curves run from the joint blue at the first step to the regret red at the
    last. A mean curve rising smoothly can hide a distribution that is really
    splitting in two — episodes the observer solves outright and episodes it
    never resolves — and that split is visible here and nowhere else.
    """
    fig, ax, _ = _new_ax(ax, figsize)
    edges = np.linspace(0.0, 1.0, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_steps = trace.p_correct.shape[1]
    colors = LinearSegmentedColormap.from_list(
        "cg_time", [palette.joint, palette.regret]
    )(np.linspace(0, 1, max(n_steps, 2)))

    for t in range(n_steps):
        density, _ = np.histogram(trace.p_correct[:, t], bins=edges, density=True)
        ax.plot(centers, density, color=colors[t], lw=1.2, alpha=0.9)

    if n_realizations:
        ax.axvline(1.0 / n_realizations, color=palette.truth, ls="--", lw=1.0,
                   alpha=0.6)
    label_axes(ax, xlabel="P(true value)", ylabel="density",
               title=f"{trace.name}: spread over episodes")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.25, color=palette.grid)
    return fig


def plot_regret_analysis(
    batch: EpisodeBatch,
    traces: Mapping[str, BeliefTrace],
    *,
    palette: Palette = PALETTE,
    figsize: tuple[float, float] = (13.5, 4.2),
) -> Figure:
    """The three views of what factorizing actually costs, side by side."""
    fig, axes = plt.subplots(1, 3, figsize=figsize, layout="constrained")
    plot_relative_accuracy(traces, ax=axes[0], palette=palette)
    plot_regret_vs_accuracy(batch, traces, ax=axes[1], palette=palette)
    plot_map_agreement(batch, traces, ax=axes[2], palette=palette)
    fig.suptitle(f"cost of factorizing ({len(batch)} episodes)", fontsize=12)
    return fig


def plot_belief_shape(
    batch: EpisodeBatch,
    traces: Mapping[str, BeliefTrace],
    *,
    palette: Palette = PALETTE,
    figsize: tuple[float, float] = (13.5, 4.2),
) -> Figure:
    """What the posteriors look like, beyond whether their mode is correct."""
    joint, naive = _observer_pair(traces)
    fig, axes = plt.subplots(1, 3, figsize=figsize, layout="constrained")
    plot_belief_profile(batch, traces, ax=axes[0], palette=palette)
    for ax, trace in zip(axes[1:], (joint, naive), strict=True):
        plot_confidence_density(
            trace, ax=ax, palette=palette,
            n_realizations=batch.cfg.n_realizations,
        )
    fig.suptitle(
        f"belief shape ({len(batch)} episodes; density curves run first step → last)",
        fontsize=12,
    )
    return fig


# --------------------------------------------------------------------------- #
# 6. everything at once
# --------------------------------------------------------------------------- #
def summary_figure(
    batch: EpisodeBatch,
    traces: Mapping[str, BeliefTrace],
    *,
    episode: int = 0,
    palette: Palette = PALETTE,
) -> list[Figure]:
    """Build the standard diagnostic set, from one episode out to the whole batch.

    Four figures, narrowing from *what happened* to *what it cost*:

    1. :func:`plot_episode` — one episode's rate surface beside its belief traces.
    2. :func:`plot_performance` — batch-averaged learning curves, ending in regret.
    3. :func:`plot_regret_analysis` — whether regret explains the joint/naive gap.
    4. :func:`plot_belief_shape` — the posteriors behind those accuracy numbers.

    The last two contrast the observers against each other, so they are omitted
    when ``n_contexts == 1``: there is no interaction to factorize away, the two
    observers coincide exactly, and both figures would be degenerate.

    Returns the figures rather than showing them, so a caller can save them all:

    >>> for i, fig in enumerate(summary_figure(batch, traces)):  # doctest: +SKIP
    ...     fig.savefig(f"figure_{i}.png", dpi=150)
    """
    figures = [
        plot_episode(batch, traces, episode=episode, palette=palette),
        plot_performance(traces, palette=palette),
    ]
    if batch.cfg.n_contexts > 1:
        figures += [
            plot_regret_analysis(batch, traces, palette=palette),
            plot_belief_shape(batch, traces, palette=palette),
        ]
    return figures
