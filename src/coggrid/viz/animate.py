"""Play a single episode back, the way ``env.render()`` shows you a gym task.

The entry point is :func:`animate_episode`. In a notebook it plays on its own —
make it the last expression in a cell and nothing else is needed::

    animate_episode(batch, traces)

Elsewhere, :meth:`EpisodeAnimation.save` writes a GIF and
:attr:`EpisodeAnimation.anim` is the underlying ``FuncAnimation``.

Unlike :mod:`coggrid.viz.plots`, whose functions return a ``Figure``, everything
here is time-varying: a **panel** is a function that draws itself once and hands
back an updater called with the timestep.

.. code-block:: python

    def my_panel(ax, view):
        (line,) = ax.plot([], [])
        p_correct = view.traces["joint"].p_correct[view.episode]
        def update(t):
            line.set_data(range(t + 1), p_correct[: t + 1])
        return update

    # panels are laid out in rows: grids on top, things that evolve below
    animate_episode(batch, traces, panels=[GRID_PANELS, [*TRACE_PANELS, my_panel]])

``view`` is an :class:`EpisodeView` carrying the batch, the traces, the episode
index and the palette, plus the derived arrays panels tend to want.
"""

from __future__ import annotations

import io
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .. import generative as gen
from ..observers import BeliefTrace, disentanglement, factorization_regret
from ..world import EpisodeBatch
from .plots import (
    MAX_PAIRS,
    PHASE_COLOR,
    _draw_phase_columns,
    _phase_pattern,
    _standard_waveform,
    _wave_at,
    display_pairs,
)
from .style import PALETTE, Palette, label_axes

__all__ = [
    "EpisodeView",
    "EpisodeAnimation",
    "animate_episode",
    "animate_interaction_phases",
    "observations_panel",
    "joint_posterior_panel",
    "naive_posterior_panel",
    "difference_panel",
    "marginal_beliefs_panel",
    "regret_panel",
    "regret_rate_panel",
    "default_panels",
    "GRID_PANELS",
    "TRACE_PANELS",
    "DEFAULT_PANELS",
]

#: A panel draws its static furniture into ``ax`` and returns an updater that is
#: called with the timestep for every frame.
Panel = Callable[[Axes, "EpisodeView"], Callable[[int], Any]]

#: Height of one row of panels, in inches. Grids are square-ish and want the
#: room; time series are wide and short, so later rows get less.
GRID_ROW_HEIGHT = 4.2
TRACE_ROW_HEIGHT = 2.6
PANEL_WIDTH = 4.2


@dataclass(frozen=True)
class EpisodeView:
    """One episode, plus the derived arrays panels keep needing.

    Everything is already sliced down to a single episode, so a panel never has
    to remember which axis is the batch.
    """

    batch: EpisodeBatch
    traces: Mapping[str, BeliefTrace]
    episode: int = 0
    palette: Palette = PALETTE
    max_pairs: int = MAX_PAIRS

    @property
    def pairs(self) -> list[tuple[int, int]]:
        """Variable pairs worth drawing, goal-involving first. Empty when C == 1."""
        return display_pairs(self.cfg.n_contexts, self.goal_context, self.max_pairs)

    @property
    def primary_pair(self) -> tuple[int, int] | None:
        """The pair the grid panels draw, or ``None`` with a single variable."""
        return self.pairs[0] if self.pairs else None

    @property
    def cfg(self):
        return self.batch.cfg

    @property
    def n_steps(self) -> int:
        return self.batch.n_steps

    @property
    def truth(self) -> np.ndarray:
        """``(n_contexts,)`` realization each active variable actually took."""
        return self.batch.ctx_vals[self.episode]

    @property
    def goal_context(self) -> int:
        """Which column of the active variables the agent is scored on."""
        return int(self.batch.goal_ind[self.episode])

    def variable_label(self, c: int) -> str:
        """Axis label naming the variable and its role this episode."""
        role = "goal" if c == self.goal_context else "context"
        return f"var {c} (idx {self.batch.ctx_inds[self.episode, c]}) — {role}"

    def variable_color(self, c: int) -> str:
        """Green for the goal variable, a shade of orange for a context one.

        With three or more variables a single orange would draw two or more
        indistinguishable lines, so the context variables are spread across an
        orange ramp — still obviously "not the goal", but telling apart.
        """
        if c == self.goal_context:
            return self.palette.goal
        others = [k for k in range(self.cfg.n_contexts) if k != self.goal_context]
        if len(others) < 2:
            return self.palette.other
        ramp = LinearSegmentedColormap.from_list(
            "cg_context", ["#a8501a", self.palette.other, "#f7cd93"]
        )
        return to_hex(ramp(others.index(c) / (len(others) - 1)))

    @property
    def observations(self) -> np.ndarray:
        """``(n_steps, n_observations)`` boolean evidence stream."""
        return self.batch.observations[self.episode]

    @cached_property
    def _joint_posterior(self) -> np.ndarray:
        """``(n_steps, R, ..., R)`` normalized posterior over the whole grid."""
        log_post = self.traces["joint"].log_posterior[self.episode]
        axes = tuple(range(1, log_post.ndim))
        grid = np.exp(log_post - log_post.max(axis=axes, keepdims=True))
        return grid / grid.sum(axis=axes, keepdims=True)

    def joint_pair_grid(self, i: int, j: int) -> np.ndarray:
        """``(n_steps, R, R)`` joint posterior marginalized onto variables i, j.

        With more than two active variables the full posterior is
        ``R ** n_contexts`` wide and cannot be shown, so a pair is projected out
        of it. With exactly two this is the posterior itself.
        """
        drop = tuple(1 + k for k in range(self.cfg.n_contexts) if k not in (i, j))
        grid = self._joint_posterior.sum(axis=drop) if drop else self._joint_posterior
        return grid if i < j else grid.transpose(0, 2, 1)

    def naive_pair_grid(self, i: int, j: int) -> np.ndarray:
        """``(n_steps, R, R)`` joint over i, j that the naive observer *implies*.

        It holds one belief per variable, so the joint it stands behind is their
        outer product — which is exactly the assumption under test.
        """
        belief = self.traces["naive"].belief[self.episode]
        return belief[:, i, :, None] * belief[:, j, None, :]

    def _pair(self) -> tuple[int, int]:
        if self.primary_pair is None:
            raise ValueError(
                "grid panels need at least two active variables; with "
                "n_contexts=1 use marginal_beliefs_panel instead"
            )
        return self.primary_pair

    def marginal_belief(self, name: str) -> np.ndarray:
        """``(n_steps, n_contexts, n_realizations)`` one observer's marginals."""
        return self.traces[name].belief[self.episode]

    @cached_property
    def goal_regret(self) -> np.ndarray:
        """``(n_steps,)`` factorization regret on the goal variable.

        ``D_KL(B_joint || B_naive)`` over the goal variable (arXiv:2603.27134
        §3.1). An accumulating measure, so it compares whole trajectories rather
        than resolving individual steps — which is what :attr:`disentanglement`
        is for.
        """
        return factorization_regret(
            self.traces["joint"], self.traces["naive"]
        )[self.episode]

    @cached_property
    def disentanglement(self) -> np.ndarray:
        """``(n_steps, n_contexts)`` how non-Markovian each step's update is.

        This episode's slice of :func:`~coggrid.observers.disentanglement`
        (arXiv:2603.27134 §B.3).
        """
        return disentanglement(
            self.traces["joint"], self.traces["naive"], self.batch, per_variable=True
        )[self.episode]

    @cached_property
    def cumulative_disentanglement(self) -> np.ndarray:
        """``(n_steps, n_contexts)`` running total of :attr:`disentanglement`."""
        return np.cumsum(self.disentanglement, axis=0)

    @cached_property
    def joint_grid(self) -> np.ndarray:
        """:meth:`joint_pair_grid` for :attr:`primary_pair`."""
        return self.joint_pair_grid(*self._pair())

    @cached_property
    def naive_grid(self) -> np.ndarray:
        """:meth:`naive_pair_grid` for :attr:`primary_pair`."""
        return self.naive_pair_grid(*self._pair())

    @cached_property
    def difference_grid(self) -> np.ndarray:
        """``joint_grid - naive_grid``: the mass factorizing puts in the wrong place."""
        return self.joint_grid - self.naive_grid

    @cached_property
    def grid_vmax(self) -> float:
        """Shared colour ceiling, so panels stay comparable and do not flicker."""
        return float(max(self.joint_grid.max(), self.naive_grid.max()))

    @cached_property
    def difference_vmax(self) -> float:
        """Symmetric limit for the difference panel, so zero stays neutral."""
        return float(np.abs(self.difference_grid).max()) or 1e-12

    @staticmethod
    def modes(grid: np.ndarray) -> np.ndarray:
        """``(n_steps, 2)`` argmax cell of each frame of a ``(T, R, R)`` grid."""
        flat = grid.reshape(len(grid), -1).argmax(axis=1)
        return np.stack(np.unravel_index(flat, grid.shape[1:]), axis=-1)


# --------------------------------------------------------------------------- #
# built-in panels
# --------------------------------------------------------------------------- #
def _mark_truth(ax: Axes, view: EpisodeView) -> None:
    ax.add_patch(
        Rectangle(
            (view.truth[0] - 0.5, view.truth[1] - 0.5), 1, 1,
            edgecolor=view.palette.goal, facecolor="none", linewidth=2.0,
        )
    )


def _label_grid(ax: Axes, view: EpisodeView, title: str) -> None:
    label_axes(
        ax, xlabel=view.variable_label(0), ylabel=view.variable_label(1), title=title
    )
    ax.xaxis.label.set_color(view.variable_color(0))
    ax.yaxis.label.set_color(view.variable_color(1))


def _posterior_panel(which: str, title: str) -> Panel:
    def panel(ax: Axes, view: EpisodeView) -> Callable[[int], Any]:
        grid = getattr(view, which)
        modes = view.modes(grid)
        image = ax.imshow(
            np.zeros_like(grid[0]), origin="lower", vmin=0.0, vmax=view.grid_vmax,
            cmap="magma", aspect="equal",
        )
        _mark_truth(ax, view)
        # Where this observer would answer right now. Reading it against the
        # green truth box is the whole question the episode is asking.
        (mode,) = ax.plot(
            [], [], marker="o", ms=7, mfc="none", mec="white", mew=1.8, ls="none",
            label="mode",
        )
        _label_grid(ax, view, title)

        def update(t: int) -> Any:
            image.set_data(grid[t].T)
            mode.set_data([modes[t, 0]], [modes[t, 1]])
            return image

        return update

    return panel


#: The Bayes-optimal posterior over both variables at once.
joint_posterior_panel = _posterior_panel("joint_grid", "joint")

#: The same grid as a factorized observer would have to represent it.
naive_posterior_panel = _posterior_panel("naive_grid", "naive (factorized)")


def difference_panel(ax: Axes, view: EpisodeView) -> Callable[[int], Any]:
    """``joint − naive``: where factorizing moves probability mass.

    Red is mass the joint observer holds and the naive one does not; blue is
    mass the naive one invents. A separable episode stays white throughout —
    the colour appearing *is* the cost of factorizing, localized on the grid.
    """
    grid = view.difference_grid
    lim = view.difference_vmax
    image = ax.imshow(
        np.zeros_like(grid[0]), origin="lower", vmin=-lim, vmax=lim,
        cmap="RdBu_r", aspect="equal",
    )
    _mark_truth(ax, view)
    _label_grid(ax, view, "joint − naive")

    def update(t: int) -> Any:
        image.set_data(grid[t].T)
        return image

    return update


def observations_panel(ax: Axes, view: EpisodeView) -> Callable[[int], Any]:
    """The evidence stream, with everything after the current step hidden.

    Without this it is impossible to tell whether a posterior moved because the
    evidence changed or because it was still settling on evidence already seen.
    """
    obs = view.observations
    n_steps, n_obs = obs.shape
    image = ax.imshow(
        np.zeros_like(obs, dtype=float).T, aspect="auto", cmap="Greys",
        vmin=0.0, vmax=1.0, origin="lower", interpolation="nearest",
        extent=(-0.5, n_steps - 0.5, -0.5, n_obs - 0.5),
    )
    cursor = ax.axvline(-0.5, color=view.palette.goal, lw=1.8)
    ax.set_yticks(range(n_obs))
    label_axes(ax, xlabel="timestep", ylabel="channel", title="evidence so far")

    def update(t: int) -> Any:
        revealed = np.zeros_like(obs, dtype=float)
        revealed[: t + 1] = obs[: t + 1]
        image.set_data(revealed.T)
        cursor.set_xdata([t, t])
        return image

    return update


def marginal_beliefs_panel(ax: Axes, view: EpisodeView) -> Callable[[int], Any]:
    """The goal variable's marginal posterior, for both observers.

    Only the goal: it is the variable being scored, and drawing every context
    variable alongside it makes the panel unreadable as soon as there are more
    than two. Solid is the joint observer, dashed the naive one, and the dotted
    line marks the true value.
    """
    n_r = view.cfg.n_realizations
    grid = np.arange(n_r)
    goal = view.goal_context
    lines = {}
    fills = {}  # Dictionary to track the fill collections
    
    for name, style, width, alpha in (
        ("joint", "-", 2.4, 1.0), ("naive", "--", 1.8, 0.75)
    ):
        color = view.palette.for_observer(name)
        
        (line,) = ax.plot(
            [], [], ls=style, lw=width, alpha=alpha,
            color=color, label=name,
        )
        # Initialize the fill with an array of zeros to match the x-grid size.
        # Note: No comma after fill, as it returns a single PolyCollection!
        fill = ax.fill_between(x=grid, y1=0, y2=0, alpha=0.3, color=color)
        
        lines[name] = line
        fills[name] = fill

    ax.axvline(view.truth[goal], color=view.palette.goal, ls=":", lw=1.4)

    ax.set_xlim(-0.5, n_r - 0.5)
    ax.set_ylim(0.0, 1.02)
    ax.legend(fontsize=8)
    label_axes(ax, xlabel="realization", ylabel="belief",
               title=f"marginal belief — var {goal} (goal)")

    def update(t: int) -> Any:
        artists = []
        for name, line in lines.items():
            y_data = view.marginal_belief(name)[t, goal]
            
            # Update the line
            line.set_data(grid, y_data)
            artists.append(line)
            
            # Update the fill by removing the old one and drawing a new one
            fills[name].remove()
            color = view.palette.for_observer(name)
            fills[name] = ax.fill_between(x=grid, y1=0, y2=y_data, alpha=0.3, color=color)
            artists.append(fills[name])
            
        return artists

    return update

def regret_panel(ax: Axes, view: EpisodeView) -> Callable[[int], Any]:
    """Disagreement injected against disagreement realized — both accumulated.

    Two *levels*, both in nats, both for the goal variable:

    * **factorization regret** — how far the naive posterior has drifted from
      the optimal one. Realized.
    * **cumulative dis-entanglement** — the running total of per-step update
      divergence. Injected.

    The gap between them is the point. KL is not additive across sequential
    Bayesian updates, so a step's divergence turns into regret only in
    proportion to where the posterior already has mass. Neither curve bounds
    the other: injected commonly exceeds realized when successive
    disagreements cancel, and realized can exceed injected when they compound.
    """
    goal = view.goal_context
    regret = view.goal_regret
    injected = view.cumulative_disentanglement[:, goal]

    (realized_line,) = ax.plot(
        [], [], color=view.palette.regret, lw=2.4, label="regret (realized)"
    )
    (injected_line,) = ax.plot(
        [], [], color=view.palette.joint, lw=1.8, ls="--",
        label="dis-entanglement (injected)",
    )
    ax.set_xlim(0, max(view.n_steps - 1, 1))
    ax.set_ylim(0.0, float(max(regret.max(), injected.max())) * 1.1 + 1e-9)
    ax.grid(alpha=0.25, color=view.palette.grid)
    ax.legend(fontsize=7, loc="upper left")
    label_axes(ax, xlabel="timestep", ylabel="nats",
               title=f"accumulated — var {goal} (goal)")

    def update(t: int) -> Any:
        span = range(t + 1)
        realized_line.set_data(span, regret[: t + 1])
        injected_line.set_data(span, injected[: t + 1])
        return [realized_line, injected_line]

    return update


def regret_rate_panel(ax: Axes, view: EpisodeView) -> Callable[[int], Any]:
    """The same two quantities as per-step rates, where they are comparable.

    Plotting a rate against an accumulating level crushes the rate, so the
    step-wise view gets its own panel: the change in regret against the update
    divergence driving it.

    Note the asymmetry this exposes. Update divergence is a KL, so it is never
    negative; the change in regret is negative on a large fraction of steps,
    because the naive posterior drifts back toward the optimal one about as
    often as away. The update predicts how far regret *moves*, not which way —
    which is why it tracks the magnitude of the change far better than its
    signed value.
    """
    goal = view.goal_context
    delta = np.diff(view.goal_regret, prepend=0.0)
    updates = view.disentanglement[:, goal]

    (delta_line,) = ax.plot(
        [], [], color=view.palette.regret, lw=2.2, label="change in regret"
    )
    (update_line,) = ax.plot(
        [], [], color=view.palette.joint, lw=1.8, ls="--",
        label="dis-entanglement",
    )
    ax.axhline(0.0, color=view.palette.truth, lw=0.8, alpha=0.5)
    ax.set_xlim(0, max(view.n_steps - 1, 1))
    lim = float(max(np.abs(delta).max(), updates.max())) * 1.15 + 1e-9
    ax.set_ylim(-lim, lim)
    ax.grid(alpha=0.25, color=view.palette.grid)
    ax.legend(fontsize=7, loc="upper left")
    label_axes(ax, xlabel="timestep", ylabel="nats / step",
               title=f"per step — var {goal} (goal)")

    def update(t: int) -> Any:
        span = range(t + 1)
        delta_line.set_data(span, delta[: t + 1])
        update_line.set_data(span, updates[: t + 1])
        return [delta_line, update_line]

    return update


#: Panels that draw the realization grid. These go on the top row.
GRID_PANELS: tuple[Panel, ...] = (
    joint_posterior_panel,
    naive_posterior_panel,
    difference_panel,
)

#: Panels that evolve along the time axis. These go underneath.
TRACE_PANELS: tuple[Panel, ...] = (observations_panel,)

#: The two-variable default layout. :func:`default_panels` adapts it.
DEFAULT_PANELS: tuple[tuple[Panel, ...], ...] = (GRID_PANELS, TRACE_PANELS)


def default_panels(
    n_contexts: int, extended: bool = False
) -> tuple[tuple[Panel, ...], ...]:
    """Panels suited to this many active variables.

    The realization grid is inherently two-dimensional, so the layout has to
    adapt rather than assume:

    * **one variable** — there is no pair and the two observers coincide, so the
      grids are dropped entirely and the marginals carry the figure.
    * **two** — the full posterior *is* the grid; draw it.
    * **three or more** — the posterior is ``R ** n_contexts`` wide and cannot be
      drawn, so the grids show one pair projected out of it, chosen by
      :func:`~coggrid.viz.display_pairs` to involve the goal variable.

    ``extended`` adds the per-variable marginals and, where the observers differ,
    how history-dependent their updates are.
    """
    state: list[Panel] = []
    if n_contexts >= 2:
        state += list(GRID_PANELS)
    if extended or n_contexts < 2:
        state.append(marginal_beliefs_panel)

    evolving: list[Panel] = list(TRACE_PANELS)
    if extended and n_contexts >= 2:
        evolving += [regret_panel, regret_rate_panel]
    return (tuple(state), tuple(evolving))


# --------------------------------------------------------------------------- #
# the animation
# --------------------------------------------------------------------------- #
def _as_rows(panels: Sequence[Any]) -> list[list[Panel]]:
    """Accept either one flat row of panels, or a sequence of rows."""
    items = list(panels)
    if not items:
        raise ValueError("animate_episode needs at least one panel")
    if all(callable(p) for p in items):
        return [items]
    if any(callable(p) for p in items):
        raise ValueError(
            "`panels` must be either all panel callables (a single row) or all "
            "sequences of them (one per row), not a mix"
        )
    rows = [list(row) for row in items]
    for row in rows:
        if not row:
            raise ValueError("`panels` contains an empty row")
        if not all(callable(p) for p in row):
            raise ValueError("every entry of a panel row must be callable")
    return rows


class EpisodeAnimation:
    """A rendered episode that plays itself in a notebook.

    Wraps a ``FuncAnimation`` so that being the last expression in a cell is
    enough to see it — no ``to_jshtml`` and no ffmpeg. The figure is deliberately
    *not* registered with pyplot, because the inline backend would otherwise also
    emit a single still frame alongside the animation.
    """

    def __init__(
        self,
        fig: Figure,
        anim: FuncAnimation,
        fps: int,
        n_frames: int,
        draw: Callable[[int], Any] | None = None,
        colors: int | None = None,
    ) -> None:
        self.fig = fig
        self.anim = anim
        self.fps = fps
        #: Frame count, tracked here because matplotlib keeps its own private.
        self.n_frames = n_frames
        #: Per-frame draw callback, needed only by the shared-palette encoder.
        self.draw = draw
        #: Palette size for that encoder; ``None`` uses matplotlib's writer.
        self.colors = colors
        self._gif: bytes | None = None

    def save(self, path: str | Path) -> Path:
        """Write the animation to a GIF and return the path."""
        path = Path(path).with_suffix(".gif")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.to_gif())
        return path

    def to_gif(self) -> bytes:
        """Encode once and cache; a re-display should not re-render every frame."""
        if self._gif is None:
            if self.colors is not None and self.draw is not None:
                self._gif = self._shared_palette_gif()
            else:
                # matplotlib resolves the output path on disk, so an in-memory
                # buffer is not an option here.
                with tempfile.TemporaryDirectory() as tmp:
                    scratch = Path(tmp) / "episode.gif"
                    self.anim.save(scratch, writer=PillowWriter(fps=self.fps))
                    self._gif = scratch.read_bytes()
        return self._gif

    def _shared_palette_gif(self) -> bytes:
        """Encode with one palette for every frame, and no dithering.

        matplotlib's writer quantizes each frame independently and dithers,
        which makes even *unchanged* regions differ pixel by pixel between
        frames and defeats GIF's frame-to-frame compression. Pinning a single
        palette and turning dithering off keeps static areas byte-identical,
        which on a figure that is mostly still is worth several times the file
        size.
        """
        from PIL import Image

        assert self.draw is not None and self.colors is not None
        rendered = []
        for frame in range(self.n_frames):
            self.draw(frame)
            buf = io.BytesIO()
            self.fig.savefig(buf, format="png")
            buf.seek(0)
            rendered.append(Image.open(buf).convert("RGB"))

        # Build the palette from a frame partway in, so it reflects the colours
        # the animation actually spends its time showing.
        reference = rendered[len(rendered) // 3].quantize(
            colors=self.colors, dither=Image.Dither.NONE
        )
        frames = [
            im.quantize(palette=reference, dither=Image.Dither.NONE)
            for im in rendered
        ]
        # Every frame really was drawn, just not through matplotlib's writer,
        # so quiet the warning its Animation raises when collected unrendered.
        self.anim._draw_was_started = True
        out = io.BytesIO()
        frames[0].save(
            out, format="GIF", save_all=True, append_images=frames[1:],
            duration=round(1000 / self.fps), loop=0, optimize=True,
        )
        return out.getvalue()

    def to_html(self) -> str:
        """An HTML5 player with scrub and loop controls.

        Heavier than the GIF and needs the renderer to run inline JavaScript,
        but it lets you step through frame by frame.
        """
        return self.anim.to_jshtml(fps=self.fps)

    def display(self) -> bool:
        """Show this inline if we are under an IPython kernel; else do nothing.

        Being the last expression in a cell is enough on its own — this exists
        for scripts, where ``%run`` never echoes a trailing expression. Returns
        whether anything was displayed.
        """
        try:
            from IPython import get_ipython
            from IPython.display import display as ipy_display
        except ModuleNotFoundError:  # pragma: no cover - depends on environment
            return False
        shell = get_ipython()
        if shell is None or getattr(shell, "kernel", None) is None:
            return False
        ipy_display(self)
        return True

    def _repr_mimebundle_(self, include=None, exclude=None) -> dict[str, Any]:
        return {"image/gif": self.to_gif(), "text/plain": repr(self)}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"EpisodeAnimation(frames={self.n_frames}, fps={self.fps}, "
            f"panels={len(self.fig.axes)})"
        )


def animate_episode(
    batch: EpisodeBatch,
    traces: Mapping[str, BeliefTrace],
    episode: int = 0,
    *,
    panels: Sequence[Any] | None = None,
    extended: bool = False,
    fps: int = 6,
    max_pairs: int = MAX_PAIRS,
    palette: Palette = PALETTE,
    figsize: tuple[float, float] | None = None,
) -> EpisodeAnimation:
    """Play one episode back as evidence accumulates.

    Parameters
    ----------
    batch, traces:
        An :class:`~coggrid.EpisodeBatch` and the output of
        :func:`~coggrid.run_observers` over it.
    episode:
        Which episode of the batch to play.
    panels:
        Either a flat sequence of panels (one row) or a sequence of rows.
        Defaults to :func:`default_panels`, which adapts to ``n_contexts``.
    extended:
        Also show the per-variable marginals and how history-dependent the
        marginal updates are. Cannot be combined with an explicit ``panels``;
        build on ``default_panels(n_contexts, extended=True)`` instead.
    fps:
        Playback rate, also the GIF frame rate.
    symmetric:
        Whether *both* divergences — factorization regret and update
        dis-entanglement — average the two KL directions. Applies to both
        together, so the panels never mix conventions.
    baseline:
        What the joint observer's update is compared against: ``"history"``
        isolates history dependence, ``"naive"`` gives the direct joint-vs-naive
        contrast. See :class:`EpisodeView`.

    Examples
    --------
    >>> from coggrid import CogGridConfig, World, run_observers
    >>> from coggrid.viz import animate_episode
    >>> cfg = CogGridConfig(n_vars=60, n_realizations=5, n_steps=12, seed=0)
    >>> batch = World(cfg).sample_episodes(4)
    >>> clip = animate_episode(batch, run_observers(batch))
    >>> clip.n_frames
    12
    >>> len(clip.to_gif()) > 0   # renders once, then caches
    True
    """
    if panels is None:
        panels = default_panels(batch.cfg.n_contexts, extended)
    rows = _as_rows(panels)

    view = EpisodeView(
        batch=batch, traces=traces, episode=episode, palette=palette,
        max_pairs=max_pairs,
    )
    heights = [GRID_ROW_HEIGHT] + [TRACE_ROW_HEIGHT] * (len(rows) - 1)
    figsize = figsize or (PANEL_WIDTH * max(map(len, rows)), sum(heights))

    # Not plt.subplots: a pyplot-managed figure is auto-rendered by the inline
    # backend as a still frame, which would appear beside the animation.
    fig = Figure(figsize=figsize, layout="constrained")
    outer = fig.add_gridspec(len(rows), 1, height_ratios=heights)

    updaters = []
    for r, row in enumerate(rows):
        cells = outer[r].subgridspec(1, len(row))
        for c, panel in enumerate(row):
            updaters.append(panel(fig.add_subplot(cells[0, c]), view))

    split, n_steps = batch.split, view.n_steps
    shown = ""
    if view.primary_pair is not None and batch.cfg.n_contexts > 2:
        shown = f",  grids show vars {view.primary_pair[0]}x{view.primary_pair[1]}"

    def draw(t: int) -> list[Any]:
        artists = [update(t) for update in updaters]
        fig.suptitle(
            f"episode {episode} ({split} split) — step {t + 1}/{n_steps}"
            f"    green = goal / truth,  orange = context{shown}",
            fontsize=11,
        )
        return artists

    anim = FuncAnimation(
        fig, draw, frames=n_steps, interval=1000 // fps, blit=False
    )
    return EpisodeAnimation(fig, anim, fps, n_frames=n_steps)


# --------------------------------------------------------------------------- #
# the interaction phases, swept
# --------------------------------------------------------------------------- #
def _window_anchor(strengths: np.ndarray, cfg) -> float:
    """Which turn to express a swept window's position in.

    The pattern repeats every ``n_roll`` slots, so a window's position is only
    defined modulo a turn. Left at the default, a sweep that crosses a turn
    boundary makes the box jump from one edge of the panel to the other. Anchor
    it to the lowest position the sweep reaches and the box slides instead.
    """
    raw = np.asarray(strengths) * cfg.n_roll - (cfg.n_realizations - 1)
    low, width = float(raw.min()), float(raw.max() - raw.min())
    if width >= cfg.n_roll:
        return 0.0  # the sweep covers a whole turn; wrapping cannot be avoided
    room = 2 * cfg.n_roll - (cfg.n_realizations - 1)
    anchor = low
    while anchor < 0:
        anchor += cfg.n_roll
    while anchor + width > room:
        anchor -= cfg.n_roll
    return anchor if anchor >= 0 else 0.0


def _draw_embedding_circle(
    ax: Axes,
    pair: tuple[int, int],
    colours: tuple[str, str],
    angles: tuple[float, float],
    strengths: tuple[float, float],
    *,
    palette: Palette,
    emphasise_keys: bool,
) -> None:
    """The two key/query pairs on a unit circle, with the keys free to turn.

    Each query is a fixed reference direction and each key swings against it, so
    the angle you watch opening and closing *is* the interaction strength.
    """
    i, j = pair
    colour_i, colour_j = colours
    theta_i, theta_j = angles
    z_ij, z_ji = strengths
    key_style = (3.0, 1.0) if emphasise_keys else (2.2, 1.0)
    query_style = (1.8, 0.45) if emphasise_keys else (2.2, 1.0)

    circle = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(circle), np.sin(circle), color=palette.grid, lw=1.0, ls=":")
    # Keys and queries get different label radii. A key swings right up to its
    # query when z approaches 1, and at a shared radius the two labels would
    # land on the same point and overprint each other.
    for theta, colour, tip, (lw, alpha), radius in (
        (np.pi / 2, colour_j, f"$Q_{j}$", query_style, 1.45),
        (-np.pi / 2, colour_i, f"$Q_{i}$", query_style, 1.45),
        (theta_i, colour_i, f"$K_{i}$", key_style, 1.18),
        (theta_j, colour_j, f"$K_{j}$", key_style, 1.18),
    ):
        ax.annotate("", xy=(np.cos(theta), np.sin(theta)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=colour, lw=lw,
                                    alpha=alpha, shrinkA=0, shrinkB=0))
        ax.text(radius * np.cos(theta), radius * np.sin(theta), tip, color=colour,
                ha="center", va="center", fontsize=11, alpha=alpha)

    # Each arc is purple, because an angle belongs to neither variable. Its
    # value is written below in the colour of the variable whose phase it sets.
    for start, end, radius in (
        (np.pi / 2, theta_i, 0.45),
        (-np.pi / 2, theta_j, 0.72),
    ):
        span = np.linspace(start, end, 60)
        ax.plot(radius * np.cos(span), radius * np.sin(span), color=PHASE_COLOR, lw=1.4)

    # The readouts are pinned to the panel corners rather than placed along each
    # arc. As z approaches 1 the arc collapses onto its two vectors, so anything
    # following the bisector lands on their names — and that is exactly the part
    # of the sweep worth reading.
    for y, va, label, colour in (
        (0.99, "top", f"$z_{{{i}{j}}}$={z_ij:+.2f}", colour_i),
        (0.01, "bottom", f"$z_{{{j}{i}}}$={z_ji:+.2f}", colour_j),
    ):
        ax.text(0.01, y, label, transform=ax.transAxes, color=colour,
                ha="left", va=va, fontsize=10)

    ax.set_xlim(-1.95, 1.95), ax.set_ylim(-1.95, 1.95)
    ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def animate_interaction_phases(
    world: Any,
    batch: EpisodeBatch,
    episode: int = 0,
    channel: int = 0,
    pair: tuple[int, int] | None = None,
    *,
    n_frames: int = 60,
    fps: int = 6,
    swing: tuple[float, float] = (26.0, 24.0),
    centre: tuple[float, float] = (33.0, 33.0),
    colors: int = 128,
    dpi: int = 84,
    palette: Palette = PALETTE,
    figsize: tuple[float, float] = (12.5, 3.6),
) -> EpisodeAnimation:
    """Turn the keys and watch the whole likelihood follow.

    The top row is one real pair, fixed. Below it the same pair has its two
    **keys** swept while the queries hold still, so every panel to the right
    moves in step: the phase markers slide along the one standard waveform, the
    window translates across the standard pattern, and the rate table
    restructures.

    Nothing here is interpolated. Each frame's table is rebuilt from that frame's
    two strengths through the same construction the environment uses, so every
    table shown is one the world would genuinely produce for those angles.

    The keys swing rather than spin. Near a right angle the cosine is steepest,
    so a gentle rock already sweeps the phase across more than a full window —
    and small motion keeps successive frames nearly identical, which is most of
    why the GIF stays small.

    Parameters
    ----------
    world, batch, episode, channel, pair:
        As :func:`~coggrid.viz.plots.plot_interaction_phases`, except that one
        channel is shown rather than several.
    n_frames, fps:
        Frame count and playback rate. Frames drive the file size; ``fps`` is
        free, so lower it to slow the motion without paying for it.
    swing, centre:
        Amplitude and midpoint, in degrees, of each key's rock. The first key
        completes one cycle per loop and the second two, so the window traces a
        figure-eight rather than sliding along a line. The default keeps every
        swept phase inside a single turn, so nothing wraps around a panel edge
        mid-loop: the realizations occupy ``n_realizations - 1`` slots of
        ``n_roll``, which leaves the rest as room to travel, and a centre near
        the steep part of the cosine buys a wide sweep from a small rotation.
    colors, dpi:
        Palette size and resolution. Both trade picture quality against
        file size; frame count is the other lever.

    Returns
    -------
    EpisodeAnimation
        Plays inline in a notebook; ``.save(path)`` writes the GIF.
    """
    cfg = batch.cfg
    if cfg.n_contexts < 2:
        raise ValueError(
            "phase modulation needs two active variables to interact; "
            "n_contexts=1 has a single self-interaction and no pair to show"
        )
    if not 0 <= channel < cfg.n_observations:
        raise ValueError(
            f"channel must lie in [0, {cfg.n_observations}), got {channel}"
        )

    goal = int(batch.goal_ind[episode])
    if pair is None:
        pair = display_pairs(cfg.n_contexts, goal, max_pairs=1)[0]
    i, j = pair
    colours = (palette.context(i, goal), palette.context(j, goal))

    waveform = _standard_waveform(cfg)
    pattern = _phase_pattern(cfg, waveform)
    realizations = np.arange(cfg.n_realizations)
    swing_i, swing_j = np.deg2rad(swing)
    centre_i, centre_j = np.deg2rad(centre)
    phase = 2 * np.pi * np.arange(n_frames) / n_frames
    theta_i = np.pi / 2 + centre_i + swing_i * np.sin(phase)
    theta_j = -np.pi / 2 + centre_j + swing_j * np.sin(2 * phase)
    swept_ij = np.cos(theta_i - np.pi / 2)
    swept_ji = np.cos(theta_j + np.pi / 2)
    anchor = (_window_anchor(swept_ij, cfg), _window_anchor(swept_ji, cfg))

    fig = Figure(figsize=figsize, dpi=dpi, layout="constrained")
    axes = fig.subplots(1, 4, squeeze=False)[0]

    def draw(frame: int) -> list[Any]:
        for ax in axes:
            ax.clear()

        z_ij, z_ji = float(swept_ij[frame]), float(swept_ji[frame])
        table = gen.sigmoid(np.outer(
            _wave_at(cfg, waveform, z_ij * cfg.n_roll - realizations),
            _wave_at(cfg, waveform, z_ji * cfg.n_roll - realizations)))
        _draw_embedding_circle(
            axes[0], (i, j), colours,
            (float(theta_i[frame]), float(theta_j[frame])), (z_ij, z_ji),
            palette=palette, emphasise_keys=True)
        axes[0].set_title("impact of rotating embeddings", fontsize=10)
        _draw_phase_columns(axes[1:], cfg, waveform, pattern, (i, j), colours,
                            (z_ij, z_ji), table, palette=palette,
                            legend=False, window_anchor=anchor)

        fig.suptitle(
            "one standard likelihood, translated by the interaction angles  —  "
            f"vars {batch.ctx_inds[episode, i]} and {batch.ctx_inds[episode, j]}, "
            "rates dark 0 to light 1",
            fontsize=12,
        )
        return []

    anim = FuncAnimation(fig, draw, frames=n_frames, interval=1000 // fps, blit=False)
    return EpisodeAnimation(fig, anim, fps, n_frames, draw=draw, colors=colors)
