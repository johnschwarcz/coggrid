"""Colors and small axis helpers shared by every plot.

Gathering them here means a user can restyle every figure by editing one dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

__all__ = ["Palette", "PALETTE", "transparent_cmap", "label_axes"]


@dataclass(frozen=True)
class Palette:
    """Colors used across the figures.

    ``joint`` and ``naive`` identify the two observers everywhere they appear;
    ``goal`` highlights the variable the agent is scored on, so it reads the same
    in a likelihood heatmap and in a belief trace. ``regret`` is deliberately
    neither observer's color — it is a quantity *between* them.
    """

    joint: str = "#1f77b4"
    naive: str = "#8c8c8c"
    regret: str = "#b5495b"
    goal: str = "#78c765"
    other: str = "#ef8e3b"
    truth: str = "#222222"
    observation: str = "#3b3b6d"
    grid: str = "#dddddd"

    #: Per-observer colors keyed by ``BeliefTrace.name``.
    by_observer: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "by_observer", {"joint": self.joint, "naive": self.naive}
        )

    def for_observer(self, name: str) -> str:
        """Color for an observer, falling back to the matplotlib cycle."""
        return self.by_observer.get(name, "C3")

    def context(self, context: int, goal_context: int) -> str:
        """Green if this context is the goal, orange otherwise."""
        return self.goal if context == goal_context else self.other


#: The default palette. Replace attributes on your own instance and pass it in.
PALETTE = Palette()


def transparent_cmap(color: str, name: str = "cg") -> ListedColormap:
    """A colormap running from fully transparent to ``color``.

    Belief heatmaps are drawn on top of gridlines and truth markers, so the
    low-probability end needs to be see-through rather than white.
    """
    return LinearSegmentedColormap.from_list(name, [(1, 1, 1, 0), color])  # type: ignore[return-value]


def label_axes(
    ax: plt.Axes,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    fontsize: int = 10,
) -> plt.Axes:
    """Set labels only where a value was supplied."""
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize + 1)
    return ax
