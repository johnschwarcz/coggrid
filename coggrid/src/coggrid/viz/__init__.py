"""Plotting for CogGrid.

Optional: install with ``pip install 'coggrid[viz]'``. The
environment itself never imports matplotlib.

Every function takes data (an :class:`~coggrid.world.EpisodeBatch`
or a :class:`~coggrid.observers.BeliefTrace`), returns a
``Figure``, and never calls ``show()``.

>>> from coggrid import CogGridConfig, World, run_observers
>>> from coggrid.viz import plot_performance
>>> batch = World(CogGridConfig(n_states=80, seed=0)).sample_episodes(64)
>>> fig = plot_performance(run_observers(batch))
>>> fig.savefig("performance.png", dpi=150)   # doctest: +SKIP
"""

from __future__ import annotations

from .plots import (
    plot_likelihood,
    plot_performance,
    plot_regret,
    plot_trial,
    summary_figure,
)
from .style import PALETTE, Palette, label_axes, transparent_cmap

__all__ = [
    "plot_likelihood",
    "plot_trial",
    "plot_performance",
    "plot_regret",
    "summary_figure",
    "Palette",
    "PALETTE",
    "transparent_cmap",
    "label_axes",
]
