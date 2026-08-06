"""Plotting for CogGrid.

Optional: install with ``pip install 'coggrid[viz]'``. The
environment itself never imports matplotlib.

Every function takes data (an :class:`~coggrid.world.EpisodeBatch`
or a :class:`~coggrid.observers.BeliefTrace`), returns a
``Figure``, and never calls ``show()``.

>>> from coggrid import CogGridConfig, World, run_observers
>>> from coggrid.viz import plot_performance
>>> batch = World(CogGridConfig(n_vars=80, seed=0)).sample_episodes(64)
>>> fig = plot_performance(run_observers(batch))
>>> fig.savefig("performance.png", dpi=150)   # doctest: +SKIP
"""

from __future__ import annotations

from .animate import (
    DEFAULT_PANELS,
    GRID_PANELS,
    TRACE_PANELS,
    EpisodeAnimation,
    EpisodeView,
    animate_episode,
    default_panels,
    difference_panel,
    joint_posterior_panel,
    marginal_beliefs_panel,
    naive_posterior_panel,
    observations_panel,
    regret_rate_panel,
)
from .animate import (
    regret_panel as animation_regret_panel,
)
from .plots import (
    MAX_PAIRS,
    display_pairs,
    plot_belief_profile,
    plot_belief_shape,
    plot_confidence_density,
    plot_episode,
    plot_evidence_likelihood,
    plot_interaction_phases,
    plot_likelihood,
    plot_map_agreement,
    plot_performance,
    plot_regret,
    plot_regret_analysis,
    plot_regret_vs_accuracy,
    plot_relative_accuracy,
    plot_trial,
    summary_figure,
)
from .style import PALETTE, Palette, label_axes, transparent_cmap

__all__ = [
    # animation
    "animate_episode",
    "EpisodeAnimation",
    "EpisodeView",
    "DEFAULT_PANELS",
    "GRID_PANELS",
    "TRACE_PANELS",
    "joint_posterior_panel",
    "difference_panel",
    "marginal_beliefs_panel",
    "animation_regret_panel",
    "regret_rate_panel",
    "default_panels",
    "naive_posterior_panel",
    "observations_panel",
    # static figures
    "display_pairs",
    "MAX_PAIRS",
    "plot_likelihood",
    "plot_interaction_phases",
    "plot_trial",
    "plot_episode",
    "plot_evidence_likelihood",
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
    "Palette",
    "PALETTE",
    "transparent_cmap",
    "label_axes",
]
