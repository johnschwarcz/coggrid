"""Contract tests for the plotting and animation surface.

These check the things a refactor could silently break: that an animation
renders the frames it claims to, that a custom panel is actually called, and
that the figure stays out of pyplot's hands — the last one is what keeps a
notebook from showing a still frame beside the animation.
"""

from __future__ import annotations

import io

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from coggrid import CogGridConfig, World, run_observers  # noqa: E402
from coggrid.viz import (  # noqa: E402
    DEFAULT_PANELS,
    GRID_PANELS,
    TRACE_PANELS,
    animate_episode,
    observations_panel,
    plot_episode,
    plot_performance,
    summary_figure,
)

SMALL = CogGridConfig(
    n_vars=60, n_held_out_vars=6, n_contexts=2, n_realizations=5,
    n_observations=3, n_steps=8, embedding_dim=6, seed=0,
)


@pytest.fixture(scope="module")
def batch():
    return World(SMALL).sample_episodes(24)


@pytest.fixture(scope="module")
def traces(batch):
    return run_observers(batch)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# Several tests build an animation to inspect it without ever rendering it.
# matplotlib warns when such an animation is garbage-collected, and the warning
# surfaces in whichever test happens to trigger the collection.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Animation was deleted without rendering anything:UserWarning"
)


class TestAnimation:
    def test_frame_count_matches_horizon(self, batch, traces):
        assert animate_episode(batch, traces).n_frames == SMALL.n_steps

    def test_gif_is_animated(self, batch, traces):
        from PIL import Image

        data = animate_episode(batch, traces).to_gif()
        assert data[:6] in (b"GIF87a", b"GIF89a")
        assert Image.open(io.BytesIO(data)).n_frames == SMALL.n_steps

    def test_gif_is_cached(self, batch, traces):
        """Re-displaying in a notebook must not re-render every frame."""
        clip = animate_episode(batch, traces)
        assert clip.to_gif() is clip.to_gif()

    def test_displays_as_gif_in_notebooks(self, batch, traces):
        bundle = animate_episode(batch, traces)._repr_mimebundle_()
        assert "image/gif" in bundle
        # An image/png here would be the still-frame bug coming back.
        assert "image/png" not in bundle

    def test_figure_is_not_pyplot_managed(self, batch, traces):
        """The inline backend auto-renders pyplot figures as a still frame."""
        plt.close("all")
        animate_episode(batch, traces)
        assert plt.get_fignums() == []

    def test_save_writes_a_gif(self, batch, traces, tmp_path):
        path = animate_episode(batch, traces).save(tmp_path / "clip")
        assert path.suffix == ".gif" and path.stat().st_size > 0

    def test_custom_panel_is_called(self, batch, traces):
        seen: list[int] = []

        def probe(ax, view):
            assert view.n_steps == SMALL.n_steps
            assert view.truth.shape == (SMALL.n_contexts,)
            return lambda t: seen.append(t)

        clip = animate_episode(
            batch, traces, panels=[GRID_PANELS, [*TRACE_PANELS, probe]]
        )
        clip.to_gif()
        # FuncAnimation draws frame 0 twice — once to initialise, once as the
        # first frame — so check coverage and ordering rather than the raw list.
        assert set(seen) == set(range(SMALL.n_steps))
        assert seen == sorted(seen)
        assert len(clip.fig.axes) == len(GRID_PANELS) + len(TRACE_PANELS) + 1

    def test_accepts_a_flat_row_of_panels(self, batch, traces):
        clip = animate_episode(batch, traces, panels=GRID_PANELS)
        assert len(clip.fig.axes) == len(GRID_PANELS)

    def test_rejects_mixed_rows_and_panels(self, batch, traces):
        with pytest.raises(ValueError, match="not a mix"):
            animate_episode(batch, traces, panels=[GRID_PANELS, observations_panel])

    def test_rejects_empty_row(self, batch, traces):
        with pytest.raises(ValueError, match="empty row"):
            animate_episode(batch, traces, panels=[GRID_PANELS, []])

    def test_default_layout_is_two_rows(self, batch, traces):
        assert len(DEFAULT_PANELS) == 2
        clip = animate_episode(batch, traces)
        assert len(clip.fig.axes) == len(GRID_PANELS) + len(TRACE_PANELS)
        # Grids on top, traces underneath: the first row sits higher on the page.
        tops = [ax.get_position().y0 for ax in clip.fig.axes]
        assert min(tops[: len(GRID_PANELS)]) > max(tops[len(GRID_PANELS) :])

    def test_difference_grid_is_the_signed_gap(self, batch, traces):
        from coggrid.viz import EpisodeView

        view = EpisodeView(batch=batch, traces=traces)
        assert np.allclose(view.difference_grid, view.joint_grid - view.naive_grid)
        assert np.allclose(view.difference_grid.sum(axis=(-2, -1)), 0.0, atol=1e-9)

    def test_modes_track_the_argmax(self, batch, traces):
        from coggrid.viz import EpisodeView

        view = EpisodeView(batch=batch, traces=traces)
        modes = view.modes(view.joint_grid)
        assert modes.shape == (SMALL.n_steps, 2)
        for t, (i, j) in enumerate(modes):
            assert view.joint_grid[t][i, j] == view.joint_grid[t].max()

    def test_variable_roles_name_the_goal(self, batch, traces):
        from coggrid.viz import EpisodeView

        view = EpisodeView(batch=batch, traces=traces)
        goal, other = view.goal_context, 1 - view.goal_context
        assert "goal" in view.variable_label(goal)
        assert "context" in view.variable_label(other)
        assert view.variable_color(goal) != view.variable_color(other)

    def test_view_grids_are_distributions(self, batch, traces):
        from coggrid.viz import EpisodeView

        view = EpisodeView(batch=batch, traces=traces)
        for grid in (view.joint_grid, view.naive_grid):
            assert grid.shape == (SMALL.n_steps,) + (SMALL.n_realizations,) * 2
            assert np.allclose(grid.sum(axis=(-2, -1)), 1.0)

    def test_rejects_empty_panels(self, batch, traces):
        with pytest.raises(ValueError, match="at least one panel"):
            animate_episode(batch, traces, panels=[])

    def test_grid_panels_asked_for_explicitly_need_a_pair(self):
        """Adapting is automatic, but forcing the grids at C=1 should say why."""
        world = World(SMALL.replace(n_contexts=1))
        one = world.sample_episodes(4)
        with pytest.raises(ValueError, match="at least two active variables"):
            animate_episode(one, run_observers(one), panels=GRID_PANELS)


class TestStaticFigures:
    def test_summary_figure_shape(self, batch, traces):
        assert len(summary_figure(batch, traces)) == 4

    @pytest.mark.parametrize("n_contexts", [2, 3, 4])
    def test_interaction_phases_draws_a_row_per_channel(self, n_contexts):
        from coggrid.viz import plot_interaction_phases

        world = World(SMALL.replace(n_contexts=n_contexts))
        batch = world.sample_episodes(4)
        for channels in ((0, 1), (0, 1, 2), (2,)):
            fig = plot_interaction_phases(world, batch, episode=0, channels=channels)
            assert len(fig.axes) == 4 * len(channels)

    def test_interaction_phases_rejects_a_missing_channel(self):
        from coggrid.viz import plot_interaction_phases

        world = World(SMALL.replace(n_contexts=2))
        with pytest.raises(ValueError, match="channels must lie in"):
            plot_interaction_phases(
                world, world.sample_episodes(2), channels=(0, SMALL.n_observations)
            )

    def test_interaction_phases_angle_is_the_interaction_strength(self):
        """The drawn angle must be ``arccos(z)`` for the strength the env used.

        Embeddings are unit vectors, so the cosine of the angle between a key and
        a query *is* the interaction strength — the panel would be decorative
        rather than informative if these came apart.
        """
        from coggrid.viz import plot_interaction_phases

        world = World(SMALL.replace(n_contexts=2))
        batch = world.sample_episodes(4)
        plot_interaction_phases(world, batch, episode=0, channels=(0,))

        i, j = 0, 1
        for a, b in ((i, j), (j, i)):
            var_a = int(batch.ctx_inds[0, a])
            var_b = int(batch.ctx_inds[0, b])
            cosine = world.keys[var_a, 0] @ world.queries[var_b, 0]
            from coggrid.viz.plots import _pair_rank

            stored = batch.interactions[0, 0, _pair_rank(a, b, 2)]
            assert cosine == pytest.approx(stored, abs=1e-12)
            assert abs(cosine) <= 1.0

    def test_interaction_phases_needs_a_pair(self):
        """A single variable has no partner to take a phase against."""
        from coggrid.viz import plot_interaction_phases

        world = World(SMALL.replace(n_contexts=1))
        with pytest.raises(ValueError, match="two active variables"):
            plot_interaction_phases(world, world.sample_episodes(2))

    def test_interaction_phases_last_panel_is_the_real_table(self):
        """The final panel must be read from ``batch.rates``, not rebuilt.

        The earlier panels illustrate the built-in mechanism; this one has to
        stay true even when a custom ``likelihood=`` made the others meaningless.
        """
        from coggrid.viz import plot_interaction_phases

        world = World(SMALL.replace(n_contexts=2))
        batch = world.sample_episodes(4)
        fig = plot_interaction_phases(world, batch, episode=0, channels=(0,))
        drawn = np.asarray(fig.axes[3].get_images()[0].get_array())
        # Compared as multisets: which variable lands on which axis depends on
        # where the goal fell, and that is not what this test is about.
        assert np.allclose(np.sort(drawn, axis=None),
                           np.sort(batch.rates[0, 0], axis=None))

    def test_evidence_likelihood_draws_every_vector(self):
        """One panel per channel, plus one per possible observation vector."""
        from coggrid.viz import plot_evidence_likelihood

        cfg = SMALL.replace(n_contexts=2, n_observations=4)
        fig = plot_evidence_likelihood(World(cfg).sample_episodes(4), episode=0)
        assert len(fig.axes) == cfg.n_observations + 2**cfg.n_observations

    def test_evidence_likelihood_caps_the_grid(self):
        """2**n_observations explodes, so a wide batch must subsample instead."""
        from coggrid.viz import plot_evidence_likelihood

        cfg = SMALL.replace(n_contexts=2, n_observations=10, embedding_dim=30)
        fig = plot_evidence_likelihood(
            World(cfg).sample_episodes(2), episode=0, max_vectors=8
        )
        assert len(fig.axes) == cfg.n_observations + 8

    def test_evidence_likelihood_is_the_bernoulli_product(self):
        """Each panel must be the normalized product of the channel rates.

        Derived here straight from ``batch.rates`` rather than from anything the
        plotting code shares, so an error in the einsum cannot cancel out.
        """
        from coggrid.viz import plot_evidence_likelihood

        cfg = SMALL.replace(n_contexts=2, n_observations=3)
        batch = World(cfg).sample_episodes(4)
        fig = plot_evidence_likelihood(batch, episode=0)

        rates = batch.rates[0]                       # (n_obs, R, R)
        for vector in range(2**cfg.n_observations):
            bits = [(vector >> c) & 1 for c in range(cfg.n_observations)]
            expected = np.ones_like(rates[0])
            for c, bit in enumerate(bits):
                expected = expected * (rates[c] if bit else 1.0 - rates[c])
            expected = expected / expected.max()
            drawn = np.asarray(
                fig.axes[cfg.n_observations + vector].get_images()[0].get_array()
            )
            assert np.allclose(np.sort(drawn, axis=None),
                               np.sort(expected, axis=None), atol=1e-9)

    def test_pair_rank_matches_the_interaction_layout(self):
        """``interactions`` stores (0,1), (1,0), (0,2), (2,0), ... in that order."""
        from coggrid.viz.plots import _pair_rank

        assert [_pair_rank(i, j, 3) for i, j in
                ((0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1))] == [0, 1, 2, 3, 4, 5]

    def test_single_context_drops_comparisons(self):
        """With one variable the observers coincide, so contrasts are degenerate."""
        world = World(SMALL.replace(n_contexts=1))
        one = world.sample_episodes(8)
        assert len(summary_figure(one, run_observers(one))) == 2

    def test_regret_column_dropped_for_single_observer(self, batch, traces):
        solo = plot_performance({"joint": traces["joint"]})
        assert [ax.get_title() for ax in solo.axes] == [
            "P(mode correct)", "P(true value)", "squared error"
        ]

    def test_episode_figure_composes_both_halves(self, batch, traces):
        titles = [ax.get_title() for ax in plot_episode(batch, traces).axes]
        assert any("joint" in t for t in titles)
        assert any("observations" in t for t in titles)


class TestContextCounts:
    """Every figure has to survive n_contexts of 1, 2, 3 and more."""

    @staticmethod
    def _episode(n_contexts):
        cfg = SMALL.replace(n_contexts=n_contexts)
        b = World(cfg).sample_episodes(8)
        return b, run_observers(b)

    @pytest.mark.parametrize("n_contexts", [1, 2, 3, 5])
    def test_animation_adapts(self, n_contexts):
        b, t = self._episode(n_contexts)
        for extended in (False, True):
            clip = animate_episode(b, t, extended=extended)
            clip.to_gif()
            assert clip.n_frames == SMALL.n_steps

    @pytest.mark.parametrize("n_contexts", [1, 2, 3, 5])
    def test_static_figures_survive(self, n_contexts):
        from coggrid.viz import plot_belief_shape, plot_likelihood, plot_trial

        b, t = self._episode(n_contexts)
        for fig in (plot_likelihood(b), plot_trial(b, t), plot_episode(b, t),
                    plot_performance(t), plot_belief_shape(b, t)):
            assert fig.axes

    @pytest.mark.parametrize("n_contexts", [4, 6, 8])
    def test_pair_panels_stay_bounded(self, n_contexts):
        """Pairs grow as C(C-1)/2; the figure must not."""
        from coggrid.viz import MAX_PAIRS, display_pairs, plot_likelihood

        b, _ = self._episode(n_contexts)
        assert len(display_pairs(n_contexts, 0)) == MAX_PAIRS
        # one joint panel plus its two marginal strips, per shown pair
        assert len(plot_likelihood(b).axes) == 3 * MAX_PAIRS

    def test_pairs_prefer_the_goal_variable(self):
        from coggrid.viz import display_pairs

        assert display_pairs(1, 0) == []
        assert display_pairs(2, 0) == [(0, 1)]
        assert all(2 in p for p in display_pairs(4, 2)[:3])

    def test_single_variable_drops_the_grids(self):
        from coggrid.viz import default_panels, marginal_beliefs_panel

        rows = default_panels(1)
        assert marginal_beliefs_panel in rows[0]
        assert len(rows[0]) == 1  # no joint/naive/difference to draw

    def test_extended_adds_marginals_and_cost_panel(self):
        from coggrid.viz import (
            animation_regret_panel,
            default_panels,
            marginal_beliefs_panel,
        )

        plain, extended = default_panels(2), default_panels(2, extended=True)
        assert marginal_beliefs_panel in extended[0]
        assert animation_regret_panel in extended[1]
        # marginals on the state row, plus the stock and flow regret panels
        assert sum(map(len, extended)) == sum(map(len, plain)) + 3
        # With one variable the observers coincide, so the KL panel is pointless.
        assert animation_regret_panel not in default_panels(1, extended=True)[1]

    def test_disentanglement_is_zero_without_interaction(self):
        """One variable means no interaction, so the updates cannot differ."""
        from coggrid.viz import EpisodeView

        b, t = self._episode(1)
        view = EpisodeView(batch=b, traces=t)
        assert view.disentanglement.shape == (SMALL.n_steps, 1)
        assert np.allclose(view.disentanglement, 0.0, atol=1e-9)

    def test_disentanglement_is_positive_with_interaction(self):
        b, t = self._episode(2)
        from coggrid.viz import EpisodeView

        assert (EpisodeView(batch=b, traces=t).disentanglement >= -1e-9).all()
        assert EpisodeView(batch=b, traces=t).disentanglement.max() > 0

    def test_disentanglement_vanishes_when_dynamics_are_markovian(self):
        """Paper §B.3: the optimal update equals the naive update exactly when
        the belief dynamics are Markovian with respect to the marginals. With a
        single variable there is no context to entangle with, so it must be 0.
        """
        from coggrid.viz import EpisodeView

        b, t = self._episode(1)
        view = EpisodeView(batch=b, traces=t)
        assert view.disentanglement.shape == (SMALL.n_steps, 1)
        assert np.allclose(view.disentanglement, 0.0, atol=1e-9)

    def test_disentanglement_slices_the_library_metric(self):
        """The view must expose exactly its own episode of the §B.3 metric.

        The maths itself is tested in ``test_environment.py``; what matters here
        is that the animation reads the right episode out of it.
        """
        from coggrid import disentanglement
        from coggrid.viz import EpisodeView

        b, t = self._episode(2)
        batched = disentanglement(t["joint"], t["naive"], b, per_variable=True)
        for e in range(b.n_episodes):
            view = EpisodeView(batch=b, traces=t, episode=e)
            assert np.array_equal(view.disentanglement, batched[e])

    def test_cumulative_is_the_running_total(self):
        from coggrid.viz import EpisodeView

        b, t = self._episode(2)
        view = EpisodeView(batch=b, traces=t)
        assert np.allclose(
            view.cumulative_disentanglement,
            np.cumsum(view.disentanglement, axis=0),
        )
        # monotone by construction: it sums non-negative KLs
        assert (np.diff(view.cumulative_disentanglement, axis=0) >= -1e-12).all()

    def test_every_variable_gets_its_own_colour(self):
        from coggrid.viz import EpisodeView

        for n_contexts in (2, 3, 5):
            b, t = self._episode(n_contexts)
            view = EpisodeView(batch=b, traces=t)
            colours = {view.variable_color(c) for c in range(n_contexts)}
            assert len(colours) == n_contexts

    def test_pair_grid_marginalizes_the_rest(self):
        from coggrid.viz import EpisodeView

        b, t = self._episode(3)
        view = EpisodeView(batch=b, traces=t)
        grid = view.joint_pair_grid(0, 2)
        assert grid.shape == (SMALL.n_steps,) + (SMALL.n_realizations,) * 2
        assert np.allclose(grid.sum(axis=(-2, -1)), 1.0)
        # transposing the request transposes the grid
        assert np.allclose(grid, view.joint_pair_grid(2, 0).transpose(0, 2, 1))


    @pytest.mark.parametrize("n_contexts", [1, 2, 3])
    def test_single_episode_batch(self, n_contexts):
        """`sample_episodes(1)` is the normal case for an animation.

        Both measures slice a leading batch axis, so a batch of one is the
        shape most likely to collapse an axis by accident.
        """
        from coggrid.viz import EpisodeView

        cfg = SMALL.replace(n_contexts=n_contexts)
        b = World(cfg).sample_episodes(1)
        t = run_observers(b)
        view = EpisodeView(batch=b, traces=t, episode=0)

        assert view.disentanglement.shape == (SMALL.n_steps, n_contexts)
        assert view.goal_regret.shape == (SMALL.n_steps,)
        assert np.isfinite(view.disentanglement).all()
        assert np.isfinite(view.goal_regret).all()

        clip = animate_episode(b, t, extended=True)
        clip.to_gif()
        assert clip.n_frames == SMALL.n_steps
