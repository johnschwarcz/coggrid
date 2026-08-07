"""Regenerate the images the README embeds.

    python docs/make_assets.py

Seeds are pinned here deliberately. Everywhere else in this repository the
default is ``seed=None`` so each run explores a fresh episode; these are the one
exception, because a figure in the README should not change every time someone
regenerates it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from coggrid import CogGridConfig, World, run_observers
from coggrid.viz import (
    animate_episode,
    plot_episode,
    plot_evidence_likelihood,
    plot_interaction_phases,
    plot_regret_analysis,
)

# Resolve the repository root without assuming __file__ exists: VS Code's
# "run in interactive window" pastes this source into a cell rather than
# executing the file, and a pasted cell has no __file__ at all.
if globals().get("__file__") is not None:
    ROOT = Path(__file__).resolve().parent.parent
else:
    _CWD = Path.cwd().resolve()
    _CANDIDATES = (p for p in (_CWD, *_CWD.parents) if (p / "pyproject.toml").exists())
    ROOT = next(_CANDIDATES, _CWD)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "images")
    parser.add_argument("--episodes", type=int, default=15000)
    args, _ignored = parser.parse_known_args()
    args.out.mkdir(parents=True, exist_ok=True)
    
    
    cfg = CogGridConfig(n_vars=500, n_contexts=2, n_realizations=10, seed=4)
    world = World(cfg)
    batch = world.sample_episodes(args.episodes)
    traces = run_observers(batch)

    for name, fig in (
        ("episode", plot_episode(batch, traces, episode=0)),
        ("regret_analysis", plot_regret_analysis(batch, traces)),
        ("interaction_phases", plot_interaction_phases(world, batch, episode=0)),
        ("evidence_likelihood", plot_evidence_likelihood(batch, episode=0)),
    ):
        path = args.out / f"{name}.png"
        fig.savefig(path, dpi=110, bbox_inches="tight")
        # These are pyplot-managed, so they stay alive until closed.
        plt.close(fig)
        print("wrote", path, f"({path.stat().st_size / 1e3:.0f} kB)")

    # One animation per seed, numbered in order. Add a seed here and the README
    # gains episode_animation_2.gif, and so on.
    ANIMATION_SEEDS = [37]

    for n, seed in enumerate(ANIMATION_SEEDS, start=1):
        anim = CogGridConfig(
            n_vars=500, n_contexts=2, n_realizations=10, n_steps=30, seed=seed)
        W = World(anim).sample_episodes(1)
        clip = animate_episode(W, run_observers(W), extended=False, fps=6)
        path = clip.save(args.out / f"episode_animation_{n}")
        print("wrote", path, f"({path.stat().st_size / 1e3:.0f} kB)")
