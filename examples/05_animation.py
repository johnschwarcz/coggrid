"""Play an episode back, and extend the playback with your own panel.

    python examples/05_animation.py --out figures/

In a notebook you do not need this script at all — ``animate_episode`` plays
itself as the last expression in a cell::

    from coggrid.viz import animate_episode
    animate_episode(batch, traces)

``--extended`` adds the goal variable's marginal belief and a panel putting
factorization regret next to the update divergence that causes it.
"""

import argparse
from pathlib import Path

from coggrid import CogGridConfig, World, run_observers
from coggrid.viz import animate_episode, default_panels

if __name__ == "__main__":
    # allow_abbrev=False so Jupyter's own `--f=...kernel.json` cannot
    # prefix-match an option here and die converting a path to int.
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--contexts", type=int, default=2)
    parser.add_argument("--extended", action=argparse.BooleanOptionalAction, default=True)
    
    args, _ignored = parser.parse_known_args()

    cfg = CogGridConfig(
        n_vars=400, n_contexts=args.contexts, n_realizations=10, n_steps=30, seed=None
    )
    batch = World(cfg).sample_episodes(1)
    traces = run_observers(batch)

    # default_panels adapts to n_contexts: with one variable there is no grid
    # to draw, so hardcoding GRID_PANELS here would fail.
    rows = default_panels(args.contexts, args.extended)

    clip = animate_episode(
        batch, traces, episode=args.episode, panels=rows, fps=args.fps,
       
    )
    print("wrote", clip.save(args.out / "episode"))

    # %run from a notebook never echoes a trailing expression, so ask for the
    # inline playback explicitly. A no-op outside an IPython kernel.
    clip.display()
