"""Generate the standard diagnostic figures.

    python examples/04_figures.py --out figures/

Also safe to ``%run`` from a notebook: unrecognized arguments are ignored, so
Jupyter's own ``--f=...kernel.json`` does not trip the parser.
"""

import argparse
from pathlib import Path

from coggrid import CogGridConfig, World, run_observers
from coggrid.viz import summary_figure

# allow_abbrev=False so Jupyter's own `--f=...kernel.json` cannot prefix-match
# a future option here the way it matches `--fps` in 05_animation.py.
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--out", type=Path, default=Path("figures"))
parser.add_argument("--episodes", type=int, default=15000)
parser.add_argument("--seed", type=int, default=None)
args, _ignored = parser.parse_known_args()
args.out.mkdir(parents=True, exist_ok=True)

cfg = CogGridConfig(n_vars=500, n_contexts=2, n_realizations=10, seed=None)
batch = World(cfg).sample_episodes(args.episodes)
traces = run_observers(batch)

names = ["episode", "performance", "regret_analysis", "belief_shape"]
for name, fig in zip(names, summary_figure(batch, traces, episode=0), strict=False):
    path = args.out / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("wrote", path)
