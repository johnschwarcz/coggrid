"""Generate the standard diagnostic figures.

    python examples/04_figures.py --out figures/
"""

import argparse
from pathlib import Path

from coggrid import CogGridConfig, World, run_observers
from coggrid.viz import summary_figure

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, default=Path("figures"))
parser.add_argument("--episodes", type=int, default=1000)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

cfg = CogGridConfig(n_states=400, n_contexts=2, n_realizations=10, seed=args.seed)
batch = World(cfg).sample_episodes(args.episodes)
traces = run_observers(batch)

names = ["likelihood", "trial", "performance", "regret"]
for name, fig in zip(names, summary_figure(batch, traces, episode=0)):
    path = args.out / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("wrote", path)
