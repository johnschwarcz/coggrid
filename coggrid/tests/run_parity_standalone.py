"""Run the parity checks without pytest (useful in minimal environments).

`pytest tests/` is the normal entry point; this mirrors the same assertions so
the port can be validated with numpy alone.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from coggrid import CogGridConfig  # noqa: E402
from coggrid.generative import (  # noqa: E402
    joint_likelihood,
    marginal_likelihood,
    observation_rates,
    value_profile,
)
from coggrid.observers import (  # noqa: E402
    factorization_regret,
    joint_observer,
    naive_observer,
)
from coggrid.world import EpisodeBatch  # noqa: E402

CFG = CogGridConfig(
    n_states=40, n_held_out_states=4, n_observations=3, n_contexts=2,
    n_realizations=6, embedding_dim=8, n_episodes=32, n_steps=12,
    likelihood_temp=2.0, likelihood_freq=1.0,
)

failures: list[str] = []


def check(name: str, mine: np.ndarray, theirs: np.ndarray, atol: float = 1e-10) -> None:
    mine, theirs = np.asarray(mine), np.asarray(theirs)
    if mine.shape != theirs.shape:
        failures.append(f"{name}: shape {mine.shape} != {theirs.shape}")
        print(f"  FAIL {name:24s} shape {mine.shape} != {theirs.shape}")
        return
    err = float(np.abs(mine - theirs).max())
    ok = err <= atol
    if not ok:
        failures.append(f"{name}: max abs err {err:.3e}")
    print(f"  {'ok  ' if ok else 'FAIL'} {name:24s} {str(mine.shape):18s} max|Δ| = {err:.3e}")


def main() -> int:
    path = pathlib.Path(__file__).parent / "reference.npz"
    with np.load(path) as data:
        ref = {k: data[k] for k in data.files}

    rates, interactions = joint_likelihood(CFG, ref["all_K"], ref["all_Q"], ref["ctx_inds"])
    ctx_vals = ref["ctx_vals"]
    batch = EpisodeBatch(
        cfg=CFG, split="held_out",
        ctx_inds=ref["ctx_inds"], goal_ind=ref["goal_ind"],
        ctx_vals=ctx_vals,
        goal_value=ctx_vals[np.arange(len(ctx_vals)), ref["goal_ind"]],
        rates=rates,
        marginal_rates=marginal_likelihood(rates, CFG.n_contexts),
        true_rates=observation_rates(rates, ctx_vals),
        observations=ref["obs_flat"], interactions=interactions,
    )

    print("generative model")
    check("value_profile", value_profile(CFG), ref["roll_V_range"])
    check("joint_likelihood", batch.rates, ref["joint_likelihood"])
    check("interactions", batch.interactions, ref["joint_Z"])
    check("marginal_likelihood", batch.marginal_rates, ref["naive_likelihood"])
    check("observation_rates", batch.true_rates, ref["pobs"])

    print("\nobservers")
    traces = {"joint": joint_observer(batch), "naive": naive_observer(batch)}
    for name, trace in traces.items():
        check(f"{name}_belief", trace.belief, ref[f"{name}_belief"])
        check(f"{name}_acc", trace.accuracy, ref[f"{name}_acc"])
        check(f"{name}_TP", trace.p_correct, ref[f"{name}_TP"])
        check(f"{name}_mse", trace.mse, ref[f"{name}_mse"])
        check(f"{name}_est", trace.estimate, ref[f"{name}_est"])
    check("factorization_regret",
          factorization_regret(traces["joint"], traces["naive"]), ref["SII"], atol=1e-9)

    print(f"\nrate range: [{batch.rates.min():.4f}, {batch.rates.max():.4f}]  "
          f"(log-space clipping inactive)")
    print(f"final p_correct  joint={traces['joint'].final()['p_correct']:.4f}  "
          f"naive={traces['naive'].final()['p_correct']:.4f}")

    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for f in failures:
            print("  -", f)
        return 1
    print("\nAll parity checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
