"""
Drive the ORIGINAL CognitiveGridworld code (env-only path, torch stubbed) and dump
the deterministic intermediate arrays to disk. The refactored package must
reproduce these bit-for-bit given the same inputs.
"""
import sys, os, pathlib
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "torch_stubs"))
sys.path.insert(0, str(HERE.parent / "orig"))

import matplotlib
matplotlib.use("Agg")

from main.env.env_model_manager import Env_model_manager


class Ref(Env_model_manager):
    """Bypass CognitiveGridworld.__init__ so nothing auto-runs."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


CFG = dict(
    state_num=40, test_states=4, subsample_states=None,
    obs_num=3, ctx_num=2, realization_num=6, KQ_dim=8,
    batch_num=32, step_num=12,
    likelihood_temp=2.0, likelihood_freq=1,
    mode=None, show_plots=False, showtime=1e9,
)

def main():
    r = Ref(**CFG)
    r.preprocess_env()

    rng = np.random.RandomState(0)

    # ---- inputs we control explicitly (so the port can be fed the same ones) ----
    r.all_K = _gs(rng.randn(r.state_num, r.obs_num, r.KQ_dim), r.obs_num)
    r.all_Q = _gs(rng.randn(r.state_num, r.obs_num, r.KQ_dim), r.obs_num)
    r.ctx_inds = np.sort(rng.randint(0, r.state_num, size=(r.batch_num, r.ctx_num)), axis=1)
    r.goal_ind = rng.randint(0, r.ctx_num, size=r.batch_num)
    r.ctx_vals = rng.randint(0, r.realization_num, size=(r.batch_num, r.ctx_num))
    r.goal_value = r.ctx_vals[r.batch_range, r.goal_ind]

    # ---- deterministic pipeline ----
    r.default_likelihoods()

    ix = tuple(r.ctx_vals[:, i, None] for i in range(r.ctx_num))
    pobs = np.squeeze(r.joint_likelihood[r.batch_range[:, None], r.obs_range[None, :], *ix])
    r.pobs__joint = pobs
    r.obs_flat = rng.rand(r.batch_num, r.step_num, r.obs_num) < pobs[:, None, :]

    r.run_inference()

    out = dict(
        all_K=r.all_K, all_Q=r.all_Q, ctx_inds=r.ctx_inds, goal_ind=r.goal_ind,
        ctx_vals=r.ctx_vals, obs_flat=r.obs_flat,
        roll_V_range=r.roll_V_range,
        joint_likelihood=r.joint_likelihood, naive_likelihood=r.naive_likelihood,
        joint_Z=r.joint_Z, pobs=r.pobs__joint,
        joint_belief=r.joint_belief, naive_belief=r.naive_belief,
        joint_acc=r.joint_acc, naive_acc=r.naive_acc,
        joint_TP=r.joint_TP, naive_TP=r.naive_TP,
        joint_mse=r.joint_mse, naive_mse=r.naive_mse,
        joint_est=r.joint_est, naive_est=r.naive_est,
        SII=r.SII,
    )
    np.savez_compressed(HERE / "reference.npz", **out)
    for k, v in out.items():
        v = np.asarray(v)
        print(f"{k:20s} {str(v.shape):18s} {v.dtype}")


def _gs(v, obs_num):
    """Replica of the original gram_shmidt_KQ, kept here as reference input."""
    for o in range(obs_num):
        for prev in range(o):
            proj = np.einsum("sk,sk->s", v[:, o], v[:, prev])
            v[:, o] -= proj[..., None] * v[:, prev]
        v[:, o] /= np.linalg.norm(v[:, o], axis=-1, keepdims=True)
    return v


if __name__ == "__main__":
    main()
