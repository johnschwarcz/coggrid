"""Swap out a generative stage without touching the library.

Replaces the default Gaussian embeddings with a small discrete codebook, so
many latent variables share an embedding. The joint observer's advantage should
shrink, because variables become confusable.

    python examples/03_customize.py
"""

import numpy as np

from coggrid import CogGridConfig, World, run_observers


def codebook_embeddings(cfg, rng, n_codes=8):
    """An EmbeddingSource: (cfg, rng) -> (keys, queries)."""

    def draw():
        codes = rng.standard_normal((n_codes, cfg.n_observations, cfg.embedding_dim))
        for o in range(cfg.n_observations):
            for prev in range(o):
                proj = np.einsum("sk,sk->s", codes[:, o], codes[:, prev])
                codes[:, o] -= proj[:, None] * codes[:, prev]
            codes[:, o] /= np.linalg.norm(codes[:, o], axis=-1, keepdims=True)
        return codes[rng.integers(0, n_codes, size=cfg.n_vars)]

    return draw(), draw()


cfg = CogGridConfig(n_vars=400, n_contexts=2, n_realizations=8, n_steps=30, seed=0)

variants = [
    ("default (continuous)", None),
    ("codebook (8 codes)", codebook_embeddings),
]

for label, source in variants:
    world = World(cfg, embeddings=source)
    traces = run_observers(world.sample_episodes(1000))
    joint, naive = traces["joint"].final(), traces["naive"].final()
    print(
        f"{label:22s} joint={joint['accuracy']:.3f}  naive={naive['accuracy']:.3f}  "
        f"gap={joint['accuracy'] - naive['accuracy']:+.3f}"
    )
