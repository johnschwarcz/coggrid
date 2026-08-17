"""Sample a batch of episodes and score the two ideal observers.

    python examples/01_quickstart.py
"""

from coggrid import CogGridConfig, World, factorization_regret, run_observers

cfg = CogGridConfig(
    n_vars=500,      # size of the latent variable pool
    n_contexts=2,      # active variables per episode
    n_realizations=10, # values each active variable can take
    n_observations=5,  # binary channels
    n_steps=30,        # evidence samples per episode
    seed=0,
)
print(cfg)
print()
print(cfg.memory_report(batch_size=2000))
print()

world = World(cfg)
batch = world.sample_episodes(2000, split="held_out")
print(batch)

traces = run_observers(batch)
for name, trace in traces.items():
    final = trace.final()
    print(
        f"  {name:6s} accuracy={final['accuracy']:.3f}  "
        f"p_correct={final['p_correct']:.3f}  mse={final['mse']:.3f}"
    )

regret = factorization_regret(traces["joint"], traces["naive"])
print(f"\nfactorization regret at final step: {regret[:, -1].mean():.4f} nats")
