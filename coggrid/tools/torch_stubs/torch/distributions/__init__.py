class Categorical:
    def __init__(self, *a, **k): pass
    def sample(self, *a, **k): raise NotImplementedError("torch stub")
    def log_prob(self, *a, **k): raise NotImplementedError("torch stub")
    def entropy(self): raise NotImplementedError("torch stub")
Normal = Bernoulli = Categorical
