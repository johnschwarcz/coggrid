class _Sched:
    def __init__(self, *a, **k): pass
    def step(self, *a, **k): pass
StepLR = ExponentialLR = CosineAnnealingLR = LambdaLR = MultiStepLR = _Sched
