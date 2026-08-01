class _Opt:
    def __init__(self, *a, **k): pass
    def step(self, *a, **k): pass
    def zero_grad(self, *a, **k): pass
Adam = AdamW = SGD = RMSprop = _Opt
from . import lr_scheduler
