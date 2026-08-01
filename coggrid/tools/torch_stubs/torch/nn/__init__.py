class Module:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): raise NotImplementedError("torch stub")
    def parameters(self): return []
    def to(self, *a, **k): return self
class _Any:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): raise NotImplementedError("torch stub")
Linear = LayerNorm = ReLU = Tanh = GELU = Sequential = Embedding = Dropout = _Any
LSTM = GRU = RNN = Softmax = LogSoftmax = CrossEntropyLoss = MSELoss = _Any
Parameter = ParameterList = ModuleList = ModuleDict = Identity = Sigmoid = _Any
BatchNorm1d = Flatten = Unflatten = Conv1d = Conv2d = _Any
from . import functional, init
