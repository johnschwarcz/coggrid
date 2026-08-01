"""Minimal PyTorch stub: enough for the ORIGINAL env-only path to import."""
class _Cuda:
    @staticmethod
    def is_available(): return False
cuda = _Cuda()
class Tensor: pass
def device(x): return x
def from_numpy(x): raise NotImplementedError("torch stub")
class _Amp:
    @staticmethod
    def autocast(*a, **k): raise NotImplementedError("torch stub")
amp = _Amp()
def no_grad(): raise NotImplementedError("torch stub")
float32 = "float32"
from . import nn, optim, distributions
