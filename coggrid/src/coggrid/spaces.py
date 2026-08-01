"""Gymnasium spaces when available, minimal stand-ins when not.

``gymnasium`` is an optional dependency. If it is installed, the environment
subclasses ``gymnasium.Env`` and uses the real space objects, so it drops
straight into any Gymnasium-compatible training loop. If it is not, the same
environment still works — it exposes ``reset``/``step`` with identical
signatures and lightweight space objects carrying ``shape``, ``dtype``,
``sample()`` and ``contains()``.

The point is that ``pip install coggrid`` gets you a working
environment with numpy alone, and ``pip install 'coggrid[gym]'``
gets you registry integration and wrapper compatibility. Nobody has to install
an RL stack to look at the data.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["HAS_GYMNASIUM", "EnvBase", "Box", "Dict", "Discrete", "MultiBinary"]

try:  # pragma: no cover - depends on the user's install
    import gymnasium as _gym

    HAS_GYMNASIUM = True
    Box = _gym.spaces.Box
    Dict = _gym.spaces.Dict
    Discrete = _gym.spaces.Discrete
    MultiBinary = _gym.spaces.MultiBinary
    EnvBase: Any = _gym.Env

except ModuleNotFoundError:  # pragma: no cover - depends on the user's install
    HAS_GYMNASIUM = False

    class EnvBase:  # type: ignore[no-redef]
        """Stand-in for ``gymnasium.Env``.

        Provides the attributes Gymnasium users expect so that code written
        against the real thing behaves the same here.
        """

        metadata: dict[str, Any] = {"render_modes": []}
        render_mode: str | None = None
        spec: Any = None

        @property
        def unwrapped(self) -> "EnvBase":
            return self

        def close(self) -> None:
            return None

        def __enter__(self) -> "EnvBase":
            return self

        def __exit__(self, *exc: object) -> bool:
            self.close()
            return False

    class _Space:
        dtype: np.dtype
        shape: tuple[int, ...]

        def __init__(self, seed: int | None = None) -> None:
            self._rng = np.random.default_rng(seed)

        def seed(self, seed: int | None = None) -> None:
            self._rng = np.random.default_rng(seed)

        def sample(self) -> Any:  # pragma: no cover - trivial
            raise NotImplementedError

        def contains(self, x: Any) -> bool:  # pragma: no cover - trivial
            raise NotImplementedError

        def __contains__(self, x: Any) -> bool:
            return self.contains(x)

    class Discrete(_Space):  # type: ignore[no-redef]
        """``{0, 1, ..., n - 1}``."""

        def __init__(self, n: int, seed: int | None = None) -> None:
            super().__init__(seed)
            self.n = int(n)
            self.shape = ()
            self.dtype = np.dtype(np.int64)

        def sample(self) -> int:
            return int(self._rng.integers(0, self.n))

        def contains(self, x: Any) -> bool:
            return bool(np.issubdtype(type(x), np.integer) and 0 <= int(x) < self.n)

        def __repr__(self) -> str:
            return f"Discrete({self.n})"

    class MultiBinary(_Space):  # type: ignore[no-redef]
        """A vector of ``n`` binary values."""

        def __init__(self, n: int, seed: int | None = None) -> None:
            super().__init__(seed)
            self.n = int(n)
            self.shape = (self.n,)
            self.dtype = np.dtype(np.int8)

        def sample(self) -> np.ndarray:
            return self._rng.integers(0, 2, size=self.shape).astype(np.int8)

        def contains(self, x: Any) -> bool:
            x = np.asarray(x)
            return x.shape == self.shape and bool(np.isin(x, (0, 1)).all())

        def __repr__(self) -> str:
            return f"MultiBinary({self.n})"

    class Box(_Space):  # type: ignore[no-redef]
        """A box-constrained array."""

        def __init__(
            self,
            low: Any,
            high: Any,
            shape: tuple[int, ...],
            dtype: Any = np.float32,
            seed: int | None = None,
        ) -> None:
            super().__init__(seed)
            self.shape = tuple(shape)
            self.dtype = np.dtype(dtype)
            self.low = np.broadcast_to(np.asarray(low, self.dtype), self.shape)
            self.high = np.broadcast_to(np.asarray(high, self.dtype), self.shape)

        def sample(self) -> np.ndarray:
            if np.issubdtype(self.dtype, np.integer):
                return self._rng.integers(
                    self.low, np.asarray(self.high) + 1, size=self.shape
                ).astype(self.dtype)
            return self._rng.uniform(self.low, self.high, size=self.shape).astype(
                self.dtype
            )

        def contains(self, x: Any) -> bool:
            x = np.asarray(x)
            return (
                x.shape == self.shape
                and bool((x >= self.low).all())
                and bool((x <= self.high).all())
            )

        def __repr__(self) -> str:
            return f"Box({self.shape}, {self.dtype})"

    class Dict(_Space):  # type: ignore[no-redef]
        """A dictionary of named subspaces."""

        def __init__(self, spaces: dict[str, Any], seed: int | None = None) -> None:
            super().__init__(seed)
            self.spaces = dict(spaces)
            self.shape = ()
            self.dtype = np.dtype(object)

        def sample(self) -> dict[str, Any]:
            return {k: v.sample() for k, v in self.spaces.items()}

        def contains(self, x: Any) -> bool:
            return isinstance(x, dict) and all(
                k in x and v.contains(x[k]) for k, v in self.spaces.items()
            )

        def __getitem__(self, key: str) -> Any:
            return self.spaces[key]

        def keys(self):  # pragma: no cover - trivial
            return self.spaces.keys()

        def items(self):  # pragma: no cover - trivial
            return self.spaces.items()

        def __repr__(self) -> str:
            inner = ", ".join(f"{k!r}: {v!r}" for k, v in self.spaces.items())
            return f"Dict({inner})"
