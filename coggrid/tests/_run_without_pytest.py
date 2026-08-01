"""Tiny pytest stand-in so the suite can be verified without network access.

Supports only what tests/ actually uses: module-scoped fixtures, parametrize,
raises, warns, approx and tmp_path. Not a general implementation — the real
`pytest` is the supported runner.
"""

from __future__ import annotations

import contextlib
import inspect
import sys
import tempfile
import traceback
import warnings
from pathlib import Path


class _Approx:
    def __init__(self, value, rel=1e-6, abs=1e-12):
        self.value, self.rel, self.abs = value, rel, abs

    def __eq__(self, other):
        return abs(other - self.value) <= max(self.abs, self.rel * abs(self.value))

    def __repr__(self):
        return f"approx({self.value})"


class _Raises:
    def __init__(self, exc, match=None):
        self.exc, self.match = exc, match

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError(f"expected {self.exc.__name__}, nothing raised")
        if not issubclass(et, self.exc):
            return False
        if self.match:
            import re

            if not re.search(self.match, str(ev)):
                raise AssertionError(f"{ev!r} does not match {self.match!r}")
        return True


class _Warns:
    def __init__(self, category, match=None):
        self.category, self.match = category, match

    def __enter__(self):
        self.ctx = warnings.catch_warnings(record=True)
        self.log = self.ctx.__enter__()
        warnings.simplefilter("always")
        return self.log

    def __exit__(self, et, ev, tb):
        self.ctx.__exit__(et, ev, tb)
        if et is not None:
            return False
        hits = [w for w in self.log if issubclass(w.category, self.category)]
        if not hits:
            raise AssertionError(f"no {self.category.__name__} raised")
        if self.match:
            import re

            if not any(re.search(self.match, str(w.message)) for w in hits):
                raise AssertionError(f"no warning matching {self.match!r}")
        return False


class _Mark:
    @staticmethod
    def parametrize(argnames, argvalues):
        names = [n.strip() for n in argnames.split(",")]

        def deco(fn):
            cases = getattr(fn, "_cases", [{}])
            expanded = []
            for case in cases:
                for values in argvalues:
                    values = values if isinstance(values, tuple) else (values,)
                    expanded.append({**case, **dict(zip(names, values))})
            fn._cases = expanded
            return fn

        return deco


class _Pytest:
    mark = _Mark()
    approx = staticmethod(lambda v, **k: _Approx(v, **k))
    raises = _Raises
    warns = _Warns

    @staticmethod
    def fixture(*a, **k):
        def deco(fn):
            fn._is_fixture = True
            return fn

        return deco(a[0]) if a and callable(a[0]) else deco

    @staticmethod
    def skip(reason=""):
        raise _Skip(reason)


class _Skip(Exception):
    pass


sys.modules["pytest"] = _Pytest  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))


def run_module(name: str) -> tuple[int, int, int]:
    module = __import__(name)
    fixtures = {
        n: f for n, f in vars(module).items() if getattr(f, "_is_fixture", False)
    }
    cache: dict[str, object] = {}
    tmp = Path(tempfile.mkdtemp())

    def resolve(param: str):
        if param == "tmp_path":
            return tmp
        if param in cache:
            return cache[param]
        fn = fixtures[param]
        args = [resolve(p) for p in inspect.signature(fn).parameters]
        cache[param] = fn(*args)
        return cache[param]

    tests: list[tuple[str, object, object]] = []
    for attr, obj in vars(module).items():
        if attr.startswith("Test") and inspect.isclass(obj):
            for meth, fn in vars(obj).items():
                if meth.startswith("test_"):
                    tests.append((f"{attr}::{meth}", fn, obj()))
        elif attr.startswith("test_") and inspect.isfunction(obj):
            tests.append((attr, obj, None))

    passed = failed = skipped = 0
    for label, fn, instance in tests:
        for case in getattr(fn, "_cases", [{}]):
            params = list(inspect.signature(fn).parameters)
            args = []
            for p in params:
                if p == "self":
                    args.append(instance)
                elif p in case:
                    args.append(case[p])
                else:
                    args.append(resolve(p))
            tag = f"{label}{case if case else ''}"
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fn(*args)
                passed += 1
            except _Skip:
                skipped += 1
                print(f"  SKIP {tag}")
            except Exception:
                failed += 1
                print(f"  FAIL {tag}")
                print("    " + traceback.format_exc().replace("\n", "\n    ")[:900])
    return passed, failed, skipped


if __name__ == "__main__":
    total_p = total_f = total_s = 0
    for mod in ("test_environment", "test_parity"):
        print(f"\n{mod}")
        with contextlib.suppress(SystemExit):
            p, f, s = run_module(mod)
        total_p += p
        total_f += f
        total_s += s
        print(f"  {p} passed, {f} failed, {s} skipped")
    print(f"\nTOTAL: {total_p} passed, {total_f} failed, {total_s} skipped")
    sys.exit(1 if total_f else 0)
