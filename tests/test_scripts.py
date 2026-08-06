"""Every runnable script must survive being pasted into an interactive console.

VS Code's "run in interactive window" does not execute a file — it **pastes the
source into a cell**. That namespace differs from a real script run in two ways
that break code which looks fine from a terminal:

* there is no ``__file__``, so anything anchoring paths to the script's own
  location raises ``NameError``;
* ``sys.argv`` belongs to the kernel, and carries ``--f=...kernel.json``, which
  an over-eager ``argparse`` will either reject or prefix-match onto a real
  option.

Both have bitten this repository. These tests reproduce that namespace exactly —
``__name__`` is ``"__main__"`` so the guarded body runs, ``__file__`` is absent,
and the kernel's own arguments are prepended to ``sys.argv`` — rather than
checking for the patterns statically, because it is the behaviour that matters.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _close_figures():
    """Each script opens several pyplot figures; without this they accumulate.

    Past twenty, matplotlib warns — and the warning lands in whichever test
    happens to cross the threshold rather than the one that caused it.
    """
    yield
    plt.close("all")

#: What a Jupyter/VS Code kernel actually leaves in ``sys.argv``.
KERNEL_ARGV = [
    "ipykernel_launcher.py",
    "--f=C:/Users/example/AppData/Roaming/jupyter/runtime/kernel-v3a1b2c3.json",
]

#: Every script a user might run, with arguments that keep the test quick.
#: ``None`` means the script takes no arguments at all.
SCRIPTS: dict[str, list[str]] = {
    "examples/01_quickstart.py": [],
    "examples/02_gym_loop.py": [],
    "examples/03_customize.py": [],
    "examples/04_figures.py": ["--episodes", "12"],
    "examples/05_animation.py": ["--contexts", "2"],
    "docs/make_assets.py": ["--episodes", "12"],
}

WRITES_FIGURES = {"examples/04_figures.py", "examples/05_animation.py",
                  "docs/make_assets.py"}


def _paste_and_run(script: Path, argv: list[str], cwd: Path) -> dict:
    """Execute ``script``'s source the way an interactive cell would.

    Compiled with a filename for readable tracebacks, but executed in a namespace
    that deliberately has **no** ``__file__`` — which is what a pasted cell
    gives you.
    """
    source = script.read_text(encoding="utf-8")
    namespace: dict = {"__name__": "__main__"}
    assert "__file__" not in namespace

    original_argv, original_cwd = sys.argv, Path.cwd()
    try:
        sys.argv = [*KERNEL_ARGV, *argv]
        import os

        os.chdir(cwd)
        exec(compile(source, str(script), "exec"), namespace)  # noqa: S102
    finally:
        sys.argv = original_argv
        import os

        os.chdir(original_cwd)
    return namespace


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_runs_when_pasted_into_a_cell(name, tmp_path, monkeypatch):
    """The namespace has no __file__ and the kernel's own argv. Must still run."""
    monkeypatch.setenv("MPLBACKEND", "Agg")
    args = list(SCRIPTS[name])
    if name in WRITES_FIGURES:
        args += ["--out", str(tmp_path / "figures")]
    _paste_and_run(ROOT / name, args, cwd=ROOT)


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_runs_as_a_plain_script(name, tmp_path, monkeypatch):
    """The ordinary path still works: real __file__, real __main__."""
    monkeypatch.setenv("MPLBACKEND", "Agg")
    args = list(SCRIPTS[name])
    if name in WRITES_FIGURES:
        args += ["--out", str(tmp_path / "figures")]
    monkeypatch.setattr(sys, "argv", [str(ROOT / name), *args])
    runpy.run_path(str(ROOT / name), run_name="__main__")


def test_a_script_that_needs_its_own_location_still_finds_it(tmp_path, monkeypatch):
    """``docs/make_assets.py`` defaults its output next to itself.

    Pasted into a cell it cannot use ``__file__`` to do that, so it searches
    upward from the working directory instead. Run from a *subdirectory* to prove
    the search actually happens rather than the answer being the cwd by luck.
    """
    monkeypatch.setenv("MPLBACKEND", "Agg")
    namespace = _paste_and_run(
        ROOT / "docs/make_assets.py",
        ["--episodes", "12", "--out", str(tmp_path / "figures")],
        cwd=ROOT / "src" / "coggrid",
    )
    assert namespace["ROOT"] == ROOT
