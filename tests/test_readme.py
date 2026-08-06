"""The README has to actually work.

Two things rot independently: the code blocks, and the argument tables. Both are
checked here against the running library rather than by reading.

Blocks are executed in one shared namespace, in order, exactly as a reader would
follow them — so a block that uses ``batch`` gets the ``batch`` an earlier block
defined. Sizes are shrunk first, because the README quotes numbers a reader would
want (2000 episodes) and a test does not need them.
"""

from __future__ import annotations

import dataclasses
import doctest
import gc
import inspect
import re
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from coggrid import (  # noqa: E402
    CogGridConfig,
    CogGridEnv,
    World,
)

README = Path(__file__).resolve().parent.parent / "README.md"
TEXT = README.read_text(encoding="utf-8")

# One README block builds an animation to show that it plays inline, and never
# renders it — which is the point of that example, not an oversight.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Animation was deleted without rendering anything:UserWarning"
)


def _same_default(shown: str, actual: object) -> bool:
    """Whether a README cell states ``actual``, ignoring quoting style.

    The README writes string defaults as Python literals (``"held_out"``) while
    ``repr`` prefers single quotes, and both are correct.
    """
    normalise = str.maketrans("", "", "`'\"")
    expected = "None" if actual is None else str(actual)
    return shown.translate(normalise) == expected.translate(normalise)

#: Blocks that are deliberately illustrative rather than runnable.
SKETCHES = ("def codebook_embeddings",)

#: Shrink the worked example so the suite stays quick. Anything here must keep
#: the block's *meaning*: only sizes change, never the API being demonstrated.
SHRINK = {
    "sample_episodes(4096)": "sample_episodes(20)",
    "sample_episodes(2000)": "sample_episodes(40)",
    "sample_episodes(1000,": "sample_episodes(20,",
    "n_vars=500": "n_vars=60",
}


def _python_blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", TEXT, re.S)


def _table_rows(header: str) -> dict[str, str]:
    """Parse the markdown table that follows ``header`` into ``{name: cell}``.

    Returns ``{}`` when the table is absent. These tests check that what the
    README *does* say is true; they must not force it to keep saying anything.
    """
    if header not in TEXT:
        return {}
    body = TEXT.split(header, 1)[1]
    rows = {}
    for line in body.splitlines():
        if not line.startswith("|"):
            if rows:  # table finished
                break
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        name = re.fullmatch(r"`([a-z_]+)`", cells[0])
        if name:
            rows[name.group(1)] = cells[1].strip("`")
    return rows


# --------------------------------------------------------------------------- #
# the code blocks run
# --------------------------------------------------------------------------- #
def test_every_python_block_executes():
    """Run the README top to bottom in one namespace, as a reader would."""
    # The gym-loop block calls a reader-supplied policy. Supplying a trivial one
    # is better than skipping the block: the loop itself still has to run.
    namespace: dict = {"my_agent": lambda obs: 0}
    ran = 0
    for block in _python_blocks():
        if any(s in block for s in SKETCHES) or block.lstrip().startswith(">>>"):
            continue
        for old, new in SHRINK.items():
            block = block.replace(old, new)
        exec(compile(block, "README.md", "exec"), namespace)  # noqa: S102
        ran += 1
    assert ran, "no runnable python blocks found — did the extraction break?"

    # Drop the unrendered animation here rather than at interpreter shutdown,
    # where matplotlib's warning would land outside this module's filter.
    namespace.clear()
    gc.collect()
    plt.close("all")


def test_doctest_blocks_produce_the_output_shown():
    """``>>>`` blocks state their own output, so run them as doctests."""
    blocks = [b for b in _python_blocks() if b.lstrip().startswith(">>>")]
    if not blocks:
        pytest.skip("README has no doctest-style block")
    for block in blocks:
        results = doctest.DocTestRunner()
        parser = doctest.DocTestParser()
        test = parser.get_doctest(block, {"CogGridConfig": CogGridConfig},
                                  "README", "README.md", 0)
        results.run(test, clear_globs=False)
        assert results.failures == 0, f"doctest in README failed:\n{block}"


# --------------------------------------------------------------------------- #
# the tables describe the real API
# --------------------------------------------------------------------------- #
def test_config_table_matches_the_dataclass():
    """Every field documented, every documented field real, defaults correct."""
    documented = _table_rows("| Field | Default | Meaning |")
    if not documented:
        pytest.skip("README has no config table")
    actual = {f.name: f.default for f in dataclasses.fields(CogGridConfig)}

    assert set(documented) == set(actual), (
        f"undocumented: {sorted(set(actual) - set(documented))}; "
        f"invented: {sorted(set(documented) - set(actual))}"
    )
    for name, shown in documented.items():
        assert _same_default(shown, actual[name]), (
            f"{name}: README says {shown!r}, actual default is {actual[name]!r}"
        )


def test_environment_table_matches_the_constructors():
    """Each documented argument exists on at least one env, with that default."""
    documented = _table_rows("| Argument | Default | Meaning |")
    if not documented:
        pytest.skip("README has no environment table")
    signatures = {"CogGridEnv": inspect.signature(CogGridEnv.__init__).parameters}
    for name, shown in documented.items():
        homes = [cls for cls, sig in signatures.items() if name in sig]
        assert homes, f"README documents {name!r}, which CogGridEnv does not accept"
        for cls in homes:
            default = signatures[cls][name].default
            if default is inspect.Parameter.empty:
                continue
            assert _same_default(shown, default), (
                f"{cls}.{name}: README says {shown!r}, actual default is {default!r}"
            )


def test_no_environment_argument_is_left_undocumented():
    documented = set(_table_rows("| Argument | Default | Meaning |"))
    if not documented:
        pytest.skip("README has no environment table")
    for cls in (CogGridEnv,):
        params = set(inspect.signature(cls.__init__).parameters) - {"self"}
        assert params <= documented, (
            f"{cls.__name__} accepts undocumented arguments: "
            f"{sorted(params - documented)}"
        )


# --------------------------------------------------------------------------- #
# specific claims the prose makes
# --------------------------------------------------------------------------- #
def test_world_argument_overrides_config():
    world = World(CogGridConfig(n_vars=60, seed=1))
    env = CogGridEnv(CogGridConfig(n_vars=444), world=world)
    assert env.cfg.n_vars == 60


@pytest.mark.parametrize("mode,expected", [("ansi", str), (None, type(None))])
def test_render_mode_behaves_as_documented(mode, expected):
    env = CogGridEnv(config=CogGridConfig(n_vars=60, seed=0), render_mode=mode)
    env.reset()
    assert isinstance(env.render(), expected)


def test_expose_likelihood_is_off_by_default():
    cfg = CogGridConfig(n_vars=60, seed=0)
    assert "rates" not in CogGridEnv(config=cfg).reset()[1]
    info = CogGridEnv(config=cfg, expose_likelihood=True).reset()[1]
    assert "rates" in info and "marginal_rates" in info


def test_every_referenced_image_exists():
    for src in re.findall(r'src="([^"]+)"', TEXT):
        assert (README.parent / src).exists(), f"README references missing {src}"


def test_every_relative_link_resolves():
    for target in re.findall(r"\]\((?!https?:|#)([^)]+)\)", TEXT):
        assert (README.parent / target).exists(), f"README links to missing {target}"
