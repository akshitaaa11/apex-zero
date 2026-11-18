"""Enforce the import rules of AGENTS.md §2 by parsing every module under ``src/`` with ``ast``.

Rules checked:

1. **Layer table.** A module may import ``apex_zero`` modules from its own layer or a lower one.
   L3 (``sim``, ``qss``) may use only ``track`` from L2 ("track only, read-only").
2. **L0 third-party allowlist.** L0 (``config``, ``utils``, ``errors``, the root package) may
   import only the stdlib, ``numpy`` and ``pydantic``; ``structlog`` is allowed in
   ``utils.logging`` only (P0.5 puts structured logging there).
3. **No torch in the reference sim.** ``sim`` must not import ``torch`` except ``sim.batched``.
4. **Network libraries stay in data.** ``fastf1`` and ``httpx`` only inside ``data``.

Every import statement counts, including function-local and ``TYPE_CHECKING`` imports.
Dynamic imports (``importlib.import_module("...")``) are not detected.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT_PACKAGE = "apex_zero"
SRC_DIR = Path(__file__).resolve().parents[2] / "src"

#: Top-level component (first name below ``apex_zero``; ``""`` = root package) -> layer.
LAYERS: dict[str, int] = {
    "": 0,
    "errors": 0,
    "config": 0,
    "utils": 0,
    "data": 1,
    "track": 2,
    "ghosts": 2,
    "sim": 3,
    "qss": 3,
    "env": 4,
    "rl": 5,
    "analysis": 6,
    "export": 6,
    "cli": 7,
}

#: Components of a lower layer that a layer may *not* use despite the layer order.
#: L3 may use L2 for ``track`` only.
FORBIDDEN_LOWER: dict[int, frozenset[str]] = {3: frozenset({"ghosts"})}

L0_THIRD_PARTY: frozenset[str] = frozenset({"numpy", "pydantic"})
#: Extra third-party imports allowed for specific L0 modules.
L0_EXCEPTIONS: dict[str, frozenset[str]] = {
    f"{ROOT_PACKAGE}.utils.logging": frozenset({"structlog"})
}

NETWORK_LIBS: frozenset[str] = frozenset({"fastf1", "httpx"})


@dataclass(frozen=True, slots=True)
class Violation:
    """A forbidden import: ``module`` imports ``imported`` at ``path:line``."""

    path: Path
    line: int
    module: str
    imported: str
    rule: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.module} imports {self.imported} ({self.rule})"


def module_name(path: Path, src_dir: Path) -> str:
    """Dotted module name for a file under ``src_dir`` (``pkg/__init__.py`` -> ``pkg``)."""
    parts = list(path.relative_to(src_dir).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def imported_modules(path: Path, module: str) -> list[tuple[int, str]]:
    """All absolute module names imported by a file, with line numbers.

    ``from pkg import name`` yields ``pkg.name`` (it may be a submodule); relative imports are
    resolved against the file's package.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_parts = package.split(".")
                if node.level > 1:
                    base_parts = base_parts[: -(node.level - 1)]
                base = ".".join([*base_parts, node.module] if node.module else base_parts)
            else:
                base = node.module or ""
            for alias in node.names:
                found.append((node.lineno, base if alias.name == "*" else f"{base}.{alias.name}"))
    return found


def component(module: str) -> str:
    """First name below the root package (``apex_zero.sim.batched.x`` -> ``sim``)."""
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else ""


def rules_broken(module: str, imported: str) -> list[str]:
    """Names of the rules that ``module`` importing ``imported`` breaks (empty if legal)."""
    comp = component(module)
    layer = LAYERS[comp]
    top = imported.split(".")[0]
    broken: list[str] = []
    if top == ROOT_PACKAGE:
        # `from apex_zero import x` resolves to `apex_zero.x`. A name that is not a known
        # component (e.g. `__version__`) is re-exported by the root package, which is L0.
        target = component(imported)
        target = target if target in LAYERS else ""
        target_layer = LAYERS[target]
        if target_layer > layer:
            broken.append(f"L{layer} {comp or 'root'} may not import L{target_layer} {target}")
        elif target in FORBIDDEN_LOWER.get(layer, frozenset()):
            broken.append(f"L{layer} may not import {target}")
        return broken
    if top in sys.stdlib_module_names or top == "__future__":
        return broken
    if layer == 0 and top not in L0_THIRD_PARTY | L0_EXCEPTIONS.get(module, frozenset()):
        broken.append(f"L0 may only use stdlib/{'/'.join(sorted(L0_THIRD_PARTY))}")
    if top == "torch" and comp == "sim" and not module.startswith(f"{ROOT_PACKAGE}.sim.batched"):
        broken.append("torch is only allowed in sim.batched within sim")
    if top in NETWORK_LIBS and comp != "data":
        broken.append(f"{top} may only be imported inside data")
    return broken


def check_tree(src_dir: Path) -> list[Violation]:
    """Return every import-rule violation in ``src_dir/apex_zero``."""
    violations: list[Violation] = []
    for path in sorted((src_dir / ROOT_PACKAGE).rglob("*.py")):
        module = module_name(path, src_dir)
        comp = component(module)
        if comp not in LAYERS:
            violations.append(Violation(path, 0, module, "-", f"unknown component {comp!r}"))
            continue
        for line, imported in imported_modules(path, module):
            violations.extend(
                Violation(path, line, module, imported, rule)
                for rule in rules_broken(module, imported)
            )
    return violations


def test_src_respects_layering() -> None:
    violations = check_tree(SRC_DIR)
    assert not violations, "Layering violations:\n" + "\n".join(map(str, violations))


def test_every_package_is_in_layer_table() -> None:
    packages = {p.name for p in (SRC_DIR / ROOT_PACKAGE).iterdir() if (p / "__init__.py").exists()}
    modules = {p.stem for p in (SRC_DIR / ROOT_PACKAGE).glob("*.py") if p.stem != "__init__"}
    assert packages | modules <= set(LAYERS), "add new top-level packages to LAYERS"


# ------------------------------------------------------------------------------------------
# The checker must actually catch violations (gate: "fails when a forbidden import is added").
# ------------------------------------------------------------------------------------------


def _make_tree(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        f = tmp_path / ROOT_PACKAGE / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text, encoding="utf-8")
    for d in [tmp_path / ROOT_PACKAGE, *(tmp_path / ROOT_PACKAGE).rglob("*")]:
        if d.is_dir() and not (d / "__init__.py").exists():
            (d / "__init__.py").write_text("", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    ("rel", "source", "rule_fragment"),
    [
        ("track/x.py", "from apex_zero.sim import vehicle\n", "may not import L3"),
        ("data/x.py", "import apex_zero.rl.ppo\n", "may not import L5"),
        ("utils/x.py", "from apex_zero.data import cache\n", "may not import L1"),
        ("track/x.py", "from ..env import racing_env\n", "may not import L4"),
        ("sim/x.py", "from apex_zero.ghosts import ghost\n", "may not import ghosts"),
        ("sim/reference.py", "import torch\n", "torch is only allowed"),
        ("sim/x.py", "def f():\n    from torch import nn\n", "torch is only allowed"),
        ("track/x.py", "import fastf1\n", "fastf1 may only"),
        ("analysis/x.py", "from httpx import Client\n", "httpx may only"),
        ("utils/x.py", "import pandas\n", "L0 may only use"),
        ("config/x.py", "import structlog\n", "L0 may only use"),
        ("newpkg/x.py", "import os\n", "unknown component"),
    ],
)
def test_checker_detects_violation(
    tmp_path: Path, rel: str, source: str, rule_fragment: str
) -> None:
    violations = check_tree(_make_tree(tmp_path, {rel: source}))
    assert any(rule_fragment in v.rule for v in violations), violations


@pytest.mark.parametrize(
    ("rel", "source"),
    [
        ("sim/batched/x.py", "import torch\n"),
        ("rl/x.py", "import torch\nfrom apex_zero.env import vec_env\n"),
        ("data/x.py", "import fastf1\nimport httpx\nfrom apex_zero.utils import units\n"),
        ("ghosts/x.py", "from apex_zero.track import frenet\n"),
        ("qss/x.py", "from apex_zero.sim import params\nfrom apex_zero.track import asset\n"),
        ("track/x.py", "from . import frenet\nfrom ..utils import units\n"),
        ("utils/logging.py", "import structlog\nimport numpy\n"),
        ("cli.py", "import typer\nfrom apex_zero.export import exporter\n"),
        ("errors.py", "from apex_zero import __version__\n"),
    ],
)
def test_checker_allows_legal_imports(tmp_path: Path, rel: str, source: str) -> None:
    assert check_tree(_make_tree(tmp_path, {rel: source})) == []
