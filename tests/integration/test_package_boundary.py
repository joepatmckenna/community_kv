from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).parents[2]
PACKAGE = ROOT / "community_kv"


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_package_does_not_import_external_workflow_modules() -> None:
    forbidden = {
        "benchmarks",
        "dev",
        "evals",
        "tests",
    }
    violations: list[str] = []
    for path in PACKAGE.rglob("*.py"):
        for name in _imports(path):
            root = name.split(".", 1)[0]
            if root in forbidden:
                violations.append(f"{path.relative_to(ROOT)} imports {name}")
    assert not violations, "\n".join(violations)


def test_graph_package_does_not_depend_on_attention_or_request_runtime() -> None:
    forbidden = ("community_kv.attention", "community_kv.runtime")
    violations = [
        f"{path.relative_to(ROOT)} imports {name}"
        for path in (PACKAGE / "graph").rglob("*.py")
        for name in _imports(path)
        if name.startswith(forbidden)
    ]
    assert not violations, "\n".join(violations)


def test_attention_runtime_dependency_is_type_checking_only() -> None:
    violations = [
        f"{path.relative_to(ROOT)} imports {name}"
        for path in (PACKAGE / "attention").rglob("*.py")
        if path.name != "adapter.py"
        for name in _imports(path)
        if name.startswith("community_kv.runtime")
    ]
    assert not violations, "\n".join(violations)


def test_package_contains_no_harness_or_evaluation_modules() -> None:
    forbidden_parts = {"benchmark", "eval", "oracle", "profile", "sweep"}
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in PACKAGE.rglob("*.py")
        if any(part in path.stem for part in forbidden_parts)
    ]
    assert offenders == []


def test_expected_domain_layout_exists() -> None:
    expected = {
        "attention/adapter.py",
        "attention/cache.py",
        "attention/decode.py",
        "attention/flash_attention.py",
        "attention/kernels/packed_segments.py",
        "graph/partition.py",
        "graph/runtime.py",
        "graph/state.py",
        "graph/leiden/algorithm.py",
        "runtime.py",
    }
    actual = {
        path.relative_to(PACKAGE).as_posix()
        for path in PACKAGE.rglob("*.py")
    }
    assert expected <= actual
    assert "attention/selection.py" not in actual
