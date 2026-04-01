from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPLEXITY_NODES = (
    ast.AsyncFor,
    ast.ExceptHandler,
    ast.For,
    ast.If,
    ast.IfExp,
    ast.Try,
    ast.While,
    ast.With,
    ast.AsyncWith,
)
NESTING_NODES = (
    ast.AsyncFor,
    ast.For,
    ast.If,
    ast.Try,
    ast.While,
    ast.With,
    ast.AsyncWith,
)


class ComplexityVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.complexity = 1

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, COMPLEXITY_NODES):
            self.complexity += 1
        elif isinstance(node, ast.BoolOp):
            self.complexity += max(0, len(node.values) - 1)
        elif isinstance(node, ast.comprehension):
            self.complexity += 1
        super().generic_visit(node)



def _parse_module(relative_path: str) -> ast.Module:
    return ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))



def _find_node(module: ast.Module, qualified_name: str) -> ast.AST:
    current: ast.AST = module
    for part in qualified_name.split("."):
        body = getattr(current, "body", None)
        if body is None:
            raise AssertionError(f"{qualified_name} not found")
        for child in body:
            if getattr(child, "name", None) == part:
                current = child
                break
        else:
            raise AssertionError(f"{qualified_name} not found")
    return current



def _measure_complexity(node: ast.AST) -> int:
    visitor = ComplexityVisitor()
    visitor.visit(node)
    return visitor.complexity



def _measure_max_nesting(node: ast.AST, depth: int = 0) -> int:
    max_depth = depth
    for child in ast.iter_child_nodes(node):
        child_depth = depth + 1 if isinstance(child, NESTING_NODES) else depth
        max_depth = max(max_depth, _measure_max_nesting(child, child_depth))
    return max_depth


@pytest.mark.parametrize(
    ("relative_path", "qualified_name", "max_complexity"),
    [
        ("tg_mem/memory/main.py", "Memory.add", 15),
        ("tg_mem/memory/main.py", "Memory._add_to_vector_store", 15),
        ("tg_mem/memory/main.py", "AsyncMemory.add", 15),
        ("tg_mem/memory/main.py", "AsyncMemory._add_to_vector_store", 15),
        ("tg_mem/memory/storage.py", "MySQLManager.upsert_memory_record", 15),
    ],
)
def test_target_functions_stay_under_complexity_threshold(
    relative_path: str,
    qualified_name: str,
    max_complexity: int,
) -> None:
    module = _parse_module(relative_path)
    node = _find_node(module, qualified_name)

    complexity = _measure_complexity(node)

    assert complexity <= max_complexity, (
        f"{relative_path}:{qualified_name} has complexity {complexity}, "
        f"expected <= {max_complexity}"
    )


@pytest.mark.parametrize(
    ("relative_path", "qualified_name", "max_nesting"),
    [
        ("tg_mem/memory/main.py", "Memory._add_to_vector_store", 5),
        ("tg_mem/memory/main.py", "AsyncMemory._add_to_vector_store", 5),
    ],
)
def test_target_functions_stay_under_nesting_threshold(
    relative_path: str,
    qualified_name: str,
    max_nesting: int,
) -> None:
    module = _parse_module(relative_path)
    node = _find_node(module, qualified_name)

    nesting = _measure_max_nesting(node)

    assert nesting <= max_nesting, (
        f"{relative_path}:{qualified_name} has nesting depth {nesting}, "
        f"expected <= {max_nesting}"
    )
