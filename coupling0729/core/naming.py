"""Canonical naming conventions for layers, nodes, and components."""

from __future__ import annotations

LAYER_SEP = ":"


def node_col(layer: str, node: str) -> str:
    """Return the canonical column name for a node in a layer."""
    return f"{layer}{LAYER_SEP}{node}"


def split_col(col: str) -> tuple[str, str]:
    """Split a canonical column name back into ``(layer, node)``."""
    layer, _, node = str(col).partition(LAYER_SEP)
    if not node:
        raise ValueError(f"not a canonical node column: {col!r}")
    return layer, node


def layer_of(col: str) -> str:
    return split_col(col)[0]


def cols_in_layer(cols, layer: str) -> list[str]:
    return [c for c in cols if layer_of(c) == layer]
