"""Shared geography for the power and transport layers.

A spatially correlated hazard is only meaningful across layers if both layers
live in the *same* coordinate frame. Otherwise the hazard damages them
independently and geographic interdependence -- one of the standard
interdependency types -- is silently discarded.

Sioux Falls ships real longitude/latitude node coordinates. IEEE33 ships none,
so feeder bus positions are synthesised: the feeder graph is laid out in the
plane and then affinely mapped into the road network's bounding box. This is a
*plausible, reproducible, synthetic* geography, not a real feeder, and results
that depend on it should be reported as such.

All coordinates are returned in kilometres on a local equirectangular
projection about the road network's centroid, so hazard parameters such as
attenuation length and correlation length are in kilometres.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

EARTH_RADIUS_KM = 6371.0


def project_to_km(
    lonlat: Mapping[str, tuple[float, float]],
    origin: tuple[float, float] | None = None,
) -> dict[str, tuple[float, float]]:
    """Equirectangular projection of lon/lat degrees to kilometres."""
    if not lonlat:
        return {}
    lons = np.array([v[0] for v in lonlat.values()], dtype=float)
    lats = np.array([v[1] for v in lonlat.values()], dtype=float)
    lon0, lat0 = origin if origin is not None else (float(lons.mean()), float(lats.mean()))
    scale = math.cos(math.radians(lat0))
    out = {}
    for node, (lon, lat) in lonlat.items():
        x = math.radians(float(lon) - lon0) * EARTH_RADIUS_KM * scale
        y = math.radians(float(lat) - lat0) * EARTH_RADIUS_KM
        out[str(node)] = (x, y)
    return out


def bounding_box(positions: Mapping[str, tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    return (min(xs), max(xs), min(ys), max(ys))


def embed_feeder(
    edges: Sequence[tuple[str, str]],
    nodes: Sequence[str],
    target_box: tuple[float, float, float, float],
    margin: float = 0.12,
    seed: int = 20260729,
) -> dict[str, tuple[float, float]]:
    """Lay out a feeder graph and affinely map it into ``target_box``.

    Uses a Kamada-Kawai layout when networkx is available -- it respects graph
    distances, so electrically adjacent buses end up spatially near each other,
    which is what makes a correlated hazard damage feeder neighbourhoods
    together. Falls back to a deterministic spiral otherwise.
    """
    nodes = [str(n) for n in nodes]
    layout = _graph_layout(edges, nodes, seed)

    xs = np.array([layout[n][0] for n in nodes], dtype=float)
    ys = np.array([layout[n][1] for n in nodes], dtype=float)
    x_min, x_max, y_min, y_max = target_box
    pad_x = margin * (x_max - x_min)
    pad_y = margin * (y_max - y_min)

    return {
        node: (
            _rescale(x, xs.min(), xs.max(), x_min + pad_x, x_max - pad_x),
            _rescale(y, ys.min(), ys.max(), y_min + pad_y, y_max - pad_y),
        )
        for node, x, y in zip(nodes, xs, ys)
    }


def _graph_layout(
    edges: Sequence[tuple[str, str]], nodes: Sequence[str], seed: int
) -> dict[str, tuple[float, float]]:
    try:
        import networkx as nx

        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from((str(a), str(b)) for a, b in edges)
        # Kamada-Kawai needs a connected graph; fall back per component.
        if nx.number_connected_components(graph) == 1:
            raw = nx.kamada_kawai_layout(graph)
        else:
            raw = nx.spring_layout(graph, seed=seed)
        return {str(k): (float(v[0]), float(v[1])) for k, v in raw.items()}
    except Exception:
        return _spiral_layout(nodes)


def _spiral_layout(nodes: Sequence[str]) -> dict[str, tuple[float, float]]:
    out = {}
    for i, node in enumerate(nodes):
        angle = 0.55 * i
        radius = 0.35 * math.sqrt(i + 1)
        out[str(node)] = (radius * math.cos(angle), radius * math.sin(angle))
    return out


def _rescale(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if hi - lo < 1e-12:
        return float(0.5 * (out_lo + out_hi))
    return float(out_lo + (value - lo) / (hi - lo) * (out_hi - out_lo))
