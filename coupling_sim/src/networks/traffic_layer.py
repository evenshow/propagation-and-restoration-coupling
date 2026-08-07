"""TNTP traffic-network parsing utilities.

The loader supports Sioux Falls and other datasets that follow the common
TransportationNetworks file naming convention:

    <Prefix>_net.tntp
    <Prefix>_trips.tntp
    <Prefix>_node.tntp   (optional)

The downstream user-equilibrium simulator works with the TrafficNetwork object,
so switching from Sioux Falls to Anaheim should only require a different data
directory and prefix once the TNTP files are available locally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

import numpy as np


@dataclass
class TrafficNetwork:
    n_nodes: int
    node_ids: list[str]
    fb: np.ndarray
    tb: np.ndarray
    cap: np.ndarray
    t0: np.ndarray
    bpr_b: np.ndarray
    bpr_p: np.ndarray
    od: np.ndarray
    coords: dict[str, tuple[float, float]] = field(default_factory=dict)
    name: str = "TNTP"

    @property
    def n_links(self) -> int:
        return len(self.fb)

    def pos(self, node_label: str) -> int:
        return self.node_ids.index(str(node_label))


def parse_tntp_net(path) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("<") or line.startswith("~"):
            continue
        parts = line.rstrip(";").split()
        if len(parts) < 8:
            continue
        rows.append(
            dict(
                fb=int(parts[0]),
                tb=int(parts[1]),
                cap=float(parts[2]),
                t0=float(parts[4]),
                b=float(parts[5]),
                p=float(parts[6]),
            )
        )
    return rows


def parse_tntp_trips(path, node_ids_or_n: int | list[str]) -> np.ndarray:
    """Parse OD demand into a node-indexed square matrix.

    Some TNTP datasets have fewer zones than physical nodes. The simulation
    graph, however, is indexed by physical node position. Passing node labels
    lets OD entries for zone labels land on the matching physical nodes while
    leaving non-zone nodes with zero OD demand.
    """

    if isinstance(node_ids_or_n, int):
        n_nodes = node_ids_or_n
        pos = {str(i + 1): i for i in range(n_nodes)}
    else:
        node_ids = [str(node) for node in node_ids_or_n]
        n_nodes = len(node_ids)
        pos = {node: idx for idx, node in enumerate(node_ids)}

    od = np.zeros((n_nodes, n_nodes))
    origin = None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("<"):
            continue
        match = re.match(r"Origin\s+(\d+)", line)
        if match:
            origin = match.group(1)
            continue
        if origin is None:
            continue
        for dest_s, dem_s in re.findall(r"(\d+)\s*:\s*([\d.]+)", line):
            if origin not in pos or dest_s not in pos:
                continue
            od[pos[origin], pos[dest_s]] = float(dem_s)
    return od


def parse_tntp_nodes(path) -> dict[str, tuple[float, float]]:
    coords = {}
    for line in Path(path).read_text().splitlines():
        parts = line.strip().rstrip(";").split()
        if len(parts) >= 3 and parts[0].isdigit():
            coords[parts[0]] = (float(parts[1]), float(parts[2]))
    return coords


def _detect_tntp_prefix(data_dir: Path) -> str:
    candidates = sorted(data_dir.glob("*_net.tntp"))
    if not candidates:
        raise FileNotFoundError(f"cannot find a *_net.tntp file in {data_dir}")
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise ValueError(f"multiple TNTP network files found in {data_dir}; pass prefix explicitly: {names}")
    return candidates[0].name.removesuffix("_net.tntp")


def load_tntp_network(data_dir, prefix: str | None = None, network_name: str | None = None) -> TrafficNetwork:
    data_path = Path(data_dir)
    prefix = prefix or _detect_tntp_prefix(data_path)
    net_file = data_path / f"{prefix}_net.tntp"
    trips_file = data_path / f"{prefix}_trips.tntp"
    if not trips_file.exists():
        raise FileNotFoundError(f"cannot find TNTP trips file at {trips_file}")

    rows = parse_tntp_net(net_file)
    if not rows:
        raise ValueError(f"no TNTP links parsed from {net_file}")

    labels = sorted({row["fb"] for row in rows} | {row["tb"] for row in rows})
    node_ids = [str(label) for label in labels]
    pos = {label: idx for idx, label in enumerate(labels)}
    n = len(labels)
    net = TrafficNetwork(
        n_nodes=n,
        node_ids=node_ids,
        fb=np.array([pos[row["fb"]] for row in rows], dtype=int),
        tb=np.array([pos[row["tb"]] for row in rows], dtype=int),
        cap=np.array([row["cap"] for row in rows]),
        t0=np.array([row["t0"] for row in rows]),
        bpr_b=np.array([row["b"] for row in rows]),
        bpr_p=np.array([row["p"] for row in rows]),
        od=parse_tntp_trips(trips_file, node_ids),
        name=network_name or prefix,
    )
    node_file = data_path / f"{prefix}_node.tntp"
    if node_file.exists():
        net.coords = parse_tntp_nodes(node_file)
    return net


def load_siouxfalls(data_dir) -> TrafficNetwork:
    return load_tntp_network(data_dir, prefix="SiouxFalls", network_name="Sioux Falls")
