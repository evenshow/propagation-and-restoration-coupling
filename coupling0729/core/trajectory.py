"""Simulation output: the two indicator curves plus provenance."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .scenario import Scenario


@dataclass
class TrajectoryResult:
    """Two normalised performance curves for one run.

    ``operational`` is the service indicator (what users experience) and
    carries the three headline metrics. ``infrastructure`` is the physical
    integrity indicator, retained as a mechanism diagnostic rather than a
    parallel set of headline results.
    """

    operational: pd.DataFrame
    infrastructure: pd.DataFrame
    scenario: Scenario
    baseline: dict = field(default_factory=dict)
    damage_summary: dict = field(default_factory=dict)
    coupling_summary: dict = field(default_factory=dict)
    events: list = field(default_factory=list)

    @property
    def arm(self) -> str:
        return self.scenario.arm.name

    def system_curve(self, kind: str = "operational", weights: dict | None = None) -> pd.Series:
        """Weighted mean across columns, forming the system-level curve."""
        frame = getattr(self, kind)
        if frame.empty:
            raise ValueError(f"{kind} trajectory is empty")
        if weights is None:
            return frame.mean(axis=1)
        w = pd.Series({c: float(weights.get(c, 0.0)) for c in frame.columns})
        total = float(w.sum())
        if total <= 0:
            raise ValueError("system weights must have a positive total")
        return (frame * w).sum(axis=1) / total

    # -- persistence -----------------------------------------------------
    def save(self, root: str | Path, key: str | None = None) -> Path:
        key = key or self.scenario.scenario_hash()
        directory = Path(root) / key
        directory.mkdir(parents=True, exist_ok=True)
        self.operational.to_parquet(directory / "operational.parquet")
        self.infrastructure.to_parquet(directory / "infrastructure.parquet")
        payload = {
            "scenario": self.scenario.to_dict(),
            "baseline": self.baseline,
            "damage_summary": self.damage_summary,
            "coupling_summary": self.coupling_summary,
            "events": self.events,
        }
        (directory / "meta.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return directory

    @staticmethod
    def exists(root: str | Path, key: str) -> bool:
        directory = Path(root) / key
        return (directory / "operational.parquet").exists() and (directory / "meta.json").exists()
