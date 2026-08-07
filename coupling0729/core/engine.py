"""Phase-aware co-simulation engine with an explicit hazard window.

Timeline enforced by the engine::

      pre-event        Phase I          Phase II            Phase III
    ------------|----------------|------------------|--------------------->
               t_oe             t_ee          t_ee + t_mob
                 damage accrues   embargo: no repair    repair proceeds

The repair embargo is the structural change that makes the resilience
trapezoid appear and, more importantly, keeps the two headline mechanism
metrics uncontaminated:

* ``Lambda`` (depth) is read over ``[t_oe, t_ee + t_mob]``, a window in which no
  repair whatsoever takes place, so it is a pure propagation-phase quantity;
* ``T_rec`` is measured from ``t_ee``, an exogenous instant, so it captures both
  *how long until repair actually gets going* -- the channel that restoration
  coupling acts on most visibly -- and *how fast it then goes*.

Propagation reads source performance from the **previous** step. The one-step
lag removes any dependence on layer execution order, so results cannot be an
artefact of which layer happens to be stepped first.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .coupling import PROPAGATION, RESTORATION, CouplingOperator, build_operator
from .damage import DamageMapping
from .hazard import DamageState, LognormalFragility, RadialHazardField, sample_damage
from .naming import node_col
from .repair import ResourceConstrainedRepair
from .scenario import Scenario
from .trajectory import TrajectoryResult


@dataclass(frozen=True)
class CouplingSpec:
    """Wiring for one coupling operator: which layer depends on which."""

    phase: str
    source_layer: str
    target_layer: str
    assignment: str = "nearest"
    placement_priority: Mapping[str, float] | None = None

    def signature(self) -> dict:
        return {
            "phase": self.phase,
            "source_layer": self.source_layer,
            "target_layer": self.target_layer,
            "assignment": self.assignment,
        }


class PhaseAwareEngine:
    """Generic engine. Knows about phases and coupling; not about any hazard."""

    def __init__(
        self,
        layers: Mapping[str, object],
        hazard_field: RadialHazardField,
        fragility: Mapping[str, LognormalFragility],
        propagation_spec: CouplingSpec,
        restoration_spec: CouplingSpec,
        damage_mapping: DamageMapping | None = None,
        repair_model: ResourceConstrainedRepair | None = None,
        execution_order: Sequence[str] | None = None,
        hazard_shape: str = "ramp",
        warmup_steps: int = 5,
    ) -> None:
        self.layers = dict(layers)
        self.hazard_field = hazard_field
        self.fragility = dict(fragility)
        self.propagation_spec = propagation_spec
        self.restoration_spec = restoration_spec
        self.damage_mapping = damage_mapping or DamageMapping()
        self.repair_model = repair_model or ResourceConstrainedRepair()
        self.execution_order = list(execution_order or self.layers.keys())
        self.hazard_shape = str(hazard_shape)
        self.warmup_steps = int(warmup_steps)

        if propagation_spec.phase != PROPAGATION:
            raise ValueError("propagation_spec must declare the propagation phase")
        if restoration_spec.phase != RESTORATION:
            raise ValueError("restoration_spec must declare the restoration phase")
        missing = set(self.execution_order) - set(self.layers)
        if missing:
            raise ValueError(f"execution_order references unknown layers: {sorted(missing)}")

    # -- provenance ------------------------------------------------------
    def cache_key(self, scenario: Scenario) -> str:
        env = {
            "layers": {k: dict(v.signature()) for k, v in sorted(self.layers.items())},
            "hazard_field": self.hazard_field.__dict__,
            "fragility": {
                k: [list(v.medians), list(v.betas)] for k, v in sorted(self.fragility.items())
            },
            "propagation": self.propagation_spec.signature(),
            "restoration": self.restoration_spec.signature(),
            "damage_mapping": self.damage_mapping.signature(),
            "repair": self.repair_model.signature(),
            "hazard_shape": self.hazard_shape,
            "warmup_steps": self.warmup_steps,
            "execution_order": list(self.execution_order),
        }
        payload = json.dumps(
            {"scenario": scenario.to_dict(), "env": env},
            sort_keys=True, ensure_ascii=False, default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    # -- main loop -------------------------------------------------------
    def run(
        self,
        scenario: Scenario,
        results_root: str | Path | None = None,
        use_cache: bool = True,
    ) -> TrajectoryResult:
        key = self.cache_key(scenario)
        window = scenario.window

        for layer in self.layers.values():
            layer.reset(scenario.seeds.sim)

        positions, classes = self._component_inventory()
        # ``positions`` also carries service nodes, which coupling needs for
        # nearest-neighbour edge assignment but which take no damage of their
        # own. Only components have a fragility class, so only they are sampled.
        component_positions = {c: positions[c] for c in classes}
        damage = sample_damage(
            positions=component_positions,
            component_class=classes,
            fragility=self.fragility,
            field_model=self.hazard_field,
            window=window,
            seed_hazard=scenario.seeds.hazard,
            shape=self.hazard_shape,
        )

        op_prop, op_rec = self._build_operators(scenario, positions)

        # Warm-up to a steady pre-event state, then freeze the baseline.
        for k in range(self.warmup_steps, 0, -1):
            self._step_layers(t=window.t_oe - k * scenario.dt, dt=scenario.dt)
        baseline = self._collect("operational")

        grid = np.round(
            np.arange(window.t_oe, scenario.t_end + scenario.dt / 2, scenario.dt), 6
        )
        performance = {col: 1.0 for col in baseline}
        integrity = {col: 1.0 for col in baseline}
        applied: dict[str, DamageState] = {c: DamageState.NONE for c in component_positions}

        op_rows, infra_rows, events = [], [], []
        for t in grid:
            # Damage is frozen once the hazard ends: intensity_factor saturates
            # at 1.0, so no state can change after t_ee. Phase III is the bulk
            # of the timeline, so skipping the scan there matters.
            if t <= window.t_ee + scenario.dt:
                events += self._accrue_damage(t, damage, applied)

            if not op_prop.is_empty():
                self._apply_propagation(op_prop, performance)

            self._step_layers(t=float(t), dt=scenario.dt)

            operational = self._normalise(self._collect("operational"), baseline)
            infrastructure = self._collect("infrastructure")
            op_rows.append(operational)
            infra_rows.append(infrastructure)

            if t >= window.t_repair_start:
                events += self.repair_model.step(
                    t=float(t),
                    dt=scenario.dt,
                    layers=self.layers,
                    operator=op_rec,
                    performance=performance,
                    resources=scenario.resources_per_step,
                    infrastructure=integrity,
                )

            performance = operational
            integrity = infrastructure

        index = pd.Index(grid, name="t")
        result = TrajectoryResult(
            operational=pd.DataFrame(op_rows, index=index).sort_index(axis=1),
            infrastructure=pd.DataFrame(infra_rows, index=index).sort_index(axis=1),
            scenario=scenario,
            baseline=baseline,
            damage_summary=damage.summary(),
            coupling_summary={
                "propagation": op_prop.summary(self._propagation_targets()),
                "restoration": op_rec.summary(self._restoration_targets()),
            },
            events=events,
        )
        if results_root is not None:
            result.save(results_root, key=key)
        return result

    # -- pieces ----------------------------------------------------------
    def _component_inventory(self) -> tuple[dict, dict]:
        positions: dict[str, tuple[float, float]] = {}
        classes: dict[str, str] = {}
        for name, layer in self.layers.items():
            layer_positions = dict(layer.positions())
            for component in layer.components():
                col = node_col(name, component)
                positions[col] = layer_positions[component]
                classes[col] = layer.component_class(component)
            for node in layer.service_nodes():
                col = node_col(name, node)
                if col not in positions and node in layer_positions:
                    positions[col] = layer_positions[node]
        return positions, classes

    def _propagation_targets(self) -> list[str]:
        layer = self.layers[self.propagation_spec.target_layer]
        return [node_col(self.propagation_spec.target_layer, n) for n in layer.service_nodes()]

    def _propagation_sources(self) -> list[str]:
        layer = self.layers[self.propagation_spec.source_layer]
        return [node_col(self.propagation_spec.source_layer, n) for n in layer.service_nodes()]

    def _restoration_targets(self) -> list[str]:
        layer = self.layers[self.restoration_spec.target_layer]
        return [node_col(self.restoration_spec.target_layer, c) for c in layer.components()]

    def _restoration_sources(self) -> list[str]:
        layer = self.layers[self.restoration_spec.source_layer]
        return [node_col(self.restoration_spec.source_layer, n) for n in layer.service_nodes()]

    def _build_operators(
        self, scenario: Scenario, positions: Mapping[str, tuple[float, float]]
    ) -> tuple[CouplingOperator, CouplingOperator]:
        design = scenario.effective_design
        seed = scenario.seeds.design
        op_prop = build_operator(
            PROPAGATION,
            sources=self._propagation_sources(),
            targets=self._propagation_targets(),
            q=design.q_prop,
            kappa=design.kappa_prop,
            seed_design=seed,
            placement=design.placement_prop,
            priority=self.propagation_spec.placement_priority,
            positions=positions,
            assignment=self.propagation_spec.assignment,
        )
        op_rec = build_operator(
            RESTORATION,
            sources=self._restoration_sources(),
            targets=self._restoration_targets(),
            q=design.q_rec,
            kappa=design.kappa_rec,
            # Offset so that the two operators do not inherit the same layout.
            seed_design=seed + 9973,
            placement=design.placement_rec,
            priority=self.restoration_spec.placement_priority,
            positions=positions,
            assignment=self.restoration_spec.assignment,
        )
        return op_prop, op_rec

    def _accrue_damage(self, t: float, damage, applied: dict) -> list[dict]:
        events = []
        for col, previous in applied.items():
            state = damage.state_at(col, float(t))
            if state <= previous:
                continue
            layer_name, component = col.split(":", 1)
            layer = self.layers[layer_name]
            added = self.damage_mapping.work_for(state) - self.damage_mapping.work_for(previous)
            layer.apply_damage(
                component,
                capacity_factor=self.damage_mapping.capacity_factor(state),
                work=max(added, 0.0),
            )
            applied[col] = state
            events.append(
                {"t": float(t), "type": "damage", "target": col, "state": state.name}
            )
        return events

    def _apply_propagation(self, operator: CouplingOperator, performance: Mapping[str, float]) -> None:
        target_layer = self.propagation_spec.target_layer
        targets = self._propagation_targets()
        support = operator.support_levels(targets, performance)
        layer = self.layers[target_layer]
        layer.set_cross_layer_support(
            {col.split(":", 1)[1]: value for col, value in support.items()}
        )

    def _step_layers(self, t: float, dt: float) -> None:
        for name in self.execution_order:
            self.layers[name].step(float(t), float(dt))

    def _collect(self, kind: str) -> dict[str, float]:
        values: dict[str, float] = {}
        for name, layer in self.layers.items():
            source = layer.operational() if kind == "operational" else layer.infrastructure()
            for node, value in source.items():
                values[node_col(name, node)] = float(value)
        return values

    @staticmethod
    def _normalise(values: Mapping[str, float], baseline: Mapping[str, float]) -> dict[str, float]:
        out = {}
        for col, value in values.items():
            base = float(baseline.get(col, 0.0))
            out[col] = float(np.clip(value / base, 0.0, 1.0)) if base > 1e-12 else 1.0
        return out
