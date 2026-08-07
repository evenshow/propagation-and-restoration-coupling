"""Experimental design and attribution."""

from .factorial import HEADLINE_METRICS, effect_summary, paired_effects, run_factorial
from .attribution import (
    attribution_audit,
    attribution_table,
    conditional_recovery_table,
    diagonal_dominance,
    superadditivity,
)
from .interaction_share import interaction_share, share_surface
from .sweep import (
    COMPOSITION_COLUMNS,
    DESIGN_COLUMNS,
    collapse_diagnostic,
    containment_threshold,
    design_means,
    effective_coupling_fit,
    marginal_slopes,
    placement_decomposition,
    predicted_absorption_threshold,
    psi_grid,
    quantisation_report,
    replication_rows,
    run_design_sweep,
    run_design_sweep_parallel,
    sweep_effects,
)

__all__ = [
    "HEADLINE_METRICS",
    "run_factorial",
    "paired_effects",
    "effect_summary",
    "interaction_share",
    "share_surface",
    "attribution_table",
    "attribution_audit",
    "conditional_recovery_table",
    "diagonal_dominance",
    "superadditivity",
    "psi_grid",
    "run_design_sweep",
    "run_design_sweep_parallel",
    "replication_rows",
    "sweep_effects",
    "design_means",
    "collapse_diagnostic",
    "placement_decomposition",
    "effective_coupling_fit",
    "predicted_absorption_threshold",
    "marginal_slopes",
    "containment_threshold",
    "quantisation_report",
    "DESIGN_COLUMNS",
    "COMPOSITION_COLUMNS",
]
