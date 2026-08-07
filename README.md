# Phase-specific coupling in interdependent infrastructure networks

A simulation framework in which the dependencies that transmit disturbance
between infrastructure networks and the dependencies that constrain repair
afterwards are represented as **two independent weighted operators**, switchable
separately under a shared hazard realisation.

Running the four on/off combinations of the two operators on the same hazard
draw gives an exact additive decomposition of the coupling-induced resilience
loss into a propagation contribution, a restoration contribution, and their
interaction, with no residual for any outcome defined in all four arms.

Two application cases are included: an IEEE33 distribution feeder and an EPANET
Net3 water network, each coupled to the Sioux Falls road network under a
spatially correlated pluvial flood.

## Requirements

Python 3.9 or later.

```bash
pip install -r requirements.txt
```

`pandapower` and `wntr` both changed solver defaults between minor releases, so
versions are pinned rather than bounded; the stored results are reproducible
against these.

## Quick start

```bash
python examples/run_case1_factorial.py --replications 20 --outdir results/demo
```

Run the test suite (74 tests, about two minutes):

```bash
python -m pytest tests -q
```

## Layout

```text
coupling0729/          the framework
  core/                scenario, hazard, damage, coupling operators, repair, engine
  metrics/headline.py  1 - R, Lambda, time to recovery, censoring, lock-in
  experiments/         factorial arms, attribution, design sweeps, interaction share
  cases/               power-transport, water-transport, and a transparent toy case
coupling_sim/          the two layer simulators and the Sioux Falls road network
examples/              experiment runners and plotting scripts
tests/                 design-invariant tests
hpc/                   SGE array-job scripts for the cluster runs
data/net3/Net3.inp     the EPANET network used by the water case
results/               stored output of every experiment
```

`coupling_sim/` holds the AC power-flow and traffic-assignment simulators and
the Sioux Falls TNTP files, taken from the larger simulation package this work
was built on. Only the modules the framework imports are included, so it is a
dependency of this repository rather than a copy of that project.

## Stored results

`results/` contains the output of 105,600 simulation runs. Each experiment
directory holds a `runs.csv` with one row per replication per arm.

| directory | design | replications |
|---|---|---|
| `hpc_n1000/case{1,2}_n1000` | the baseline design point | 1,000 × 2 cases |
| `hpc_n1000/case{1,2}_gate_infra_n1000` | repair access reads physical integrity | 1,000 × 2 cases |
| `hpc_n1000/case{1,2}_rho_sweep` | `q` × `κ`, 5 × 5 | 200 per cell × 2 cases |
| `case1_gate_grid` | `a_thr` × `κ`, 3 × 6 | 200 per cell |
| `case1_cross` | `κ_prop` × `κ_rec`, 4 × 4 | 200 per cell |
| `case1_severity` | flood peak 0.40–0.90 m, 6 levels | 100 per level |
| `case{1,2}_layouts` | 20 independently drawn coupling layouts | 100 each × 2 cases |
| `case{1,2}_multiT` | one trajectory evaluated at 8 horizons | 300 × 2 cases |
| `case1_gate_horizon`, `case2_horizon` | evaluation horizon × 5 | 200 × 2 cases |

Figures are rebuilt from these files without re-simulating:

```bash
python examples/plot_attribution.py
```

## Seed hierarchy

Getting this wrong silently destroys the identification, so it is explicit in
`SeedBundle`:

| seed | controls | across the four arms | across replications |
|---|---|---|---|
| `hazard` | hazard field and damage draw | **shared** | varies |
| `sim` | simulator internals | **shared** | varies |
| `design` | where coupling edges are placed | fixed — it *is* the design | fixed |

`hazard` and `sim` are the common random numbers: holding them identical across
arms is what makes the paired differences isolate the coupling effect. `design`
is held fixed across replications so a design point means the same thing for
every hazard realisation; only the layout experiment varies it. The two
operators are drawn from offset design seeds, so the restoration operator does
not inherit the propagation operator's layout.

`run_factorial` asserts that the four arms of a replication share a pairing key.

## Coupling semantics

One rule for both phases: **`κ` is the fraction of the source's degradation that
is transmitted**. An edge `(i → j, κ)` with source performance `p_i` transmits
`κ · (1 − p_i)`, and the effective support of target `j` is

```text
s_j = 1 − max_i κ_ij (1 − p_i)
```

So `κ = 0` is indistinguishable from a missing edge, and the four arms are
corner points of one continuous design space rather than a separate mechanism.

The operators are deliberately not inverses of one another. `D_prop` runs from a
source-layer node to a transport node and acts on delivered service at every
step. `D_rec` runs from a transport node to a damaged component and acts only on
repair, after mobilisation, through two channels driven by the same `s_j`:
**start gating**, below which a component is postponed and the crew redeploys,
and **rate derating**, in which work proceeds at rate `s_j`.

## Event timeline

```text
      pre-event        Phase I           Phase II            Phase III
    ------------|----------------|------------------|--------------------->
               t_oe             t_ee          t_ee + t_mob
                 damage accrues    embargo: no repair    repair proceeds
```

The repair embargo is load-bearing rather than cosmetic. `Lambda` is read over
`[t_oe, t_ee + t_mob]`, a window containing no repair, so the depth measure
cannot absorb restoration-coupling effects. Time to recovery is anchored at
`t_ee`, an exogenous instant, so it captures both how long until repair begins
and how fast it then proceeds; anchoring at the observed onset of recovery would
subtract out precisely the delay that restoration coupling causes.

## Running on a cluster

`hpc/` contains the SGE array-job scripts used for the large runs. The work is
embarrassingly parallel across replication blocks, so each task takes a single
slot and the array length is the number of blocks.

Cluster paths in these scripts are placeholders. Either edit the SGE directives
at the top of each jobscript, or supply them at submission time:

```bash
qsub -wd <project> -o <logdir> -e <logdir> -N c1 -t 1-50 -v SGE_ARRAY_N=50 hpc/array_job.sh examples/run_case1_factorial.py --replications 1000 --outdir results/case1_n1000
```

Shell paths inside the scripts read `PROJECT_ROOT`, `PYLIBS` and `LOGDIR` from
the environment, falling back to defaults under `$HOME`.

## License

MIT — see `LICENSE`.
