# Theory and synthetic-source reproduction

These programs implement the finite deterministic-mixture coding class in the
paper **Finite-Mode Semantic Coding for Edge IoT under Target Distribution and
Shared-Randomness Constraints**. They are independent of PyTorch and CompressAI.
The numerical checks complement the analytic statements; a quick run is not a
new independent proof or a rerun of the archived full Monte Carlo experiment.

## Fast CPU checks

From the release root, after installing the core dependencies:

```bash
python -m theory.verify
python -m unittest discover -s tests/theory -v
```

The first command recomputes the strict Bernoulli interval witness, exhaustively
classifies all 210 non-Bernoulli vertex systems, checks all 90 archived feasible
simulation points against the optimizer's analytic metrics, and runs a small
exact-sampler/channel smoke test. It writes `outputs/theory_verification.json`.
Only NumPy and mpmath are needed for these commands. The supplied 19 unit tests
also retain zero-gain controls, an exclusion witness, and a counterexample to an
over-strong task-endpoint interpretation; they do not select only favorable points.

The canonical strict witness is
`q=0.18, y=0.10, epsilon=0.235, R=0.082, C=0.90, Rc=0.78`.
The three-mode distortion is approximately `0.2791706047763627`, and the
four-mode distortion is approximately `0.26021327396586824`.

## Code-to-paper map

| Paper component | Implementation / archived data |
|---|---|
| Exact Bernoulli candidates and three-mode faces (Theorem 2; Table II; Fig. 2) | `theory/bernoulli_mode_limited.py` |
| Strict fourth-mode open region (Theorem 3) | `theory/k4_strict_open_set_certificate.py`; 80-decimal outward interval arithmetic |
| Non-Bernoulli support-sharpness witness (Table I) | `theory/general_finite_alphabet_vertex_certificate.py`; `data/theory/general_finite_alphabet/` |
| M/O task quantities and feasibility / common-frontier controls | `theory/audit_bernoulli_mo_advantage.py`; `tests/theory/test_receiver_visibility_feasibility.py`; `data/theory/bernoulli_mo_advantage/` |
| Bernoulli operational and exact fair-bit sampling experiments (Figs. 4–5) | `theory/simulate_bernoulli_finite_mode.py`; `data/theory/bernoulli_sim/` |
| Optional symbolic and full scalar-root branch checks | `theory/general_support_theorem_symbolic_audit.py`, `theory/output_only_*` |

The phase-grid and exact publication-figure source arrays are mapped separately
by the figure-reproduction documentation. The 3,514-point M/O `ab_region.csv`
is not the 20,402-cell phase grid used for the two-panel strict-gain figure.

## Standalone interval-certificate commands

```bash
python -m theory.k4_strict_open_set_certificate
python -m theory.general_finite_alphabet_vertex_certificate --output-dir outputs/nonbernoulli
```

The latter writes the complete vertex enumeration, including every infeasible
active-system classification, not just the two optimal vertices.

## Full operational simulation

The archived 90-row table was generated with 10 replications, one million source
symbols per replication, 200,000 sampler draws per schedule, and base PCG64 seed
`20260917`:

```bash
python -m theory.simulate_bernoulli_finite_mode --rc-min 0 --rc-max 2 --rc-step 0.05 --n 1000000 --reps 10 --sampler-draws 200000 --seed 20260917 --output-dir outputs/bernoulli_sim
```

This is deliberately not part of the quick command. There are 123 configured
budget/mode combinations; 33 are analytically infeasible and have no executable
schedule, leaving the 90 feasible operating points in the archive. The infeasible
region remains visible in the paper, rather than being interpreted as zero loss.

To check the public CLI on a small sample instead:

```bash
python -m theory.simulate_bernoulli_finite_mode --rc-min 0.78 --rc-max 0.78 --rc-step 0.05 --n 10000 --reps 2 --sampler-draws 1000 --seed 20260917 --output-dir outputs/bernoulli_smoke
```

The smoke output is a fresh low-cost diagnostic, not a replacement for the
published table. New results go to `outputs/`, never overwrite `data/`, and a
Git revision is recorded only when the release root itself is a Git checkout;
an unpacked ZIP reports `unknown` rather than accidentally using a parent repo.

## Optional symbolic checks

Install SymPy to use the additional symbolic / full M/O branch programs:

```bash
python -m pip install sympy
python -m theory.general_support_theorem_symbolic_audit
```

The `output_only_*` modules retain the positive and negative correlation branch
and root-free candidate constructions. Their `main()` entry points contain
larger numerical audits and are optional; do not confuse them with the fast
certificate command. Their source algorithms were copied without numerical edits.

## Archived data and units

- `data/theory/bernoulli_sim/bernoulli_finite_mode_sim_v1.csv`: all 90 feasible
  point-level aggregate rows, 30 for each `K=2,3,4`. `ana_*` is analytic;
  `sim_*` is the mean over 10 replications; `hw_*` is the marginal two-sided
  95% t-interval half-width. `ky_*` is fair bits per schedule draw. Entropies and
  rates are bits/source-symbol; distortion and weights are dimensionless.
- The original simulation archive stores aggregate rows and their uncertainty,
  **not individual symbol draws or the ten per-replication estimates**. These
  can be regenerated from the complete frozen seeds/configuration and simulator.
  The release does not mislabel the aggregate table as raw observations.
- `data/theory/bernoulli_mo_advantage/rc_sweep.csv`: all archived feasible
  analytic schedules in the sweep. `delta_h` is
  `H(S|Y)-H(S|Y,V)`; `delta_d_k3_minus_k4` compares designs. Empty CSV fields mean
  unavailable comparisons, not zero. `ab_region.csv` contains every retained
  grid point, including zero or floating-roundoff-near-zero information gains.
- `data/theory/general_finite_alphabet/`: exact rational inputs and full
  outward-interval certificate for Table I, plus all certified feasible vertices.
  The JSON includes counts for all 210 attempted systems, including singular,
  negative-weight and budget-violating systems.

All seven archived files are copied byte-for-byte from the research workspace;
source relative paths and the portable-code adaptation record are in
`docs/theory_inventory.json`. Historical commit strings identify the original
generator context, not the later public-release commit.
