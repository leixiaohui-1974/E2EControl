# Model Accuracy Check (Baseline Gate)

This document defines the reproducible automation entrypoint for the model-accuracy baseline gate.

## Entry Command

Run from repository root:

```bash
python scripts/generate_model_accuracy_reference.py
python scripts/check_model_accuracy.py --quick --json-out reports/acceptance/model_accuracy_summary.json
```

Use full duration (same duration as `scripts/verify_solvers.py`) by removing `--quick`.
The first command refreshes the tracked golden reference pack at `tests/reference_data/model_accuracy_reference_pack.json`.

## Default Gate Rule

Default mandatory checks are:

- `V1`: mass conservation
- `V2`: steady-state analytical deviation
- `V3`: wave-arrival timing
- `V4`: basic physical reasonableness
- `VREF`: comparison against the tracked golden reference pack

Default strict thresholds are:

- `V1 <= 5.0%`
- `V2 <= 3.0%`
- `V3 <= 30.0%` relative error against `t_dynamic`
- `V5 <= 3.0%` spread when cross-consistency is required
- `VREF gauge RMSE <= 0.03 m`
- `VREF profile RMSE <= 0.01 m`

Gate decision:

- `PASS`: all selected solvers pass all mandatory checks
- `FAIL`: any selected solver fails at least one mandatory check

The command always prints:

```text
MODEL_ACCURACY_STATUS=PASS
```

or

```text
MODEL_ACCURACY_STATUS=FAIL
```

It also exits with:

- exit code `0` on `PASS`
- exit code `1` on `FAIL`

The gate uses the raw metrics from `scripts/verify_solvers.py`, but it does not trust that script's permissive `passed` field as the final admission decision. `scripts/check_model_accuracy.py` re-evaluates each selected solver against the stricter thresholds above.
For the canonical quick/full step-response scenarios, it also loads `tests/reference_data/model_accuracy_reference_pack.json` and compares each baseline solver against a fine-grid `TVD-MUSCL` golden reference.

## Solver Scope

By default, the gate runs baseline high-resolution solvers:

- `Lax-Wendroff`
- `MacCormack`
- `Godunov-HLL`
- `TVD-MUSCL`

You can narrow the scope during debugging:

```bash
python scripts/check_model_accuracy.py --quick --solver-filter "Lax-Wendroff"
```

## Optional Modes

- Skip wave-speed check during debugging only:

```bash
python scripts/check_model_accuracy.py --quick --skip-wave-speed
```

- Skip golden-reference comparison during debugging only:

```bash
python scripts/check_model_accuracy.py --quick --skip-reference-comparison
```

- Require cross-consistency check (`V5`) as a hard gate:

```bash
python scripts/check_model_accuracy.py --quick --require-cross-consistency
```

- Override thresholds explicitly:

```bash
python scripts/check_model_accuracy.py --quick \
  --max-mass-error-pct 4.0 \
  --max-steady-error-pct 2.0 \
  --max-wave-arrival-error-pct 20.0 \
  --max-cross-spread-pct 2.0 \
  --max-reference-gauge-rmse-m 0.02 \
  --max-reference-profile-rmse-m 0.005
```

## Output Artifact

When `--json-out` is provided, the script writes a machine-readable summary with:

- overall pass/fail (`overall_pass`, `status`)
- gate rules (`gate_rule`)
- per-solver check details (`solvers`)
- failed mandatory items (`failed_items`)
- V5 cross-consistency result (`cross_consistency_v5`)
- golden-reference linkage (`gate_rule.reference_pack`, `gate_rule.reference_scenario_id`)

Each check record also carries:

- `legacy_pass`: the original verdict from `verify_solvers.py`
- `gate_pass`: the stricter baseline-gate verdict
- `gate_threshold`: the threshold used by `check_model_accuracy.py`
- `VREF` details include `gauge_rmse_m`, `max_gauge_rmse_m`, `profile_rmse_m`, and the `reference_scenario_id`

This file can be used directly in CI or manual acceptance evidence.
