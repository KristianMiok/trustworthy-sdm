# Upstream API dependency note

`trustworthy_sdm.regen.assemble_inputs` calls `sdm_robustness.execution.runner._prepare_entity_data`
— a private function (leading underscore) in the upstream package.

## Why we accept this

The alternative is to reimplement the entity-data prep ourselves: filter the
master table by `Crayfish_scientific_name`, apply native/alien treatment,
build the benchmark with `accuracy=High` and `distance_m <= 200`, dedupe by
`subc_id`, build the snap and lowacc pools with the right thresholds, and
call `prepare_accessible_area`. About 40 lines of code that must stay in
exact lockstep with upstream. Drift produces silently miscalibrated results.

Better to import the upstream function and accept the risk that upstream
might rename or refactor it.

## How we mitigate

- Upstream is pinned to v1.0 in the published paper (will switch from
  editable install to git+ref before submission).
- A smoke test in `tests/test_upstream_compat.py` (TODO) imports the function
  by name and checks its return-dict keys. If upstream renames it, our CI
  fails immediately rather than producing garbage.

## Trigger to revisit

If we ever target a different upstream version, or if `_prepare_entity_data`
gets renamed, replace the import with our own implementation copied from
runner.py at the matching git ref.
