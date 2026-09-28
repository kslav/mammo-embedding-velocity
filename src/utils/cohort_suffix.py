"""Cohort-suffix construction shared between step 02 and downstream
analysis scripts.

The suffix is the canonical cross-stage tag that ties together:
  - matched CSV filenames (`matched_{cases,controls}_Density_nPriors{suffix}.csv`)
  - per-cohort analysis output dirs (`drift_analysis{suffix}{,_hoppr}/`)
  - ablation output dirs (`drift_ablation_{model}{suffix}_{cell}/`)
  - the `COHORT` env var passed to `slurm_drift_ablation.sh`

Default (no flags non-default) → empty string, the tight cohort
convention. Any non-default flag combination produces a deterministic
underscore-prefixed suffix.

Quick CLI for previewing what suffix a flag combination would produce:

    python -m utils.cohort_suffix --max-days 800 --dt-tolerance-days 180
    # → _d365_d800_dt180
"""
from __future__ import annotations

from typing import Optional


def build_suffix(
    min_days: int = 365,
    max_days: int = 425,
    dt_tolerance_days: Optional[int] = None,
    include_mesh_path_proven: bool = False,
) -> str:
    """Build the cohort suffix from matching parameters.

    Components (concatenated with underscores, prefixed by a single `_`):
      d{min}_d{max}   — non-default screening window (omitted when 365–425)
      dt{tol}         — dt-vector tolerance in days (omitted when None)
      mesh            — mesh-path-proven augmentation enabled (omitted when off)

    Returns "" for the tight default to preserve the original
    matched_cases_Density_nPriors.csv filename.
    """
    parts: list[str] = []
    if (min_days, max_days) != (365, 425):
        parts.append(f"d{min_days}_d{max_days}")
    if dt_tolerance_days is not None:
        parts.append(f"dt{dt_tolerance_days}")
    if include_mesh_path_proven:
        parts.append("mesh")
    return "_" + "_".join(parts) if parts else ""


def main() -> None:
    """CLI for previewing suffixes without running step 02."""
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--min-days", type=int, default=365)
    p.add_argument("--max-days", type=int, default=425)
    p.add_argument("--dt-tolerance-days", type=int, default=None)
    args = p.parse_args()
    suffix = build_suffix(
        min_days=args.min_days,
        max_days=args.max_days,
        dt_tolerance_days=args.dt_tolerance_days,
    )
    print(suffix if suffix else "(empty — tight default)")


if __name__ == "__main__":
    main()
