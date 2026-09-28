"""Central path configuration for the analysis and diagnostic scripts.

Reads ``config.yaml`` from the repository root, or the file named in the
``MAMMO_CONFIG`` environment variable. Scripts import the paths they need from
here (for example ``from config import DATA_DIR, SCRATCH_DIR``) instead of
hardcoding absolute paths, so the whole pipeline is driven by one file.

Scripts under ``src/diagnostics/`` reach this module by putting ``src/`` on the
path first::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import DATA_DIR, SCRATCH_DIR
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_CFG = Path(os.environ.get("MAMMO_CONFIG", _ROOT / "config.yaml"))

try:
    with open(_CFG) as _f:
        _c = yaml.safe_load(_f) or {}
except FileNotFoundError as e:  # pragma: no cover - configuration error
    raise FileNotFoundError(
        f"Config file not found at {_CFG}. Edit config.yaml with your paths, "
        f"or set MAMMO_CONFIG to point at your own config file."
    ) from e


def _path(key: str) -> Path | None:
    v = _c.get(key)
    return Path(v) if v else None


DATA_DIR = _path("data_dir")
SCRATCH_DIR = _path("scratch_dir")
BIOMEDCLIP_SCRATCH_DIR = _path("biomedclip_scratch_dir")
SOURCE_TABLES_DIR = _path("source_tables_dir")
DICOM_CATALOGUE = _path("dicom_catalogue")
CHECKPOINTS = {k: Path(v) for k, v in (_c.get("checkpoints") or {}).items()}

_results = str(_c.get("results_dir", "results"))
RESULTS_DIR = Path(_results) if os.path.isabs(_results) else _ROOT / _results
REPO_ROOT = _ROOT
