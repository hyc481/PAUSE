from __future__ import annotations

import os
from pathlib import Path

"""
Centralized path configuration for the bench project.

By default, BENCH_ROOT is inferred from this file's location:
- this file lives in bench/utils/paths.py
- parents[1] gives the bench/ directory

You can optionally override BENCH_ROOT via the BENCH_ROOT environment
variable, but in normal usage no environment configuration is required.
"""

_DEFAULT_BENCH_ROOT = Path(__file__).resolve().parents[1]

# Public root for the project; may be overridden by BENCH_ROOT env var.
BENCH_ROOT: Path = Path(os.getenv("BENCH_ROOT", str(_DEFAULT_BENCH_ROOT))).resolve()

# Top-level dirs
BACKEND_DIR: Path = BENCH_ROOT / "backend"
PIPELINE_DIR: Path = BENCH_ROOT / "pipeline"
EVALUATION_DIR: Path = BENCH_ROOT / "evaluation"
ANALYSIS_DIR: Path = BENCH_ROOT / "analysis"
PROMPTS_DIR: Path = BENCH_ROOT / "prompts"

# Common run / snapshot locations (conventions; not required to exist)
RUNS_DIR: Path = BENCH_ROOT / "runs"
SNAPSHOTS_DIR: Path = BENCH_ROOT / "snapshots"

# Backend data and tool schema locations
DATA_DIR: Path = BACKEND_DIR / "data"
TOOL_SCHEMAS_DIR: Path = BACKEND_DIR / "tool_schemas"
TOOL_SCHEMAS_GEMINI_DIR: Path = BACKEND_DIR / "tool_schemas_gemini"

