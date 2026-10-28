"""Start-up hook of experiment 0072 (on PYTHONPATH only for its runs): applies the patches named in EXP0072_PATCH in
every process, spawned workers included."""

import importlib.util
import os
from pathlib import Path

if os.environ.get("EXP0072_PATCH"):
    _spec = importlib.util.spec_from_file_location(
        "run72", Path(__file__).resolve().parents[1] / "run.py"
    )
    assert _spec is not None and _spec.loader is not None
    _run72 = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_run72)
    _run72.apply(os.environ["EXP0072_PATCH"].split(","))
