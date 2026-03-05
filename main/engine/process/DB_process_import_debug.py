from __future__ import annotations

import importlib
import os
import sys
from datetime import datetime
from types import ModuleType
from typing import Optional, Tuple


def _module_file_mtime(path: Optional[str]) -> str:
    if not path or not os.path.exists(path):
        return "<missing>"
    ts = os.path.getmtime(path)
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def try_import(module_name: str) -> Tuple[bool, str]:
    try:
        m: ModuleType = importlib.import_module(module_name)
        p = getattr(m, "__file__", None)
        return True, f"{p} | mtime={_module_file_mtime(p)}"
    except Exception as e:
        return False, f"<failed> {repr(e)}"


def dump_import_debug(log_fn=print) -> None:
    log_fn("[import_debug] sys.path head:")
    for i, p in enumerate(sys.path[:8]):
        log_fn(f"[import_debug]   {i}: {p}")

    targets = [
        ("main.engine.process.DB_process_runner", "DB_process_runner"),
        ("main.engine.process.DB_process_orchestrator", "DB_process_orchestrator"),
        ("main.engine.process.printing.DB_process_printing", "DB_process_printing"),
        ("main.engine.process.DB_process_config", "DB_process_config"),
    ]

    log_fn("[import_debug] Loaded module file paths:")
    for mod, label in targets:
        ok, msg = try_import(mod)
        log_fn(f"[import_debug] {label}: {msg}")