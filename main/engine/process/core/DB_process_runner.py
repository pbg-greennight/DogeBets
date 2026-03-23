"""DB_process_runner.py

Drop-in runner for the refactored DB_DATA_PROCESS pipeline.

Intended usage in your project:
  python DB_process_runner.py

Behavior target: identical to the original DB_DATA_PROCESS.py entrypoint.
"""

from __future__ import annotations

import importlib
import os
from datetime import datetime

from main.engine.process.core.DB_process_orchestrator import main


def _fmt_mtime(path: str) -> str:
    try:
        ts = os.path.getmtime(path)
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "?"


def debug_import_paths() -> None:
    """Guardrail for 'Cause #1 (stale orchestrator / partial copy)' issues.

    Prints the exact module file paths Python loaded, plus file mtimes, so logs
    immediately reveal if you're executing an older duplicate file.
    """
    mod_names = [
        "DB_process_runner",
        "DB_process_orchestrator",
        "DB_process_printing",
        "DB_process_config",
    ]

    print("[import_debug] Loaded module file paths:")
    for name in mod_names:
        try:
            mod = importlib.import_module(name)
            p = getattr(mod, "__file__", "?") or "?"
            print(f"[import_debug] {name}: {p} | mtime={_fmt_mtime(p)}")
        except Exception as e:
            print(f"[import_debug] {name}: <failed> {e!r}")


if __name__ == "__main__":
    debug_import_paths()
    main()
