# main/engine/process/printing/DB_process_SectionLogger.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
import logging

DEFAULT_METHOD_MAP = {
    # MSBC
    "msbc_leg1": "pv_MSBC_Leg1",
    "msbc_leg2": "pv_MSBC_Leg2",
    # GBC
    "gbc_sd": "gbc_sd",
    "gbc_diag": "gbc_diag",
    "gbc_bc_diag": "gbc_bc_diag",
    # CSD / DCSD
    "csd_leg1": "csd_leg1",
    "dcsd_leg1": "dcsd_leg1",
    "csd_leg2": "csd_leg2",
    "dcsd_leg2": "dcsd_leg2",
    # Hysteresis
    "hyst_header": "hyst_header",
    "hyst_episode": "hyst_episode",
    "hyst_probe": "hyst_probe",
    "hyst_eta": "hyst_eta",
    "hyst_ladder": "hyst_ladder",
    "hyst_debug": "hyst_debug",
    # Other common blocks
    "perf": "perf",
    "tail_anchor": "tail_anchor",
    "pv_ref": "pv_ref",
    "gcs": "gcs",
    "pv_tail_channels": "pv_tail_channels",
}

@dataclass
class SectionLogger:
    """A small wrapper around a standard logger that gates output by section id."""
    base: logging.Logger
    sections: Dict[str, bool]
    method_map: Dict[str, str] = None

    def __post_init__(self) -> None:
        if self.method_map is None:
            self.method_map = dict(DEFAULT_METHOD_MAP)

    def is_enabled(self, section_id: str) -> bool:
        return bool(self.sections.get(section_id, False))

    def section(self, section_id: str, msg: str, *args: Any, level: str = "info", **kwargs: Any) -> None:
        if not self.is_enabled(section_id):
            return
        # Keep the user's preferred bracket tag style.
        if msg and not msg.lstrip().startswith("["):
            msg = f"[{section_id}] {msg}"
        else:
            # If msg already starts with [, leave it.
            pass
        getattr(self.base, level)(msg, *args, **kwargs)

    # passthroughs
    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.base.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.base.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.base.error(msg, *args, **kwargs)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.base.debug(msg, *args, **kwargs)

    def __getattr__(self, name: str) -> Callable[..., None]:
        # Allow calls like logger.msbc_leg1("...")
        section_id = self.method_map.get(name)
        if section_id is None:
            raise AttributeError(name)

        def _fn(msg: str, *args: Any, **kwargs: Any) -> None:
            self.section(section_id, msg, *args, **kwargs)

        return _fn
