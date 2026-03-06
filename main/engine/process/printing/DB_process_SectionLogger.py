# main/engine/process/printing/DB_process_SectionLogger.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping
import logging


def _as_dict(v: Any) -> Dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _enabled(node: Any, default: bool = False) -> bool:
    if isinstance(node, bool):
        return node
    if isinstance(node, dict):
        if "ENABLED" in node:
            return bool(node.get("ENABLED"))
    return bool(default)


@dataclass
class SectionLogger:
    base: logging.Logger
    config: Mapping[str, Any] = field(default_factory=dict)

    def _print(self) -> Dict[str, Any]:
        root = self.config if isinstance(self.config, dict) else {}
        return _as_dict(root.get("PRINT"))

    def _log(self, on: bool, msg: str, *args: Any, **kwargs: Any) -> None:
        if on:
            self.base.info(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.base.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.base.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.base.error(msg, *args, **kwargs)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.base.debug(msg, *args, **kwargs)

    def HEADER(self, msg: str, *args: Any, **kwargs: Any) -> None:
        p = self._print()
        self._log(bool(p.get("HEADER", True)), msg, *args, **kwargs)

    def PERF(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(_enabled(self._print().get("PERF"), True), msg, *args, **kwargs)

    def TAIL_ANCHOR(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(_enabled(self._print().get("TAIL_ANCHOR"), True), msg, *args, **kwargs)

    def PV_REF(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(_enabled(self._print().get("PV_REF"), True), msg, *args, **kwargs)

    def MSBC_Leg1(self, msg: str, *args: Any, **kwargs: Any) -> None:
        p = self._print()
        msbc = _as_dict(p.get("MSBC"))
        self._log(_enabled(_as_dict(msbc.get("LEG1")), True), msg, *args, **kwargs)

    def MSBC_Leg2(self, msg: str, *args: Any, **kwargs: Any) -> None:
        p = self._print()
        msbc = _as_dict(p.get("MSBC"))
        self._log(_enabled(_as_dict(msbc.get("LEG2")), True), msg, *args, **kwargs)

    def MSBC_Leg2_series(self, msg: str, *args: Any, **kwargs: Any) -> None:
        p = self._print()
        msbc = _as_dict(p.get("MSBC"))
        leg2 = _as_dict(msbc.get("LEG2"))
        self._log(_enabled(_as_dict(leg2.get("SERIES")), False), msg, *args, **kwargs)

    def MSBC_DIAGNOSTICS(self, msg: str, *args: Any, **kwargs: Any) -> None:
        p = self._print()
        msbc = _as_dict(p.get("MSBC"))
        self._log(_enabled(_as_dict(msbc.get("DIAGNOSTICS")), True), msg, *args, **kwargs)

    def GCS(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(_enabled(self._print().get("GCS"), True), msg, *args, **kwargs)

    def GCS_Leg1(self, msg: str, *args: Any, **kwargs: Any) -> None:
        gcs = _as_dict(self._print().get("GCS"))
        self._log(_enabled(_as_dict(gcs.get("LEG1")), True), msg, *args, **kwargs)

    def GCS_Leg1_series(self, msg: str, *args: Any, **kwargs: Any) -> None:
        gcs = _as_dict(self._print().get("GCS"))
        leg1 = _as_dict(gcs.get("LEG1"))
        self._log(_enabled(_as_dict(leg1.get("SERIES")), False), msg, *args, **kwargs)

    def GCS_Leg2(self, msg: str, *args: Any, **kwargs: Any) -> None:
        gcs = _as_dict(self._print().get("GCS"))
        self._log(_enabled(_as_dict(gcs.get("LEG2")), True), msg, *args, **kwargs)

    def GCS_Leg2_series(self, msg: str, *args: Any, **kwargs: Any) -> None:
        gcs = _as_dict(self._print().get("GCS"))
        leg2 = _as_dict(gcs.get("LEG2"))
        self._log(_enabled(_as_dict(leg2.get("SERIES")), False), msg, *args, **kwargs)

    def PV_TAIL_CHANNELS(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(_enabled(self._print().get("PV_TAIL_CHANNELS"), True), msg, *args, **kwargs)

    def PV_TAIL_CHANNELS_PER_SIGMA(self, msg: str, *args: Any, **kwargs: Any) -> None:
        ptc = _as_dict(self._print().get("PV_TAIL_CHANNELS"))
        self._log(_enabled(_as_dict(ptc.get("PER_SIGMA")), True), msg, *args, **kwargs)

    def CSD_DCSD_Leg1_CSD(self, msg: str, *args: Any, **kwargs: Any) -> None:
        cfg = _as_dict(_as_dict(_as_dict(self._print().get("CSD_DCSD")).get("LEG1")).get("CSD"))
        self._log(_enabled(cfg, True), msg, *args, **kwargs)

    def CSD_DCSD_Leg1_DCSD(self, msg: str, *args: Any, **kwargs: Any) -> None:
        cfg = _as_dict(_as_dict(_as_dict(self._print().get("CSD_DCSD")).get("LEG1")).get("DCSD"))
        self._log(_enabled(cfg, True), msg, *args, **kwargs)

    def CSD_DCSD_Leg2_CSD(self, msg: str, *args: Any, **kwargs: Any) -> None:
        cfg = _as_dict(_as_dict(_as_dict(self._print().get("CSD_DCSD")).get("LEG2")).get("CSD"))
        self._log(_enabled(cfg, True), msg, *args, **kwargs)

    def CSD_DCSD_Leg2_DCSD(self, msg: str, *args: Any, **kwargs: Any) -> None:
        cfg = _as_dict(_as_dict(_as_dict(self._print().get("CSD_DCSD")).get("LEG2")).get("DCSD"))
        self._log(_enabled(cfg, True), msg, *args, **kwargs)

    def GBC_SD(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(_enabled(_as_dict(self._print().get("GBC")).get("SD"), True), msg, *args, **kwargs)

    def GBC_DIAG(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(_enabled(_as_dict(self._print().get("GBC")).get("DIAG"), True), msg, *args, **kwargs)

    def GBC_BC_DIAG(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(_enabled(_as_dict(self._print().get("GBC")).get("BC_DIAG"), True), msg, *args, **kwargs)

    def HYST_HEADER(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(_enabled(_as_dict(self._print().get("HYSTERESIS")).get("HEADER"), True), msg, *args, **kwargs)

    def HYST_EPISODE(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(_enabled(_as_dict(self._print().get("HYSTERESIS")).get("EPISODE"), True), msg, *args, **kwargs)

    def HYST_PROBE(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(_enabled(_as_dict(self._print().get("HYSTERESIS")).get("PROBE"), True), msg, *args, **kwargs)

    def HYST_ETA(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(_enabled(_as_dict(self._print().get("HYSTERESIS")).get("ETA"), True), msg, *args, **kwargs)

    def HYST_LADDER(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(_enabled(_as_dict(self._print().get("HYSTERESIS")).get("LADDER"), True), msg, *args, **kwargs)

    def HYST_DEBUG(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(_enabled(_as_dict(self._print().get("HYSTERESIS")).get("DEBUG"), False), msg, *args, **kwargs)

    def TREND_DECISION(self, msg: str, *args: Any, **kwargs: Any) -> None:
        trend = _as_dict(self._print().get("TREND"))
        self._log(bool(trend.get("DECISION", True)), msg, *args, **kwargs)

    def TREND_SCORES(self, msg: str, *args: Any, **kwargs: Any) -> None:
        trend = _as_dict(self._print().get("TREND"))
        self._log(bool(trend.get("SCORES", True)), msg, *args, **kwargs)

    def TREND_CALC(self, msg: str, *args: Any, **kwargs: Any) -> None:
        trend = _as_dict(self._print().get("TREND"))
        self._log(bool(trend.get("CALC", True)), msg, *args, **kwargs)

    def TREND_FEATURES(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Optional feature dump for the Trend Decision section."""
        trend = _as_dict(self._print().get("TREND"))
        self._log(bool(trend.get("FEATURES", True)), msg, *args, **kwargs)


"""(module)

NOTE: TREND_FEATURES is implemented as a method on SectionLogger.
If you see a stray top-level TREND_FEATURES here, it means the file was
previously corrupted by an indentation error.
"""


def get_section_logger(base: logging.Logger, config: Any) -> SectionLogger:
    return SectionLogger(base=base, config=(config if isinstance(config, dict) else {}))
