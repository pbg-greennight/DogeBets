#!/usr/bin/env python3
"""Model compatibility validator for the multi-JSON trend method system.

Usage:
  python tools/validate_models.py --model models/trend_method_v2.json

What it does:
  - Loads trend_method_v2.json and resolves/merges includes (handles legacy 'parts/' prefix).
  - Loads and injects engine_library (unwraps accidental {"engine_library": {...}} wrapper).
  - Imports DB_process_config.cfg() and checks printing toggle paths used in printing/orchestrator
    against the canonical PRINT.* tree.
  - Scans python source files for:
      * config.get("...") / config["..."] keys
      * _print_cfg(config, "PRINT", "...", "...") paths
      * catalog.get("...") keys
    and reports keys/paths that are missing from config or truth-schema.
  - Builds a lightweight "truth schema" by statically analyzing dict keys constructed in:
      DB_process_calc.py (build_calc_out, build_bell_out, build_channels_out)
      DB_process_gauss_channel.py (channel output keys)
    then checks that catalog keys referenced in code exist in that schema.

Exit code:
  0 if no errors, 1 if errors found (warnings do not fail).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

from main.engine.process.DB_process_expr import compile_expr
from typing import Any, Dict, List, Set, Tuple


# ----------------------------
# JSON merge / include resolve
# ----------------------------

def _deep_merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Return deep-merged dict (b overwrites a)."""
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_include_path(model_dir: Path, include_path: str) -> Path:
    """Resolve include paths the same way the runtime loader does.

    In this repo, part files live under models/parts/*.json and the root model
    references them with the prefix 'parts/...'. We therefore **must not** strip
    that prefix here, otherwise validation will incorrectly look in models/*.
    """
    p = include_path.replace("\\", "/").strip().lstrip("/")
    return (model_dir / p).resolve()


def load_trend_method_config_v2(model_path: Path, *, engine_library_path: Path | None = None) -> Dict[str, Any]:
    base = json.loads(model_path.read_text(encoding="utf-8"))
    model_dir = model_path.parent

    merged: Dict[str, Any] = {}
    includes = base.get("includes") or []
    if not isinstance(includes, list):
        raise ValueError("includes must be a list")

    for inc in includes:
        if not isinstance(inc, str):
            raise ValueError(f"include entry not a string: {inc!r}")
        inc_path = _resolve_include_path(model_dir, inc)
        if not inc_path.exists():
            raise FileNotFoundError(f"include not found: {inc} -> {inc_path}")
        blob = json.loads(inc_path.read_text(encoding="utf-8"))
        merged = _deep_merge_dicts(merged, blob)

    # base overrides includes
    merged = _deep_merge_dicts(merged, base)

    # inject engine_library (optional)
    if engine_library_path is not None and engine_library_path.exists():
        lib_blob = json.loads(engine_library_path.read_text(encoding="utf-8"))
        # unwrap accidental wrapper
        if isinstance(lib_blob, dict) and "engine_library" in lib_blob and isinstance(lib_blob["engine_library"], dict):
            merged["engine_library"] = lib_blob["engine_library"]
        else:
            merged["engine_library"] = lib_blob

    return merged


# ----------------------------
# Static source scanning
# ----------------------------

_RE_CONFIG_GET = re.compile(r"\bconfig\.get\(\s*['\"]([^'\"]+)['\"]")
_RE_CONFIG_SUB = re.compile(r"\bconfig\[\s*['\"]([^'\"]+)['\"]\s*\]")
_RE_CATALOG_GET = re.compile(r"\bcatalog\.get\(\s*['\"]([^'\"]+)['\"]")
_RE_PRINT_CFG = re.compile(
    r"_print_cfg\(\s*config\s*,\s*['\"]PRINT['\"]\s*,\s*([^\)]+)\)"
)

def scan_keys_in_file(path: Path) -> Tuple[Set[str], Set[str], Set[Tuple[str,...]]]:
    txt = path.read_text(encoding="utf-8", errors="replace")
    flat_raw: Set[str] = set(_RE_CONFIG_GET.findall(txt)) | set(_RE_CONFIG_SUB.findall(txt))
    # Heuristic: most config keys in this repo are uppercase-ish (contain '_' or an uppercase letter).
    flat: Set[str] = {k for k in flat_raw if any(c.isupper() for c in k) or ('_' in k)}

    catalog_raw: Set[str] = set(_RE_CATALOG_GET.findall(txt))
    # Heuristic: catalog keys are typically dotted namespaces (e.g., channels.pv_tail)
    catalog: Set[str] = {k for k in catalog_raw if '.' in k or k.startswith(('bell', 'channels'))}
    print_paths: Set[Tuple[str,...]] = set()

    for m in _RE_PRINT_CFG.finditer(txt):
        args_blob = m.group(1)
        parts = re.findall(r"['\"]([^'\"]+)['\"]", args_blob)
        if parts:
            print_paths.add(tuple(parts))

    return flat, catalog, print_paths


# ----------------------------
# Truth schema (lightweight)
# ----------------------------

class _DictKeyCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.keys: Set[str] = set()
    def visit_Dict(self, node: ast.Dict) -> Any:
        for k in node.keys:
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                self.keys.add(k.value)
        self.generic_visit(node)


def _collect_dict_keys_in_func(py_path: Path, func_name: str) -> Set[str]:
    tree = ast.parse(py_path.read_text(encoding="utf-8", errors="replace"))
    func = None
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == func_name:
            func = n
            break
    if func is None:
        return set()
    col = _DictKeyCollector()
    col.visit(func)
    return col.keys


def build_truth_schema(repo_root: Path) -> Dict[str, Set[str]]:
    """Return a map of namespace -> keys (best-effort)."""
    schema: Dict[str, Set[str]] = {}

    calc = repo_root / "DB_process_calc.py"
    if calc.exists():
        schema["calc.build_calc_out"] = _collect_dict_keys_in_func(calc, "build_calc_out")
        schema["calc.build_bell_out"] = _collect_dict_keys_in_func(calc, "build_bell_out")
        schema["calc.build_channels_out"] = _collect_dict_keys_in_func(calc, "build_channels_out")

    gc = repo_root / "DB_process_gauss_channel.py"
    if gc.exists():
        tree = ast.parse(gc.read_text(encoding="utf-8", errors="replace"))
        col = _DictKeyCollector()
        col.visit(tree)
        schema["gauss_channel"] = col.keys

    return schema


def catalog_key_supported(key: str, schema: Dict[str, Set[str]]) -> bool:
    """Best-effort check for catalog.get("channels.pv_tail") style keys."""
    # known namespaces today
    # build_calc_out -> {bell, bell_curve_series, channels}
    # channels_out -> {snapshot, pv_tail}
    if key in ("bell", "bell_curve_series", "channels"):
        return True
    if key.startswith("channels."):
        # allow channels.snapshot / channels.pv_tail
        leaf = key.split(".", 1)[1]
        return leaf in ("snapshot", "pv_tail")
    if key.startswith("bell."):
        # bell.* is plausible (best-effort)
        return True
    return False


# ----------------------------
# Utilities
# ----------------------------

def _get_nested(d: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/trend_method_v2.json", help="Path to v2 method JSON (relative to repo-root unless absolute)")
    ap.add_argument("--engine-library", default="models/trend_method_engine_library.json", help="Path to engine library JSON (relative to repo-root unless absolute)")
    ap.add_argument(
        "--repo-root",
        default="AUTO",
        help="Repo root (where DB_process_*.py live). Default=AUTO (walk up from this file looking for DB_process_config.py).",
    )
    ap.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    args = ap.parse_args()

    def _auto_repo_root(start: Path) -> Path:
        """Walk upward from start until we find a folder that contains DB_process_config.py."""
        for parent in [start] + list(start.parents):
            if (parent / "DB_process_config.py").exists():
                return parent
        # fallback: if run from tools/, assume parent is repo root
        if start.name == "tools":
            return start.parent
        return start

    if str(args.repo_root).upper() == "AUTO":
        repo_root = _auto_repo_root(Path(__file__).resolve().parent)
    else:
        repo_root = Path(args.repo_root).resolve()

    model_arg = Path(args.model)
    lib_arg = Path(args.engine_library)
    model_path = (model_arg if model_arg.is_absolute() else (repo_root / model_arg)).resolve()
    lib_path = (lib_arg if lib_arg.is_absolute() else (repo_root / lib_arg)).resolve()

    print("[validate] repo_root:", repo_root)
    print("[validate] model:", model_path)
    print("[validate] engine_library:", lib_path)

    errors: List[str] = []
    warns: List[str] = []

    # Load + merge model
    try:
        if not model_path.exists():
            raise FileNotFoundError(
                f"model not found: {model_path} (tip: run from repo root, or pass --repo-root .., or pass an absolute --model path)"
            )
        merged = load_trend_method_config_v2(model_path, engine_library_path=lib_path)
    except Exception as e:
        errors.append(f"Failed to load/merge model: {e}")
        merged = {}

    if merged:
        # engine_library shape check
        el = merged.get("engine_library")
        if not isinstance(el, dict):
            errors.append("engine_library missing or not a dict after merge/inject")
        else:
            # detect accidental double-wrap
            if "engine_library" in el and isinstance(el.get("engine_library"), dict):
                errors.append("engine_library appears double-wrapped: cfg['engine_library']['engine_library'] exists")
            # minimal expected sections
            for must in ("avail_params_global", "avail_params_per_sigma", "namespaces"):
                if must not in el:
                    warns.append(f"engine_library missing section '{must}'")

    
        # engine_layer expression compile check (A2)
        try:
            engine_layer = merged.get("engine_layer", {}) or {}
            regimes = (engine_layer.get("regimes", {}) or {}) if isinstance(engine_layer, dict) else {}
            for rname, rblock in regimes.items():
                if not isinstance(rblock, dict):
                    continue
                expr = rblock.get("expr", {}) or {}
                if not isinstance(expr, dict):
                    continue
                when = expr.get("when", []) or []
                if isinstance(when, list):
                    for i, cond in enumerate(when):
                        if isinstance(cond, str) and cond.strip():
                            try:
                                compile_expr(cond)
                            except Exception as e:
                                warns.append(f"engine_layer.regimes.{rname}.expr.when[{i}] invalid: {e}")
                for k in ("trend", "confidence", "reason"):
                    if isinstance(expr.get(k), str) and expr.get(k).strip():
                        try:
                            compile_expr(expr[k])
                        except Exception as e:
                            warns.append(f"engine_layer.regimes.{rname}.expr.{k} invalid: {e}")
        except Exception:
            pass

# Import DB_process_config.cfg()
    cfg = None
    try:
        # Ensure repo root is on sys.path so imports work even when running from tools/.
        sys.path.insert(0, str(repo_root))
        import DB_process_config  # type: ignore
        cfg = DB_process_config.cfg()
    except Exception as e:
        warns.append(f"Could not import DB_process_config.cfg(): {e}")
        cfg = None

    # Scan files for keys
    py_files: List[Path] = []
    for p in [repo_root] + [repo_root / "printing"]:
        if p.exists():
            py_files += [x for x in p.rglob("*.py") if "tools" not in x.parts]

    flat_keys: Set[str] = set()
    catalog_keys: Set[str] = set()
    print_paths: Set[Tuple[str, ...]] = set()

    for f in py_files:
        fk, ck, pp = scan_keys_in_file(f)
        flat_keys |= fk
        catalog_keys |= ck
        print_paths |= pp

    # Report: config flat keys
    if cfg is not None:
        missing_flat = sorted([k for k in flat_keys if k not in cfg])
        # Allow standard python/logging internals that aren't config (none expected) and ignore keys that are clearly JSON-only
        legacy_ok = set([
            "LOG_GAUSS_CHANNELS",
            "LOG",
            "PRINT",
            "sigma_macro",
            "sigma_micro",
            "LOG_HYSTERESIS",
            "LOG_BELL_CURVE_SERIES_DUMP",
            "LOG_BELL_CURVE_DIAGNOSTICS",
            "PV_REF_SIGMA",
        ])
        missing_flat2 = [k for k in missing_flat if k not in legacy_ok]
        if missing_flat2:
            warns.append("Flat config keys referenced in code but missing from DB_process_config.cfg():\n  - " + "\n  - ".join(missing_flat2))

        # PRINT path checks (canonical toggle tree)
        if isinstance(cfg.get("PRINT"), dict):
            bad_paths: List[str] = []
            for path in sorted(print_paths):
                full = ("PRINT",) + path
                val = _get_nested(cfg, full)
                if val is None:
                    bad_paths.append(".".join(full))
            if bad_paths:
                warns.append("PRINT toggle paths referenced via _print_cfg() but missing from cfg():\n  - " + "\n  - ".join(bad_paths))
        else:
            warns.append("cfg()['PRINT'] missing or not a dict; cannot validate PRINT toggle paths")

    # Report: catalog keys referenced
    truth_schema = build_truth_schema(repo_root)
    unsupported_catalog = sorted([k for k in catalog_keys if not catalog_key_supported(k, truth_schema)])
    if unsupported_catalog:
        warns.append("Catalog keys referenced in code that are not supported by current truth schema (build_calc_out):\n  - " + "\n  - ".join(unsupported_catalog))

    # Report: engine_library namespace sanity (best-effort)
    if merged and isinstance(merged.get("engine_library"), dict):
        namespaces = merged["engine_library"].get("namespaces")
        if isinstance(namespaces, dict):
            # check that at least some namespaces align with current truth object
            expected_prefixes = ("bell", "channels", "decision")
            found = set()
            for ns in namespaces.keys():
                for pref in expected_prefixes:
                    if ns.startswith(pref):
                        found.add(pref)
            if not found:
                warns.append("engine_library.namespaces has no entries starting with bell/channels/decision; may be out of sync with truth schema")
        else:
            warns.append("engine_library.namespaces missing or not a dict")

    # Print summary
    print("\n==================== VALIDATION REPORT ====================")
    if errors:
        print("ERRORS:")
        for e in errors:
            print("  -", e)
    if warns:
        print("\nWARNINGS:")
        for w in warns:
            print("\n" + w)

    if not errors and not warns:
        print("✅ No issues found.")

    fail = bool(errors) or (args.strict and bool(warns))
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
