from __future__ import annotations

"""Validator and normalizer for the v21 live source-row contract."""

from typing import Any, Dict, Iterable, List

try:
    from main.engine.process.models.v21_live.v21_live_schema import (
        ALL_SRC_FIELDS,
        SRC_ALIASES,
        SRC_DEFAULTS,
        SRC_FAMILY_FIELDS,
        SRC_OPTIONAL_FIELDS,
        SRC_REQUIRED_FIELDS,
    )
except Exception:  # pragma: no cover - local fallback for standalone testing
    from process.models.v21_live.v21_live_schema import (
        ALL_SRC_FIELDS,
        SRC_ALIASES,
        SRC_DEFAULTS,
        SRC_FAMILY_FIELDS,
        SRC_OPTIONAL_FIELDS,
        SRC_REQUIRED_FIELDS,
    )


_NULLISH_STRINGS = {"", "nan", "none", "null"}


def _is_nullish(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _NULLISH_STRINGS
    return False


def get_src_default_value(key: str) -> Any:
    return SRC_DEFAULTS.get(key)


def normalize_src_row(row: dict | None) -> dict:
    src = dict(row or {})

    # Apply alias -> canonical remaps only when the canonical value is absent.
    alias_applied: List[str] = []
    for alias, canonical in SRC_ALIASES.items():
        if canonical not in src or _is_nullish(src.get(canonical)):
            if alias in src and not _is_nullish(src.get(alias)):
                src[canonical] = src.get(alias)
                alias_applied.append(alias)

    # Fill defaults for known fields only; preserve all unknown fields untouched.
    for key in ALL_SRC_FIELDS:
        if key not in src or _is_nullish(src.get(key)):
            default = get_src_default_value(key)
            if default is not None:
                src[key] = default

    if alias_applied:
        src["src_debug_alias_applied"] = sorted(alias_applied)

    return src


def _missing_fields(row: dict, fields: Iterable[str]) -> List[str]:
    out: List[str] = []
    for key in fields:
        if key not in row or _is_nullish(row.get(key)):
            out.append(key)
    return out


def summarize_src_family_coverage(row: dict | None) -> dict:
    src = dict(row or {})
    coverage: Dict[str, float] = {}
    present_total = 0
    total_total = 0
    for family, fields in SRC_FAMILY_FIELDS.items():
        total = len(fields)
        present = sum(1 for key in fields if key in src and not _is_nullish(src.get(key)))
        coverage[family] = float(present) / float(total) if total else 0.0
        present_total += present
        total_total += total
    coverage["overall"] = float(present_total) / float(total_total) if total_total else 0.0
    return coverage


def validate_src_row(row: dict | None) -> dict:
    src = normalize_src_row(row or {})
    missing_required = _missing_fields(src, SRC_REQUIRED_FIELDS)
    missing_optional = _missing_fields(src, SRC_OPTIONAL_FIELDS)
    coverage = summarize_src_family_coverage(src)

    null_count = sum(1 for key in ALL_SRC_FIELDS if _is_nullish(src.get(key)))

    return {
        "is_valid": len(missing_required) == 0,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "alias_applied": list(src.get("src_debug_alias_applied") or []),
        "null_count": null_count,
        "family_coverage": coverage,
    }


__all__ = [
    "get_src_default_value",
    "normalize_src_row",
    "summarize_src_family_coverage",
    "validate_src_row",
]
