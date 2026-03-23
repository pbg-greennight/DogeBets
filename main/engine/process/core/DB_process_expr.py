# main/engine/process/DB_process_expr.py
"""
A2: Expression Language (safe eval)

Supports:
- numeric ops: + - * / ** % //
- comparisons: < <= > >= == !=
- boolean ops: and/or/not
- conditional expr: a if cond else b
- indexing via get("a.b.c") path resolver
- functions: abs, min, max, round, sign, clamp, exists, isfinite

Design goals:
- Safe (AST-whitelisted)
- Deterministic
- Works with nested dict/list truth objects (calc_out + cfg thresholds)
"""

from __future__ import annotations

import ast
import math
from typing import Any, Callable, Dict, Optional

PathContext = Dict[str, Any]


def _path_parts(path: str) -> list[str]:
    # supports "a.b.c" and "a['b']" style (light support)
    path = path.strip()
    if not path:
        return []
    # normalize bracket keys -> dot keys
    # a["b"] -> a.b
    path = path.replace('["', '.').replace("['", '.').replace('"]', '').replace("']", '')
    path = path.replace("[", ".").replace("]", "")
    parts = [p for p in path.split(".") if p != ""]
    return parts


def resolve_path(ctx: PathContext, path: str, default: Any = None) -> Any:
    """
    Resolve a dotted path inside nested dict/list structures.

    Special handling:
    - per_sigma.s83 -> per_sigma.83 if present
    - s83 tokens map to int 83 or str "83" keys
    """
    cur: Any = ctx
    for raw in _path_parts(path):
        key: Any = raw
        # sigma token s83 -> 83
        if isinstance(raw, str) and len(raw) >= 2 and (raw[0] in ("s", "S")) and raw[1:].isdigit():
            key_int = int(raw[1:])
            # try int then str
            if isinstance(cur, dict):
                if key_int in cur:
                    cur = cur[key_int]
                    continue
                if str(key_int) in cur:
                    cur = cur[str(key_int)]
                    continue
            key = key_int  # might still work for list indexing
        # numeric token -> int index/key
        if isinstance(raw, str) and raw.isdigit():
            key = int(raw)
        try:
            if isinstance(cur, dict):
                if key in cur:
                    cur = cur[key]
                else:
                    # also try str/int duality
                    if isinstance(key, int) and str(key) in cur:
                        cur = cur[str(key)]
                    elif isinstance(key, str) and key.isdigit() and int(key) in cur:
                        cur = cur[int(key)]
                    else:
                        return default
            elif isinstance(cur, (list, tuple)):
                if isinstance(key, int) and 0 <= key < len(cur):
                    cur = cur[key]
                else:
                    return default
            else:
                return default
        except Exception:
            return default
    return cur


def _isfinite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def sign(x: Any) -> int:
    try:
        v = float(x)
    except Exception:
        return 0
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def clamp(x: Any, lo: float, hi: float) -> float:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    return float(max(lo, min(hi, v)))


class UnsafeExpression(ValueError):
    pass


_ALLOWED_NODES = {
    ast.Expression,
    ast.UnaryOp, ast.UAdd, ast.USub, ast.Not,
    ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.BoolOp, ast.And, ast.Or,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.IfExp,
    ast.Constant,
    ast.Call,
    ast.Name,
    ast.Load,
}


def _validate_ast(node: ast.AST) -> None:
    for child in ast.walk(node):
        if type(child) not in _ALLOWED_NODES:
            raise UnsafeExpression(f"Disallowed syntax: {type(child).__name__}")
        # Disallow attribute access (foo.bar) and subscripts (foo[0]) to keep everything via get()
        if isinstance(child, ast.Attribute):
            raise UnsafeExpression("Attribute access is not allowed; use get('a.b.c')")
        if isinstance(child, ast.Subscript):
            raise UnsafeExpression("Subscript access is not allowed; use get('a.b.c')")


def compile_expr(expr: str) -> ast.Expression:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e.msg}") from e
    _validate_ast(tree)
    return tree  # type: ignore[return-value]


def eval_expr(expr: str, ctx: PathContext, *, default: Any = None) -> Any:
    """
    Evaluate expression against ctx with a restricted AST + function set.
    """
    tree = compile_expr(expr)

    def _get(path: str, dflt: Any = None) -> Any:
        return resolve_path(ctx, path, dflt)

    def _exists(path: str) -> bool:
        return resolve_path(ctx, path, None) is not None

    safe_funcs: Dict[str, Callable[..., Any]] = {
        "abs": abs,
        "min": min,
        "max": max,
        "round": round,
        "sign": sign,
        "clamp": clamp,
        "get": _get,
        "exists": _exists,
        "isfinite": _isfinite,
        "math": math,  # allowed only as a name; attribute access blocked, so harmless
    }

    # Provide no builtins
    safe_globals: Dict[str, Any] = {"__builtins__": {}}
    safe_locals: Dict[str, Any] = dict(safe_funcs)

    try:
        return eval(compile(tree, "<expr>", "eval"), safe_globals, safe_locals)  # noqa: S307 (safe due to AST whitelist)
    except Exception:
        return default
