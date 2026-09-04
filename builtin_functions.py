"""
builtin_functions.py

Predefined GridLang functions (SUM, ROWS, TEXTSPLIT, MID, LEN, etc.).

All builtins are registered in BUILTINS (case-insensitive, so SUM == sum) and
are exposed to the expression evaluator via get_builtin_functions(evaluator).

To add a new builtin:
    from builtin_functions import register_builtin

    @register_builtin("MYFUNC", aliases=["MyFuncAlias"])
    def myfunc(*args, _evaluator=None, _scope=None, _line_number=None):
        # _evaluator is the ExpressionEvaluator (for array helpers, etc.)
        # Use _evaluator.compiler.array_handler etc. if needed
        return ...

The decorator stores the raw function; get_builtin_functions() wraps it so the
evaluator/scope/line_number are injected when the function is called from
GridLang. For simple pure functions (e.g. SQRT) the wrapper is unnecessary.

The two scope builders (_build_python_fallback_scope / _get_eval_globals) both
call get_builtin_functions() so a new function is instantly available in both
evaluation paths.
"""

import math
import random
import re

from utils import is_sparse_array


# ---------------------------------------------------------------------------
# Registry and keyword exclusions (single source of truth)
# ---------------------------------------------------------------------------

BUILTINS = {}
ALIASES = {}

# All keywords that must be ignored when extracting variable dependencies.
# Keep this list at a single place – compiler.py and expression.py import it.
KEYWORDS = {
    'to', 'and', 'or', 'not', 'then', 'do', 'step', 'by', 'in', 'new', 'with',
    'true', 'false', 'of', 'as', 'dim', 'index', 'init',
    'mod', 'div', 'none', 'nan', 'inf'
}


def register_builtin(name, aliases=None, func=None):
    """Register a builtin. Can be used as @register_builtin("NAME")."""
    def decorator(fn):
        key = name.lower()
        BUILTINS[key] = fn
        if aliases:
            for a in aliases:
                ALIASES[a.lower()] = key
                BUILTINS[a.lower()] = fn
        return fn
    if func is not None:
        return decorator(func)
    return decorator


def _resolve_builtin(name):
    low = name.lower()
    return BUILTINS.get(low) or BUILTINS.get(ALIASES.get(low, ""))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_list(val):
    if is_sparse_array(val):
        return [val[k] for k in sorted(val.keys())]
    if isinstance(val, dict) and 'array' in val:
        return list(val['array'])
    if isinstance(val, (list, tuple, set)):
        return list(val)
    return [val]


def _strip_list(val):
    # for single-arg sum handling
    return _to_list(val)


# ---------------------------------------------------------------------------
# Builtins
# ---------------------------------------------------------------------------

@register_builtin("ROWS")
def builtin_rows(arr, _evaluator=None, _scope=None, _line_number=None):
    if is_sparse_array(arr):
        return max((k[0] for k in arr.keys()), default=-1) + 1
    if isinstance(arr, dict) and 'array' in arr:
        shape = arr.get('shape') or arr.get('original_shape') or []
        return shape[0] if shape else 0
    if hasattr(arr, '__len__'):
        return len(arr)
    if isinstance(arr, (list, tuple)):
        return len(arr)
    return 0


@register_builtin("LEN", aliases=["Text.Len"])
def builtin_len(val, _evaluator=None, _scope=None, _line_number=None):
    if isinstance(val, str):
        return len(val)
    if is_sparse_array(val):
        items = [val[k] for k in sorted(val.keys())]
    elif isinstance(val, dict) and 'array' in val:
        items = list(val['array'])
    elif isinstance(val, (list, tuple)):
        items = list(val)
    else:
        raise TypeError("Len expects text or an array of text values")
    lengths = []
    for item in items:
        if item is None:
            lengths.append(0)
        elif isinstance(item, str):
            lengths.append(len(item))
        else:
            raise TypeError("Len expects text or an array of text values")
    return lengths


@register_builtin("ABS", aliases=["Number.Abs"])
def builtin_abs(n, _evaluator=None, _scope=None, _line_number=None):
    return abs(n)


@register_builtin("INT", aliases=["Number.Int"])
def builtin_int(n, _evaluator=None, _scope=None, _line_number=None):
    return int(n)


@register_builtin("MID", aliases=["Text.Mid"])
def builtin_mid(text, start, length=1, _evaluator=None, _scope=None, _line_number=None):
    s = str(text)
    start_idx = max(int(start) - 1, 0)
    length = int(length)
    return s[start_idx:start_idx + length]


@register_builtin("TEXTSPLIT", aliases=["Text.Split"])
def builtin_textsplit(text, delimiter, _evaluator=None, _scope=None, _line_number=None):
    return str(text).split(str(delimiter))


@register_builtin("COUNTA")
def builtin_counta(val, _evaluator=None, _scope=None, _line_number=None):
    items = _to_list(val)
    count = 0
    for item in items:
        if isinstance(item, list) or (isinstance(item, dict) and 'array' in item):
            count += builtin_counta(item, _evaluator=_evaluator)
        elif item is None:
            continue
        elif isinstance(item, str) and item == "":
            continue
        else:
            count += 1
    return count


@register_builtin("RANDARRAY")
def builtin_randarray(n, _evaluator=None, _scope=None, _line_number=None):
    length = int(n)
    return [random.random() for _ in range(length)]


@register_builtin("SORTBY")
def builtin_sortby(arr, ord_vals, _evaluator=None, _scope=None, _line_number=None):
    arr_list = _to_list(arr)
    ord_list = _to_list(ord_vals)
    if len(arr_list) != len(ord_list):
        raise ValueError("SortBy expects arrays of the same length")
    pairs = list(zip(ord_list, arr_list))
    pairs.sort(key=lambda p: p[0])
    return [v for _, v in pairs]


@register_builtin("TRANSPOSE")
def builtin_transpose(arr, _evaluator=None, _scope=None, _line_number=None):
    # Use evaluator's array_handler for shape handling if available
    if _evaluator is not None:
        ah = _evaluator.compiler.array_handler
        if is_sparse_array(arr):
            result = {}
            for k, v in arr.items():
                if len(k) >= 2:
                    result[(k[1], k[0]) + k[2:]] = v
                else:
                    result[k] = v
            return result
        flat = ah.flatten_array(arr)
        shape = list(ah.get_array_shape(arr))
        flat = [float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v for v in flat]
        if len(shape) <= 1:
            if isinstance(arr, dict) and 'array' in arr:
                return arr
            return list(flat) if flat else arr
        new_shape = [shape[1], shape[0]] + shape[2:]
        strides = []
        acc = 1
        for s in shape:
            strides.append(acc)
            acc *= s
        new_total = acc
        new_flat = []
        for n_idx in range(new_total):
            rem = n_idx
            idxs = []
            for s in new_shape:
                idxs.append(rem % s)
                rem //= s
            old_idxs = [idxs[1], idxs[0]] + idxs[2:]
            old_flat = sum(oi * st for oi, st in zip(old_idxs, strides))
            new_flat.append(flat[old_flat])
        return {'array': list(new_flat), 'shape': list(new_shape), 'original_shape': list(new_shape)}
    # Fallback without evaluator (should not happen)
    return arr


@register_builtin("MIN")
def builtin_min(*args, _evaluator=None, _scope=None, _line_number=None):
    # Flatten single array arg
    if len(args) == 1 and isinstance(args[0], (list, dict)):
        args = tuple(_to_list(args[0]))
    return min(args)


@register_builtin("MAX")
def builtin_max(*args, _evaluator=None, _scope=None, _line_number=None):
    if len(args) == 1 and isinstance(args[0], (list, dict)):
        args = tuple(_to_list(args[0]))
    return max(args)


# Math aliases via math module (SIN, COS, etc. are available as math.sin)
for _name in ["sqrt", "sin", "cos", "tan", "log", "exp", "asin", "acos", "atan"]:
    fn = getattr(math, _name, None)
    if fn:
        register_builtin(_name, aliases=["Number." + _name])(lambda *a, _fn=fn, **kw: _fn(*a))


# ---------------------------------------------------------------------------
# SUM – handled specially because it has 3 syntaxes: sum[A1:B2], sum{a,b}, sum(...)
# We keep the range/var helpers in expression.py, but expose a sum function
# for sum(...) that delegates to them when a single string arg is given.
# ---------------------------------------------------------------------------

@register_builtin("SUM")
def builtin_sum(*args, _evaluator=None, _scope=None, _line_number=None):
    if _evaluator is None:
        return sum(args)
    if len(args) == 1 and is_sparse_array(args[0]):
        return sum(args[0].values())
    if len(args) == 1 and isinstance(args[0], dict) and 'array' in args[0]:
        return sum(args[0]['array'])
    if len(args) == 1 and isinstance(args[0], (list, tuple, set)):
        return sum(args[0])
    if len(args) == 1 and isinstance(args[0], str):
        s = args[0]
        if s.startswith('{') and s.endswith('}'):
            return _evaluator._evaluate_sum_vars(f"sum{s}", _scope, _line_number)
        if s.startswith('[') and s.endswith(']'):
            return _evaluator._evaluate_sum_range(f"sum{s}", _scope, _line_number)
    # Handle multiple args like SUM(1,2,3) or SUM(A1, B1)
    # Flatten single array arg case already handled, now handle multiple
    # If called as SUM({1,2,3}) where {1,2,3} was evaluated to set, the above handles set
    # For SUM(1,2,3) as three separate args, sum them directly
    try:
        return sum(args)
    except TypeError:
        # If args contains non-numeric like string, try to handle
        return sum(args[0]) if len(args) == 1 else sum(args)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_builtin_functions(evaluator, scope=None, line_number=None):
    """
    Return a dict of name -> callable for the evaluator.
    Each callable is wrapped so that _evaluator/_scope/_line_number are
    injected. Simple builtins that don't need those still work.
    """
    wrapped = {}
    for name, fn in BUILTINS.items():
        def _make_wrapper(fn):
            def wrapper(*args, **kwargs):
                kwargs['_evaluator'] = evaluator
                kwargs['_scope'] = scope
                kwargs['_line_number'] = line_number
                try:
                    return fn(*args, **kwargs)
                except TypeError as e:
                    if '_evaluator' in str(e) or '_scope' in str(e) or '_line_number' in str(e):
                        try:
                            kwargs.pop('_evaluator', None)
                            kwargs.pop('_scope', None)
                            kwargs.pop('_line_number', None)
                            return fn(*args, **kwargs)
                        except TypeError:
                            return fn(*args)
                    raise
            return wrapper
        wrapped[name] = _make_wrapper(fn)
    wrapped['math'] = math
    return wrapped
