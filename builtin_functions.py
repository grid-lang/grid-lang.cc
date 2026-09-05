"""
builtin_functions.py

Predefined GridLang functions (SUM, ROWS, TEXTSPLIT, MID, LEN, etc.).

All builtins are registered in BUILTINS (case-insensitive, so SUM == sum) and
are exposed to the expression evaluator via get_builtin_functions(evaluator).

To add a new builtin:
    from builtin_functions import register_builtin

    @register_builtin("MYFUNC", aliases=["MyFuncAlias"])
    def myfunc(*args):
        return ...

The decorator stores the raw function; get_builtin_functions() wraps it
when the function is called from GridLang.

The two scope builders (_build_python_fallback_scope / _get_eval_globals) both
call get_builtin_functions() so a new function is instantly available in both
evaluation paths.
"""

import math
import random
import re

from utils import is_sparse_array
from units import error_value, TYPE_ERROR


# ---------------------------------------------------------------------------
# Registry and keyword exclusions (single source of truth)
# ---------------------------------------------------------------------------

VECTORIZED = set()
BUILTINS = {}
ALIASES = {}
ARG_COUNTS = {}  # name (lower) -> expected argument count (int or (min, max))

# All keywords that must be ignored when extracting variable dependencies.
# Keep this list at a single place – compiler.py and expression.py import it.
KEYWORDS = {
    'to', 'and', 'or', 'not', 'then', 'do', 'step', 'by', 'in', 'new', 'with',
    'true', 'false', 'of', 'as', 'dim', 'index', 'init',
    'mod', 'div', 'none', 'nan', 'inf'
}


def register_builtin(name, aliases=None, func=None, arg_count=None):
    """Register a builtin. Can be used as @register_builtin("NAME"). Non-vectorized by default.

    arg_count declares how many comma-separated arguments a call may pass:
      - an int N means exactly N arguments,
      - a 2-tuple (min, max) means between min and max arguments (inclusive),
      - None means no arity check (variadic).
    The check is applied to the outer GridLang call (a single bracket array
    arg counts as one argument).
    """
    def _store(fn):
        key = name.lower()
        BUILTINS[key] = fn
        if arg_count is not None:
            ARG_COUNTS[key] = arg_count
        if aliases:
            for a in aliases:
                ALIASES[a.lower()] = key
                BUILTINS[a.lower()] = fn
                if arg_count is not None:
                    ARG_COUNTS[a.lower()] = arg_count
    def decorator(fn):
        _store(fn)
        return fn
    if func is not None:
        _store(func)
        return func
    return decorator


def register_vectorized_builtin(name, aliases=None, func=None, arg_count=None):
    """Register a vectorized builtin (applied element-wise, scalar broadcast)."""
    def decorator(fn):
        # Register as normal builtin first
        register_builtin(name, aliases=aliases, func=fn, arg_count=arg_count)
        VECTORIZED.add(name.lower())
        if aliases:
            for a in aliases:
                VECTORIZED.add(a.lower())
        return fn
    if func is not None:
        register_builtin(name, aliases=aliases, func=func, arg_count=arg_count)
        VECTORIZED.add(name.lower())
        if aliases:
            for a in aliases:
                VECTORIZED.add(a.lower())
        return func
    return decorator


def _check_arity(name, nargs, line_number=None):
    """Validate that a builtin call passes an allowed number of arguments.
    Raises a SyntaxError (compile-time) on mismatch instead of letting the
    function return a runtime error value."""
    expected = ARG_COUNTS.get(name.lower())
    if expected is None:
        return
    if isinstance(expected, (tuple, list)):
        lo, hi = expected
        ok = lo <= nargs <= hi
        desc = f"between {lo} and {hi}"
    else:
        ok = nargs == expected
        desc = f"{expected}"
    if not ok:
        where = f" at line {line_number}" if line_number is not None else ""
        raise SyntaxError(
            f"{name} expects {desc} argument(s) but received {nargs}{where}")


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


# ---------------------------------------------------------------------------
# Builtins
# ---------------------------------------------------------------------------

@register_builtin("ROWS", arg_count=1)
def builtin_rows(arr):
    if is_sparse_array(arr):
        return max((k[0] for k in arr.keys()), default=-1) + 1
    if isinstance(arr, dict) and 'array' in arr:
        shape = arr.get('shape') or arr.get('original_shape') or []
        return shape[0] if shape else 0
    if isinstance(arr, (list, tuple)):
        return len(arr)
    raise TypeError("Rows expects an array")


@register_builtin("SUM", arg_count=1)
def builtin_sum(args):
    if is_sparse_array(args):
        return sum(args.values())
    if isinstance(args, dict) and 'array' in args:
        return sum(args['array'])
    if isinstance(args, (list, tuple, set)):
        return sum(args)
    return sum([args])


@register_vectorized_builtin("LEN", aliases=["Text.Len"], arg_count=1)
def builtin_len(val):
    if isinstance(val, str):
        return len(val)
    else:
        raise TypeError("Len expects a text")


@register_vectorized_builtin("ABS", aliases=["Number.Abs"], arg_count=1)
def builtin_abs(n):
    return abs(n)


@register_vectorized_builtin("POWER", aliases=["Power", "Number.Power"], arg_count=2)
def builtin_power(a, b):
    return math.pow(a, b) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else a ** b


@register_vectorized_builtin("INT", aliases=["Number.Int"], arg_count=1)
def builtin_int(n):
    return int(n)


@register_vectorized_builtin("MID", aliases=["Text.Mid"], arg_count=(2, 3))
def builtin_mid(text, start, length=1):
    if isinstance(text, str):
        start_idx = max(int(start) - 1, 0)
        length = int(length)
        return text[start_idx:start_idx + length]
    else:
        raise TypeError("Mid expects a text as first argument")


@register_vectorized_builtin("TEXTSPLIT", aliases=["Text.Split"], arg_count=2)
def builtin_textsplit(text, delimiter):
    if isinstance(text, str) and isinstance(delimiter, str):
        return text.split(delimiter)
    else:
        raise TypeError("Split expects two text arguments")


@register_builtin("COUNTA", arg_count=1)
def builtin_counta(val):
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


@register_builtin("RANDARRAY", arg_count=1)
def builtin_randarray(n):
    length = int(n)
    return [random.random() for _ in range(length)]


@register_builtin("SORTBY", arg_count=2)
def builtin_sortby(arr, ord_vals):
    arr_list = _to_list(arr)
    ord_list = _to_list(ord_vals)
    if len(arr_list) != len(ord_list):
        raise ValueError("SortBy expects arrays of the same length")
    pairs = list(zip(ord_list, arr_list))
    pairs.sort(key=lambda p: p[0])
    return [v for _, v in pairs]


@register_builtin("TRANSPOSE", arg_count=1)
def builtin_transpose(arr):
    # Sparse dict: swap the first two tuple indices.
    if is_sparse_array(arr):
        result = {}
        for k, v in arr.items():
            if len(k) >= 2:
                result[(k[1], k[0]) + k[2:]] = v
            else:
                result[k] = v
        return result
    # Structured dict with flat array + shape: transpose the first two axes.
    if isinstance(arr, dict) and 'array' in arr:
        flat = list(arr['array'])
        shape = list(arr.get('shape', []))
        if len(shape) < 2:
            return arr
        s0, s1 = shape[0], shape[1]
        new_shape = [shape[1], shape[0]] + shape[2:]
        total = 1
        for s in shape:
            total *= s
        strides = []
        acc = 1
        for s in shape:
            strides.append(acc)
            acc *= s
        new_flat = []
        for n_idx in range(total):
            rem = n_idx
            idxs = []
            for s in new_shape:
                idxs.append(rem % s)
                rem //= s
            old_idxs = [idxs[1], idxs[0]] + idxs[2:]
            old_flat = sum(oi * st for oi, st in zip(old_idxs, strides))
            new_flat.append(flat[old_flat])
        return {'array': list(new_flat), 'shape': list(new_shape),
                'original_shape': list(new_shape)}
    # Flat list: treat as a row vector → single-column result.
    if isinstance(arr, list):
        if not arr or not any(isinstance(v, list) for v in arr):
            return [[v] for v in arr]
        # Nested list: swap first two axes.
        s0 = len(arr)
        s1 = len(arr[0]) if s0 else 0
        new_arr = [[arr[r][c] for r in range(s0)] for c in range(s1)]
        return new_arr
    return arr


@register_builtin("MIN", aliases=["Number.Min"], arg_count=1)
def builtin_min(args):
    # Flatten single array arg
    if isinstance(args, (list, dict)):
        args = tuple(_to_list(args))
    return min(args)


@register_builtin("MAX", aliases=["Number.Max"], arg_count=1)
def builtin_max(args):
    if isinstance(args, (list, dict)):
        args = tuple(_to_list(args))
    return max(args)


# Math aliases via math module (SIN, COS, etc. are available as math.sin) – vectorized
for _name in ["sqrt", "sin", "cos", "tan", "log", "exp", "asin", "acos", "atan"]:
    fn = getattr(math, _name, None)
    if fn:
        register_vectorized_builtin(_name, aliases=["Number." + _name], arg_count=1)(lambda *a, _fn=fn, **kw: _fn(*a))

# For abs/int/float/str, use Python builtins
for _name in ["str", "int", "float", "abs"]:
    fn = getattr(__builtins__, _name, None)
    if fn:
        register_vectorized_builtin(_name, aliases=["Number." + _name], arg_count=1)(lambda *a, _fn=fn, **kw: _fn(*a))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _is_array_arg(val):
    return isinstance(val, (list, tuple)) or is_sparse_array(val) or (isinstance(val, dict) and 'array' in val)

def _broadcast_builtin(fn, args, evaluator, fn_name=None):
    # Only broadcast for builtins explicitly marked as vectorized
    if fn_name and fn_name.lower() not in VECTORIZED:
        return None
    # Check if any arg is an array; if so, apply element-wise with broadcasting
    array_args = []
    for idx, arg in enumerate(args):
        if _is_array_arg(arg):
            # Convert to list for uniform handling
            if is_sparse_array(arg):
                # For sparse, flatten to list ordered by key
                vals = [arg[k] for k in sorted(arg.keys())]
                array_args.append((idx, vals))
            elif isinstance(arg, dict) and 'array' in arg:
                array_args.append((idx, list(arg['array'])))
            elif isinstance(arg, (list, tuple)):
                array_args.append((idx, list(arg)))
            else:
                array_args.append((idx, _to_list(arg)))
    if not array_args:
        return None  # No array, call normally
    # Check shapes: all array args must have same length or be scalar-broadcast
    # If one is scalar (not in array_args), it will be broadcast
    # For multiple array args, they must have same length
    lengths = {len(vals) for _, vals in array_args}
    if len(lengths) > 1:
        # Mismatched array lengths -> error
        raise ValueError(f"Array arguments must have same shape for element-wise operation, got lengths {lengths}")
    count = next(iter(lengths)) if lengths else 0
    results = []
    for i in range(count):
        elem_args = []
        for idx, arg in enumerate(args):
            # Check if this arg was an array
            is_array_idx = any(idx == a_idx for a_idx, _ in array_args)
            if is_array_idx:
                # Find the array vals for this idx
                for a_idx, vals in array_args:
                    if a_idx == idx:
                        elem_args.append(vals[i])
                        break
            else:
                elem_args.append(arg)
        results.append(fn(*elem_args))
    return results


def get_builtin_functions(evaluator, scope=None, line_number=None):
    """
    Return a dict of name -> callable for the evaluator.
    Each callable is wrapped so that _evaluator/_scope/_line_number are
    injected and array arguments are broadcast element-wise.
    """
    wrapped = {}
    for name, fn in BUILTINS.items():
        def _make_wrapper(fn, fn_name=name):
            def wrapper(*args, **kwargs):
                # Compile-time arity check: the number of top-level GridLang
                # arguments must match the builtin's declared parameter count
                # (a single bracket array/range argument counts as one).
                _check_arity(fn_name, len(args), line_number)
                try:
                    # Array broadcasting: if any arg is array, apply element-wise
                    # (except for reductions like SUM).
                    array_result = _broadcast_builtin(
                        fn, args, evaluator, fn_name=fn_name)
                    if array_result is not None:
                        return array_result
                    return fn(*args)
                except TypeError as e:
                    # A genuine operand/type mismatch: report #TYPE/I instead of
                    # raising, so builtins degrade to a sticky type error value.
                    return error_value(TYPE_ERROR)
            return wrapper
        wrapped[name] = _make_wrapper(fn)
    wrapped['math'] = math
    return wrapped
