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
VECTORIZED = set()  # names (lower) of builtins that are applied element-wise to array args
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


def _strip_list(val):
    # for single-arg sum handling
    return _to_list(val)


# ---------------------------------------------------------------------------
# Builtins
# ---------------------------------------------------------------------------

@register_builtin("ROWS", arg_count=1)
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


@register_vectorized_builtin("LEN", aliases=["Text.Len"], arg_count=1)
def builtin_len(val, _evaluator=None, _scope=None, _line_number=None):
    if isinstance(val, str):
        return len(val)
    else:
        raise TypeError("Len expects a text")


@register_vectorized_builtin("ABS", aliases=["Number.Abs"], arg_count=1)
def builtin_abs(n, _evaluator=None, _scope=None, _line_number=None):
    return abs(n)


@register_vectorized_builtin("POWER", aliases=["Power", "Number.Power"], arg_count=2)
def builtin_power(a, b, _evaluator=None, _scope=None, _line_number=None):
    return math.pow(a, b) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else a ** b


@register_vectorized_builtin("INT", aliases=["Number.Int"], arg_count=1)
def builtin_int(n, _evaluator=None, _scope=None, _line_number=None):
    return int(n)


@register_vectorized_builtin("MID", aliases=["Text.Mid"], arg_count=(2, 3))
def builtin_mid(text, start, length=1, _evaluator=None, _scope=None, _line_number=None):
    s = str(text)
    start_idx = max(int(start) - 1, 0)
    length = int(length)
    return s[start_idx:start_idx + length]


@register_vectorized_builtin("TEXTSPLIT", aliases=["Text.Split"], arg_count=2)
def builtin_textsplit(text, delimiter, _evaluator=None, _scope=None, _line_number=None):
    return str(text).split(str(delimiter))


@register_builtin("COUNTA", arg_count=1)
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


@register_builtin("RANDARRAY", arg_count=1)
def builtin_randarray(n, _evaluator=None, _scope=None, _line_number=None):
    length = int(n)
    return [random.random() for _ in range(length)]


@register_builtin("SORTBY", arg_count=2)
def builtin_sortby(arr, ord_vals, _evaluator=None, _scope=None, _line_number=None):
    arr_list = _to_list(arr)
    ord_list = _to_list(ord_vals)
    if len(arr_list) != len(ord_list):
        raise ValueError("SortBy expects arrays of the same length")
    pairs = list(zip(ord_list, arr_list))
    pairs.sort(key=lambda p: p[0])
    return [v for _, v in pairs]


@register_builtin("TRANSPOSE", arg_count=1)
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


@register_builtin("MIN", arg_count=1)
def builtin_min(*args, _evaluator=None, _scope=None, _line_number=None):
    # Flatten single array arg
    if len(args) == 1 and isinstance(args[0], (list, dict)):
        args = tuple(_to_list(args[0]))
    return min(args)


@register_builtin("MAX", arg_count=1)
def builtin_max(*args, _evaluator=None, _scope=None, _line_number=None):
    if len(args) == 1 and isinstance(args[0], (list, dict)):
        args = tuple(_to_list(args[0]))
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
# SUM – handled specially because it has 3 syntaxes: sum[A1:B2], sum{a,b}, sum(...)
# We keep the range/var helpers in expression.py, but expose a sum function
# for sum(...) that delegates to them when a single string arg is given.
# ---------------------------------------------------------------------------

@register_builtin("SUM", arg_count=1)
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
    return sum(args)


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
                # Array broadcasting: if any arg is array, apply element-wise (except for reductions like SUM)
                array_result = _broadcast_builtin(fn, args, evaluator, fn_name=fn_name)
                if array_result is not None:
                    return array_result
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
                            array_result2 = _broadcast_builtin(fn, args, evaluator, fn_name=fn_name)
                            if array_result2 is not None:
                                return array_result2
                            return fn(*args, **kwargs)
                        except TypeError:
                            array_result3 = _broadcast_builtin(lambda *a: fn(*a), args, evaluator, fn_name=fn_name)
                            if array_result3 is not None:
                                return array_result3
                            return fn(*args)
                    raise
            return wrapper
        wrapped[name] = _make_wrapper(fn)
    wrapped['math'] = math
    return wrapped
