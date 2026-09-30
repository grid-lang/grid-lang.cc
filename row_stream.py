"""Row-stream core: the ordered part of the language.

Standalone ``For x in ...`` lines are *producers*: each pushes a row stream,
and consecutive producer lines zip together into merged rows. A dynamic cell
binding ``[{i:A}{j}] := expr``, a ``Push`` or a ``Return`` is a *consumer* (an
effect) that fires on every merged row and may appear before or after its
producers (bindings may be out of order; producers and effects run in order).
A standalone ``If <cond>`` line is a *guard*: its condition is re-evaluated per
row, and consumers only run on rows where every guard holds. Guards are
position-independent within the group (``If i <= 2`` may follow the consumers
it guards). ``When`` blocks are triggers, not guards, and are handled by the
executor separately.

This module stays pure (no executor dependencies): it turns a run of
producer/rule lines into merged row dicts. The executor then executes each
merged row against the consumers with the row variables in scope.
"""

import re

_CLAUSE_RE = re.compile(
    r'^([A-Za-z_]\w*)\s+in\s+(.+?)(?:\s+index\s+([A-Za-z_]\w*))?$')
_RANGE_STEP_RE = re.compile(
    r'^(-?\d+)\s+to\s+(-?\d+)(?:\s+step\s+(-?\d+))?\s*$')
_RANGE_DYNAMIC_RE = re.compile(r'^(-?\d+)\s+to\s+([A-Za-z_]\w*)\s*$')
_SET_RE = re.compile(r'^\{([^{}]*)\}$')
_RULE_RE = re.compile(r'^\s*\[([^\]\n]*)\]\s*:=\s*(.+)$')
_ID_RE = re.compile(r'^[A-Za-z_]\w*$')


def parse_producer(line):
    """Parse a standalone ``For x in <range>`` (+ ``and``-joined clauses) line.

    Returns a list of ``(var, range_expr, index_var)`` clauses, or None when
    the line is not a supported standalone producer (block ``do`` headers,
    declaration ``For x = ...``, unsupported ranges).
    """
    m = re.match(r'^\s*[Ff]or\s+(.+)$', line)
    if not m:
        return None
    body = m.group(1).strip()
    if not body or re.search(r'\bdo\s*$', body, re.I):
        return None
    clauses = []
    for part in re.split(r'\s+and\s+', body, flags=re.I):
        part = part.strip()
        cm = _CLAUSE_RE.match(part)
        if not cm:
            return None
        clauses.append((cm.group(1).strip(), cm.group(2).strip(),
                        cm.group(3)))
    if not clauses:
        return None
    return clauses


def parse_guard(line):
    """Parse a standalone ``If <cond>`` guard line (no ``then``).

    Returns the condition string, or None when the line is not a standalone
    guard (inline ``If ... then`` headers or block headers are excluded).
    """
    m = re.match(r'^\s*[Ii]f\s+(.+?)\s*$', line)
    if not m:
        return None
    cond = m.group(1).strip()
    if not cond:
        return None
    if re.search(r'\bthen\b', cond, re.I):
        return None
    return cond


def iter_range(range_expr, ctx, env=None):
    """Expand a range expression to a list of values, or None when unsupported.

    ``ctx`` is the partially built row (earlier clauses in the same line), so
    dynamic upper bounds like ``1 to k`` resolve against it. ``env`` carries
    outer bindings for bare-identifier sources like ``For x in arr``.
    """
    sm = _RANGE_STEP_RE.match(range_expr)
    if sm:
        start, end = int(sm.group(1)), int(sm.group(2))
        step = int(sm.group(3)) if sm.group(3) else 1
        if step == 0:
            return None
        return list(range(start, end + (1 if step > 0 else -1), step))
    dm = _RANGE_DYNAMIC_RE.match(range_expr)
    if dm:
        start = int(dm.group(1))
        bound = ctx.get(dm.group(2))
        if bound is None and env is not None:
            bound = env.get(dm.group(2))
        if bound is None:
            return None
        try:
            end = int(bound)
        except (TypeError, ValueError):
            return None
        return list(range(start, end + 1))
    sm2 = _SET_RE.match(range_expr)
    if sm2:
        raw_items = [item.strip() for item in sm2.group(1).split(',')]
        values = []
        for item in raw_items:
            if not item:
                continue
            if (item.startswith('"') and item.endswith('"')) or \
                    (item.startswith("'") and item.endswith("'")):
                values.append(item[1:-1])
            else:
                try:
                    values.append(float(item) if '.' in item else int(item))
                except ValueError:
                    values.append(item)
        return values
    if _ID_RE.match(range_expr):
        value = ctx.get(range_expr)
        if value is None and env is not None:
            value = env.get(range_expr)
        if isinstance(value, (list, tuple)):
            return list(value)
        if isinstance(value, dict) or value is None:
            return None
        return [value]
    return None


def row_sequences(clauses, env=None, resolve=None):
    """Expand ``and``-joined clauses into a flat row stream.

    Clauses expand sequentially (first clause outermost), so
    ``k in 2 to 3 and j in 1 to k`` yields the flattened sequence
    ``(2,1), (2,2), (3,1), (3,2), (3,3)`` for ``(k, j)`` in that order. An
    ``index`` var, when present, counts 1-based positions in the full value
    list including error elements. Returns a list of row dicts, or None when
    any range is unsupported.
    """
    rows = [dict()]
    for var, range_expr, index_var in clauses:
        next_rows = []
        for row in rows:
            values = iter_range(range_expr, row, env)
            if values is None and resolve is not None:
                values = resolve(range_expr, row)
            if values is None:
                return None
            for position, value in enumerate(values, 1):
                next_row = dict(row)
                next_row[var] = value
                if index_var:
                    next_row[index_var] = position
                next_rows.append(next_row)
        rows = next_rows
    return rows


def zip_rows(row_lists):
    """Zip consecutive producer row streams into merged rows (row = dict)."""
    if not row_lists:
        return []
    rows = row_lists[0]
    for row_list in row_lists[1:]:
        rows = [dict(left, **right)
                for left, right in zip(rows, row_list)]
    return rows


def parse_cell_rule(line):
    """Parse a dynamic cell rule ``[{...}...] := expr``.

    Returns a ``(target, value)`` tuple, or None when the line is not a cell
    rule with at least one ``{...}`` interpolation in the target.
    """
    m = _RULE_RE.match(line)
    if not m:
        return None
    target = m.group(1).strip()
    if '{' not in target:
        return None
    if not re.match(r'^\^?[A-Za-z{]', target):
        return None
    return target, m.group(2).strip()