import re

from units import UNIVERSAL_ZERO, NA_ERROR

_DEFINITION_AND_SPLIT = re.compile(r'\s+and\s+(?![<>=]|not\b)', re.I)

_ARRAY_CELL_INDEX = re.compile(
    r'\b([A-Za-z_][A-Za-z0-9_]*)\s*!\s*\[[^\]\[]*\]')


def strip_array_cell_indices(expr):
    """Rewrite 'var![...]' array-cell index expressions to the bare variable
    name so cell addresses inside them are not mistaken for standalone grid
    cells when extracting dependencies."""
    return _ARRAY_CELL_INDEX.sub(r'\1', expr)


def iter_interpolation_placeholders(text):
    """Yield the contents of {placeholder} groups inside interpolated strings.

    Interpolated strings are delimited by $"..." (or $'...'); only braces
    inside those spans are scanned. Array-literal braces and quoted text
    values that merely look like identifiers are therefore ignored.
    """
    n = len(text)
    scan = 0
    while True:
        i_dq = text.find('$"', scan)
        i_sq = text.find("$'", scan)
        if i_dq == -1 and i_sq == -1:
            return
        if i_sq == -1 or (i_dq != -1 and i_dq < i_sq):
            start, marker = i_dq, '"'
        else:
            start, marker = i_sq, "'"
        j = start + 2
        while j < n:
            if text[j] == marker and text[j - 1] != '\\':
                break
            j += 1
        if j >= n:
            return
        for ph in re.findall(r'\{([^{}]*)\}', text[start + 2:j]):
            yield ph
        scan = j + 1


def split_var_defs(s):
    """Split a variable-definition list on 'and' that separates definitions.

    An 'and' introducing a further comparison constraint (=, <, >, <>, >=,
    <=) or a negated constraint ('not') is left intact. A '?' truth-test
    value expression (e.g. "? a = 5 and 1 = 1") is a single value, so any
    'and' inside it is left intact too.
    """
    marker = _truth_test_begin(s)
    if marker is None:
        return _DEFINITION_AND_SPLIT.split(s)
    head = s[:marker]
    tail = s[marker:]
    parts = _DEFINITION_AND_SPLIT.split(head)
    parts[-1] = parts[-1] + tail
    return [part for part in parts if part]


def _truth_test_begin(s):
    """Return the index of the '?' that starts a truth-test value.

    A truth-test is a '?' directly following the '=' that introduces a
    variable's value expression. Since it extends to the end of the string,
    only the first occurrence matters.
    """
    in_quote = False
    quote_char = None
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in ('"', "'"):
            if not in_quote:
                in_quote = True
                quote_char = ch
            elif quote_char == ch:
                in_quote = False
                quote_char = None
            i += 1
            continue
        if not in_quote and ch == '=':
            j = i + 1
            while j < len(s) and s[j].isspace():
                j += 1
            if j < len(s) and s[j] == '?':
                return j
        i += 1
    return None


def validate_cell_ref(cell_ref):
    """Validate a single 2D cell reference (e.g. 'A1').

    Delegates to parse_address (the canonical address parser) and rejects
    dotted N-D addresses and bare-number segments.
    """
    if not isinstance(cell_ref, str) or '.' in cell_ref:
        raise ValueError(f"Invalid cell reference: '{cell_ref}'")
    try:
        indices = parse_address(cell_ref)
    except ValueError:
        raise ValueError(f"Invalid cell reference: '{cell_ref}'")
    if len(indices) != 2:
        raise ValueError(f"Invalid cell reference: '{cell_ref}'")


def split_cell(cell_ref):
    # Convert to uppercase for processing
    cell_ref_upper = cell_ref.upper()
    m = re.match(r'^([A-Z]+)(\d+)$', cell_ref_upper)
    if not m:
        raise ValueError(f"Invalid cell reference: '{cell_ref}'")
    return m.groups()


def col_to_num(col):
    num = 0
    for c in col.upper():
        num = num * 26 + (ord(c) - ord('A') + 1)
    return num


def num_to_col(num):
    col = ""
    while num > 0:
        num, rem = divmod(num - 1, 26)
        col = chr(65 + rem) + col
    return col


def offset_cell(index, col_offset, row_offset):
    """Return the 0-based index tuple offset from ``index`` (row, col) by
    ``(col_offset, row_offset)``.  The canonical internal form is a 0-based
    index tuple; offsets are applied to the 0-based coordinates."""
    row, col = index
    new_col = col + col_offset
    if new_col < 0:
        raise ValueError("Column offset results in invalid column")
    new_row = row + row_offset
    if new_row < 0:
        raise ValueError("Row offset results in invalid row")
    return (new_row, new_col)


# An N-D address is a dot-separated sequence of segments. Each segment is
# either a 2D cell (letters+digits, contributing a (row, col) pair) or a bare
# integer (contributing a single index). For example 'A3.B4.8' -> [3, 1, 4, 2, 8].
_ADDRESS_FRAGMENT = r'[A-Za-z]+\d+(?:\.[A-Za-z]+\d+|\.[0-9]+)*'
_ADDRESS_PATTERN = re.compile(r'^' + _ADDRESS_FRAGMENT + r'$')


def is_address(value):
    """Return True if value is a cell address, possibly dotted for N-D use."""
    if not isinstance(value, str) or not value:
        return False
    return bool(_ADDRESS_PATTERN.match(value))


def parse_address(address):
    """Parse an N-D cell address into 1-based indices.

    'A3' -> [3, 1]; 'A3.B4.8' -> [3, 1, 4, 2, 8].
    Raises ValueError for malformed addresses.
    """
    if not isinstance(address, str) or not address:
        raise ValueError(f"Invalid address: '{address}'")
    indices = []
    for part in address.split('.'):
        if re.match(r'^[A-Za-z]+\d+$', part):
            col, row = split_cell(part)
            indices.append(int(row))
            indices.append(col_to_num(col))
        elif re.match(r'^\d+$', part):
            indices.append(int(part))
        else:
            raise ValueError(f"Invalid address segment: '{part}'")
    if not indices:
        raise ValueError(f"Invalid address: '{address}'")
    return indices


def indices_to_address(indices):
    """Build the canonical dotted address for segment-encoded 1-based indices.

    Inverse of parse_address: index pairs (row, col) become letter+digits
    segments and a trailing lone index becomes a bare number.
    """
    parts = []
    i = 0
    while i < len(indices):
        if i + 1 < len(indices):
            parts.append(f"{num_to_col(indices[i + 1])}{indices[i]}")
            i += 2
        else:
            parts.append(str(indices[i]))
            i += 1
    return '.'.join(parts)


# Wildcard addresses add '*' as a segment. Addresses work in pairs: a cell
# segment like 'C3' contributes the pair (row, col) = [3, 3], '*' contributes
# a wildcard pair ['*', '*'] and a bare number contributes a single index.
# The number may be missing from a pair: a bare letter like 'A' in '[C2.A]'
# contributes a pair with a wildcarded row, so on a 4D array that address
# selects {2, 3, *, 1}. Missing trailing positions are padded with '*' up to
# the array's rank, so on a 6D array '[C3.*.4]' selects {3, 3, *, *, 4, *}.
# Specs are internal 0-based indices: ints are 0-based, '*' is Python-None
# and a range 'n to m' is a pair (n-1, m-1).
_ADDRESS_FRAGMENT_WILDCARD = r'(?:[A-Za-z]+\d*|\*)(?:\.[A-Za-z]+\d*|\.[0-9]+|\.\*)*'
_ADDRESS_PATTERN_WILDCARD = re.compile(r'^' + _ADDRESS_FRAGMENT_WILDCARD + r'$')


def _has_bare_letter_segment(value):
    """True when any dot-separated segment is letters only (number missing)."""
    return any(re.fullmatch(r'[A-Za-z]+', part) for part in value.split('.'))


def is_wildcard_address(value):
    """Return True if value is a dotted N-D address with wildcarding.

    Wildcarding comes from '*' segments or from a pair whose number is
    missing (a bare-letter segment inside a dotted address).
    """
    if not isinstance(value, str) or not value:
        return False
    if not _ADDRESS_PATTERN_WILDCARD.match(value):
        return False
    if '*' in value:
        return True
    return '.' in value and _has_bare_letter_segment(value)


def parse_wildcard_address(address):
    """Parse a wildcarded N-D address into internal 0-based index specs.

    Segments are read in pairs: a cell segment like 'C3' contributes the pair
    (row, col) -> [2, 2] (0-based); '*' contributes a wildcard pair [None,
    None]; a bare number contributes a single index; a bare letter like 'A'
    contributes a pair whose row is wildcarded -> [None, col]. Returns a list
    whose entries are 0-based ints, None (wildcard), or (n-1, m-1) range pairs.
    """
    if not isinstance(address, str) or not address:
        raise ValueError(f"Invalid address: '{address}'")
    if not _ADDRESS_PATTERN_WILDCARD.match(address):
        raise ValueError(f"Invalid wildcard address: '{address}'")
    specs = []
    for part in address.split('.'):
        if part == '*':
            specs.extend([None, None])
        elif re.match(r'^[A-Za-z]+\d+$', part):
            col, row = split_cell(part)
            specs.extend([int(row) - 1, col_to_num(col) - 1])
        elif re.match(r'^[A-Za-z]+$', part):
            specs.extend([None, col_to_num(part) - 1])
        elif re.match(r'^\d+$', part):
            specs.append(int(part) - 1)
        else:
            raise ValueError(f"Invalid address segment: '{part}'")
    if not specs:
        raise ValueError(f"Invalid address: '{address}'")
    return specs


def prod(iterable):
    result = 1
    for x in iterable:
        result *= x
    return result


def public_type_fields(type_def: dict):
    """Return only the user-declared public fields of a type (ignore internal/hidden metadata keys)."""
    if not isinstance(type_def, dict):
        return {}
    hidden = type_def.get('_hidden_fields', set())
    if not isinstance(hidden, (set, list, tuple)):
        hidden = set()
    hidden = {str(h).lower() for h in hidden}
    return {
        k: v for k, v in type_def.items()
        if not str(k).startswith('_') and str(k).lower() not in hidden
    }


def object_public_keys(obj: dict):
    """Return the set of keys on an object, excluding internal/hidden metadata keys."""
    if not isinstance(obj, dict):
        return set()
    hidden = obj.get('_hidden_fields', set())
    if not isinstance(hidden, (set, list, tuple)):
        hidden = set()
    hidden = {str(h).lower() for h in hidden}
    return {
        k for k in obj.keys()
        if not str(k).startswith('_')
        and str(k) != 'grid'
        and not str(k).startswith('$')
        and str(k).lower() not in hidden
    }


def get_case_insensitive_key(mapping, name):
    """Return the actual key in mapping matching name case-insensitively, or None."""
    if not isinstance(mapping, dict):
        return None
    name_lower = str(name).lower()
    for key in mapping.keys():
        if str(key).lower() == name_lower:
            return key
    return None


def get_case_insensitive_value(mapping, name, default=None):
    """Return the value for name in mapping using case-insensitive key lookup."""
    key = get_case_insensitive_key(mapping, name)
    if key is None:
        return default
    return mapping.get(key, default)


def public_object_view(obj):
    """Return a view of an object containing only public fields (recursively)."""
    if not isinstance(obj, dict):
        return obj
    result = {}
    for key in object_public_keys(obj):
        val = obj.get(key)
        if isinstance(val, dict):
            result[key] = public_object_view(val)
        elif isinstance(val, list):
            result[key] = [public_object_view(v) for v in val]
        else:
            result[key] = val
    return result


def is_sparse_array(value):
    """True for sparse no-dim array storage: a dict keyed by 0-based index
    tuples (one key per populated cell, no empty entries)."""
    return (
        isinstance(value, dict)
        and bool(value)
        and all(isinstance(k, tuple) for k in value.keys())
    )


def format_display_value(value, sig_digits=15):
    """Format values for display by trimming floating-point artifacts."""
    if value is UNIVERSAL_ZERO:
        return "{ }"
    if isinstance(value, float):
        if value != value:
            return "nan"
        if value == float('inf'):
            return "inf"
        if value == float('-inf'):
            return "-inf"
        formatted = format(value, f".{sig_digits}g")
        if formatted in ("-0", "-0.0"):
            formatted = "0"
        return formatted
    if isinstance(value, (int, bool)):
        return str(value)
    if value is None:
        return NA_ERROR
    if is_sparse_array(value):
        return _format_sparse_array(value, sig_digits)
    if isinstance(value, dict) and 'array' in value:
        flat = list(value['array'])
        shape = list(value.get('shape') or value.get('original_shape') or [])
        stride = shape[0] if len(shape) > 1 and shape[0] else (len(flat) or 1)
        rows = [flat[i:i + stride] for i in range(0, len(flat), stride)]
        return _format_nd(rows, sig_digits)
    if isinstance(value, dict):
        items = []
        for k, v in value.items():
            items.append(f"{k}: {format_display_value(v, sig_digits=sig_digits)}")
        return "{" + ", ".join(items) + "}"
    if isinstance(value, (list, tuple)):
        if value and any(isinstance(v, (list, tuple)) for v in value):
            return _format_nd([_flatten_scalars(v) for v in value], sig_digits)
        return _format_nd([value], sig_digits)
    return str(value)


def _format_sparse_array(value, sig_digits):
    """Render a sparse no-dim array (dict keyed by 0-based index tuples) in
    the ``{ ... }`` form.  Rows render at their actual length: trailing
    empty cells are suppressed, interior gaps collapse into comma runs.
    Empty rows between the first and last populated rows render as a pipe."""
    rank = len(next(iter(value.keys())))
    if rank <= 1:
        items = [value[k] for k in sorted(value.keys())]
        return _format_nd([items], sig_digits)
    rows_map = {}
    for k, v in value.items():
        rows_map.setdefault(k[0], {})[k[1]] = v
    grid_rows = []
    for row in range(max(rows_map) + 1):
        if row not in rows_map:
            grid_rows.append([])
            continue
        cols = rows_map[row]
        row_vals = [cols.get(c) for c in range(max(cols) + 1)]
        while row_vals and row_vals[-1] is None:
            row_vals.pop()
        grid_rows.append(row_vals)
    return _format_nd(grid_rows, sig_digits)


def _format_nd(rows, sig_digits):
    """Render array rows inside braces with a space before each value, pipe
    and closing brace, e.g. ``{ 1, 2 | 3, 4 }``.  Empty values render as
    nothing, so missing entries collapse into runs of commas and pipes."""
    rendered = []
    for row in rows:
        parts = ["" if v is None else format_display_value(v, sig_digits=sig_digits) for v in row]
        r = " " + parts[0] if parts else ""
        for p in parts[1:]:
            r += ("," if p == "" else ", " + p)
        rendered.append(r)
    out = rendered[0] if rendered else ""
    for r in rendered[1:]:
        out += " |" + r
    return "{" + out + " }"


def _flatten_scalars(value):
    """Flatten nested lists into a flat list of scalar values."""
    if isinstance(value, (list, tuple)):
        return [x for v in value for x in _flatten_scalars(v)]
    return [value]
