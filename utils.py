import re


def validate_cell_ref(cell_ref):
    # Convert to uppercase for validation
    cell_ref_upper = cell_ref.upper()
    if not re.match(r'^[A-Z]+\d+$', cell_ref_upper):
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


def offset_cell(cell_ref, col_offset, row_offset):
    col, row = split_cell(cell_ref)
    col_num = col_to_num(col)
    new_col_num = col_num + col_offset
    if new_col_num < 1:
        raise ValueError("Column offset results in invalid column")
    new_row = int(row) + row_offset
    if new_row < 1:
        raise ValueError("Row offset results in invalid row")
    return f"{num_to_col(new_col_num)}{new_row}"


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
