import re


def validate_cell_ref(cell_ref):
    if not re.match(r'^[A-Z]+\d+$', cell_ref):
        raise ValueError(f"Invalid cell reference: '{cell_ref}'")


def split_cell(cell_ref):
    m = re.match(r'^([A-Z]+)(\d+)$', cell_ref)
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
