import csv
import re

def export_to_csv(grid, filename):
    max_col = 0
    max_row = 0
    cell_data = {}

    for cell, value in grid.items():
        try:
            col_str, row = split_cell(cell)
        except ValueError:
            continue
        col = col_to_num(col_str)
        max_col = max(max_col, col)
        max_row = max(max_row, row)
        cell_data[(row, col)] = value

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        for r in range(1, max_row + 1):
            row_data = []
            for c in range(1, max_col + 1):
                val = cell_data.get((r, c), '')
                row_data.append(val if val is not None else '')
            writer.writerow(row_data)

def split_cell(ref):
    match = re.match(r'([A-Z]+)(\d+)', ref)
    if not match:
        raise ValueError(f"Invalid cell reference: {ref}")
    return match.group(1), int(match.group(2))

def col_to_num(col):
    num = 0
    for c in col:
        num = num * 26 + ord(c) - ord('A') + 1
    return num

def num_to_col(num):
    result = ''
    while num:
        num, rem = divmod(num - 1, 26)
        result = chr(65 + rem) + result
    return result
