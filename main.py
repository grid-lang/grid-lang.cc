from compiler import GridLangCompiler
import sys
import os


def run_tests():
    args = sys.argv[1:]
    debug = '--debug' in args
    filename = None

    # Handle help or no arguments
    if len(args) == 0 or args[0] == "--help":
        print("""
GridLang Compiler
================

The `grid` command compiles `.grid` files using the GridLangCompiler, processing
cell-based data and printing results. It supports optional flags for running and
debugging.

Usage:
    grid [-r] <filename.grid> [--debug]

Options:
    -r              Optional flag to run the compiler (same as omitting -r).
    --debug         Export results to a CSV file (requires implementation in compiler.py).
    --help          Display this help message.

Arguments:
    <filename.grid> Path to a `.grid` file containing cell assignments (e.g., A1 = 42).

File Format:
    A `.grid` file contains cell assignments in the format `<cell> = <value>`.
    Example (example.grid):
        A1 = 42
        B2 = "Hello"

Examples:
    1. Compile a `.grid` file:
        grid example.grid
        Output:
            Imported gridlang.compiler successfully
            Compiling example.grid...
            A1 = 42
            B2 = Hello

    2. Compile with debug mode to export CSV:
        grid example.grid --debug
        Output:
            ... (same as above)
            CSV exported to: example.csv

    3. Use -r flag (equivalent to no -r):
        grid -r example.grid

Notes:
    - Ensure the `.grid` file exists in the current directory or provide a full path.
    - The --debug option requires the `export_to_csv` method in `compiler.py` to be implemented.
    - Run from the `gridlang_public` directory or after installing via `pip install .`.

For issues, see the README.md in the gridlang_public repository.
        """)
        return

    if args[0] == "-r":
        if len(args) < 2:
            print("Usage: grid -r <filename.grid> [--debug]")
            return
        filename = args[1]
    else:
        filename = args[0]

    if not os.path.isfile(filename):
        print(f"File not found: {filename}")
        return

    with open(filename, 'r') as file:
        code = file.read()

    print(f"Compiling {filename}...\n", code)
    compiler = GridLangCompiler()
    result = compiler.run(code)

    if debug:
        output_csv = filename.replace('.grid', '.csv')
        # Requires implementation in compiler.py
        compiler.export_to_csv(output_csv)
        print(f"CSV exported to: {output_csv}")

    for cell in sorted(compiler.grid):
        print(f"{cell} = {compiler.grid[cell]}")


if __name__ == '__main__':
    try:
        run_tests()
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        sys.exit(1)
