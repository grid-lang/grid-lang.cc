import sys
import os
from gridlang.compiler import GridLangCompiler

def main():
    if len(sys.argv) < 2:
        print("Usage: grid <filename.grid> [--debug]")
        return

    filename = sys.argv[1]
    debug = '--debug' in sys.argv

    if not os.path.isfile(filename):
        print(f"File not found: {filename}")
        return

    with open(filename, 'r') as file:
        code = file.read()

    compiler = GridLangCompiler()
    compiler.run(code)

    if debug:
        output_csv = filename.replace('.grid', '.csv')
        compiler.export_to_csv(output_csv)
        for cell in sorted(compiler.grid):
            print(f"{cell} = {compiler.grid[cell]}")
        print(f"CSV exported to: {output_csv}")
    else:
        for cell in sorted(compiler.grid):
            print(f"{cell} = {compiler.grid[cell]}")

if __name__ == '__main__':
    main()
