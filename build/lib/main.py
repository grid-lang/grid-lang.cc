import sys
from compiler import GridLangCompiler


def run_grid_program(args):
    """Run a Grid program with command line arguments"""
    if len(args) < 1:
        print("Usage: python main.py <grid_file> [arg1] [arg2] ... [--debug]")
        return

    # Check for debug flag
    debug = '--debug' in args
    if debug:
        args.remove('--debug')

    grid_file = args[0]
    program_args = args[1:]  # Command line arguments for the program

    try:
        with open(grid_file, 'r') as file:
            code = file.read()

        if debug:
            print("DEBUG: Debug mode enabled - will export CSV")

        compiler = GridLangCompiler()
        result = compiler.run(code, program_args)

        # Print results to stdout
        printed = False
        try:
            if getattr(compiler, 'grid', None):
                for cell, value in sorted(compiler.grid.items()):
                    print(f"{cell} = {value}")
                printed = True
        except Exception:
            pass

        if not printed:
            try:
                global_scope = compiler.get_global_scope()
                output_vars = []
                if hasattr(global_scope, 'output_variables') and global_scope.output_variables:
                    output_vars = sorted(global_scope.output_variables)
                for out_var in output_vars:
                    try:
                        val = global_scope.get(out_var)
                        if val is not None:
                            print(f"{out_var} = {val}")
                            printed = True
                    except Exception:
                        pass
            except Exception:
                pass

        # Export CSV if debug mode is enabled
        if debug:
            output_csv = grid_file.replace('.grid', '.csv')
            compiler.export_to_csv(output_csv)
            print(f"CSV exported to: {output_csv}")

    except FileNotFoundError:
        print(f"Error: Grid file '{grid_file}' not found")
    except Exception as e:
        print(f"Error running Grid program: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point for the grid command"""
    try:
        args = sys.argv[1:]

        # Handle help
        if not args or args[0] == "--help":
            print("""
GridLang Compiler
================

The `grid` command compiles `.grid` files using the GridLangCompiler, processing
cell-based data and printing results. It supports optional flags for running and
debugging.

Usage:
    grid <filename.grid> [arg1] [arg2] ... [--debug]

Options:
    --debug         Export results to a CSV file
    --help          Display this help message

Arguments:
    <filename.grid> Path to a `.grid` file containing GridLang code
    [arg1] [arg2]   Optional arguments to pass to the Grid program

Examples:
    1. Run a Grid program:
        grid example.grid

    2. Run with arguments:
        grid binsearch.grid 22

    3. Run with debug mode to export CSV:
        grid example.grid --debug

For issues, see the README.md in the repository.
            """)
            return

        # Check if this is a Grid program execution
        if args and not args[0].startswith('-'):
            # First argument is a file, check if it's a .grid file
            if args[0].endswith('.grid'):
                run_grid_program(args)

    except Exception as e:
        print(f"DEBUG: Fatal error in main execution: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
