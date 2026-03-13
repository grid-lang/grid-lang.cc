# <xaiArtifact artifact_id="9592cb0a-5bd6-4283-b60e-23d5eb595d32" artifact_version_id="3ea58d55-8ecd-4316-8c99-7f17ecd6aaba" title="main.py" contentType="text/python">
import sys
from compiler import GridLangCompiler


def _parse_cli_args(args):
    """Parse runner flags while preserving positional program arguments."""
    debug_enabled = False
    program_args = []
    passthrough_args = False

    if args and args[0] in ("-h", "--help"):
        return None, [], False, True
    if not args:
        return None, [], False, False

    grid_file = args[0]
    for arg in args[1:]:
        if passthrough_args:
            program_args.append(arg)
        elif arg == "--":
            passthrough_args = True
        elif arg == "--debug":
            debug_enabled = True
        else:
            program_args.append(arg)
    return grid_file, program_args, debug_enabled, False


def run_grid_program(args):
    """Run a Grid program with command line arguments"""
    grid_file, program_args, debug_enabled, show_help = _parse_cli_args(args)
    if show_help:
        print("Usage: grid <grid_file> [arg1] [arg2] ... [--debug]")
        return
    if not grid_file:
        print("Usage: grid <grid_file> [arg1] [arg2] ... [--debug]")
        return

    try:
        with open(grid_file, 'r') as file:
            code = file.read()

        compiler = GridLangCompiler()
        # Prompt for missing inputs when no CLI arguments are provided and stdin is interactive
        compiler.prompt_missing_inputs = (
            len(program_args) == 0 and sys.stdin.isatty())
        compiler.run(code, program_args)
        if debug_enabled:
            csv_path = compiler.export_to_csv(grid_file)
            print(f"Debug CSV written to {csv_path}")

    except FileNotFoundError:
        print(f"Error: Grid file '{grid_file}' not found")
    except Exception as e:
        print(f"Error running Grid program: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    try:
        run_grid_program(sys.argv[1:])
    except Exception:
        sys.exit(1)


def main():
    """Console-script entrypoint for the grid CLI."""
    run_grid_program(sys.argv[1:])
