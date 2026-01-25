# <xaiArtifact artifact_id="9592cb0a-5bd6-4283-b60e-23d5eb595d32" artifact_version_id="3ea58d55-8ecd-4316-8c99-7f17ecd6aaba" title="main.py" contentType="text/python">
import sys
from compiler import GridLangCompiler


def run_grid_program(args):
    """Run a Grid program with command line arguments"""
    if len(args) < 1 or args[0] in ("-h", "--help"):
        print("Usage: grid <grid_file> [arg1] [arg2] ...")
        return

    grid_file = args[0]
    program_args = args[1:]  # Command line arguments for the program

    try:
        with open(grid_file, 'r') as file:
            code = file.read()


        compiler = GridLangCompiler()
        # Prompt for missing inputs when no CLI arguments are provided and stdin is interactive
        compiler.prompt_missing_inputs = (
            len(program_args) == 0 and sys.stdin.isatty())
        result = compiler.run(code, program_args)


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
