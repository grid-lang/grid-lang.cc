# <xaiArtifact artifact_id="9592cb0a-5bd6-4283-b60e-23d5eb595d32" artifact_version_id="3ea58d55-8ecd-4316-8c99-7f17ecd6aaba" title="main.py" contentType="text/python">
import sys

from compiler import GridLangCompiler
from permissions import format_required_yaml, load_grants
from units import GrantError


USAGE = ("Usage: grid [options] <grid_file> [arg1] [arg2] ...\n"
         "Options:\n"
         "  --debug          Export the resulting grid as CSV after running\n"
         "  --list-required  Print the capabilities the program requires as YAML\n"
         "                   and exit without running it\n"
         "  --grant <spec>   Granted capabilities before running: a YAML file\n"
         "                   path, or inline YAML starting with '-' or a list\n"
         "  --               Everything after this is passed to the program")


def _parse_cli_args(args):
    """Parse runner flags while collecting the grid file and program arguments."""
    debug_enabled = False
    list_required = False
    grant_spec = None
    grid_file = None
    program_args = []
    passthrough_args = False

    i = 0
    while i < len(args):
        arg = args[i]
        if passthrough_args:
            program_args.append(arg)
            i += 1
            continue
        if arg == "--":
            passthrough_args = True
        elif arg in ("-h", "--help"):
            return None, [], False, True, False, None
        elif arg == "--debug":
            debug_enabled = True
        elif arg == "--list-required":
            list_required = True
        elif arg == "--grant":
            i += 1
            if i >= len(args):
                print("Error: --grant requires a YAML file path or inline YAML list")
                return None, [], False, False, False, None
            grant_spec = args[i]
        elif arg.startswith("--grant="):
            grant_spec = arg.split("=", 1)[1]
        elif arg.startswith("--"):
            print(f"Error: unknown option '{arg}'")
            return None, [], False, False, False, None
        elif grid_file is None:
            grid_file = arg
        else:
            program_args.append(arg)
        i += 1
    return grid_file, program_args, debug_enabled, False, list_required, grant_spec


def run_grid_program(args):
    """Run a Grid program with command line arguments"""
    (grid_file, program_args, debug_enabled, show_help,
     list_required, grant_spec) = _parse_cli_args(args)
    if show_help:
        print(USAGE)
        return
    if not grid_file:
        print(USAGE)
        return

    try:
        with open(grid_file, 'r') as file:
            code = file.read()

        compiler = GridLangCompiler()
        if grant_spec:
            compiler.grants = load_grants(grant_spec)

        if list_required:
            compiler.prompt_missing_inputs = False
            compiler.prompt_missing_requires = False
            compiler.halt_before_main_loop = True
            compiler.run(code, program_args)
            # Only predefined resources are grantable — whitelist via RESOURCES
            # Build list from RESOURCES (source of truth) and filter with compiler.requirements
            try:
                from builtin_functions import RESOURCES as _RESOURCES
            except Exception:
                _RESOURCES = {}
            filtered = []
            for res_lower in _RESOURCES:
                for r in compiler.requirements:
                    if (r.get('resource_lower') or str(r.get('resource') or '').lower()) == res_lower:
                        filtered.append(r)
            print(format_required_yaml(filtered), end="")
            return

        # Prompt for missing inputs when no CLI arguments are provided and stdin
        # is interactive. The same applies to missing capability requirements.
        interactive = sys.stdin.isatty()
        compiler.prompt_missing_inputs = (
            len(program_args) == 0 and interactive)
        compiler.prompt_missing_requires = interactive
        compiler.run(code, program_args)
        if debug_enabled:
            csv_path = compiler.export_to_csv(grid_file)
            print(f"Debug CSV written to {csv_path}")

    except FileNotFoundError:
        print(f"Error: Grid file '{grid_file}' not found")
    except GrantError as e:
        print(f"Grant error: {e}")
        sys.exit(2)
    except Exception as e:
        print(f"Error running Grid program: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    try:
        run_grid_program(sys.argv[1:])
    except GrantError:
        sys.exit(2)
    except Exception:
        sys.exit(1)


def main():
    """Console-script entrypoint for the grid CLI."""
    run_grid_program(sys.argv[1:])