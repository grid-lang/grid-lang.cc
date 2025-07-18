# <xaiArtifact artifact_id="9592cb0a-5bd6-4283-b60e-23d5eb595d32" artifact_version_id="3ea58d55-8ecd-4316-8c99-7f17ecd6aaba" title="main.py" contentType="text/python">
import sys
from compiler import GridLangCompiler


def run_tests(args):
    compiler = GridLangCompiler()
    if len(args) == 2 and args[0] == "-r":
        with open(args[1]) as file:
            compiler.run(file.read())
    else:
        compiler.run_tests_independent(args)


if __name__ == '__main__':
    try:
        run_tests(sys.argv[1:])
    except Exception as e:
        sys.exit(1)
