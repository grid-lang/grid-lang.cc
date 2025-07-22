from setuptools import setup, find_packages

setup(
    name='gridlang',
    version='0.1',
    py_modules=['main', 'compiler', 'expression', 'array_handler', 'utils'],
    packages=find_packages(),
    install_requires=['pyarrow'],  # Add pyarrow dependency
    entry_points={
        'console_scripts': [
            'grid=main:run_tests'  # Correct entry point
        ]
    },
)
