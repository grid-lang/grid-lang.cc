from setuptools import setup, find_packages

setup(
    name="grid-lang",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
    entry_points={
        "console_scripts": [
            "grid=grid_lang.cli:main",
        ],
    },
    python_requires=">=3.6",
) 