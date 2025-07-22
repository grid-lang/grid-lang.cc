# GridLang

GridLang is a Python-based tool for compiling and processing `.grid` files. This repository contains the source code for the `grid` command-line tool, which uses the `GridLangCompiler` to process input files and optionally export results to CSV format.

## Prerequisites

- **Python**: Version 3.7 or higher (tested with Python 3.11).
- **pip**: Python package manager (included with Python).
- **Git**: For cloning the repository (optional if downloading as a ZIP).
- **pyarrow**: Required dependency for data handling.
- **PyInstaller**: Optional, for creating a standalone `grid.exe` executable.

## Installation

Follow these steps to clone or download the `gridlang_public` repository, install the package, and run the `grid` command.

### 1. Clone or Download the Repository

#### Option 1: Clone with Git

1. Open a terminal or command prompt.
2. Clone the repository using:
   https://github.com/grid-lang/grid-lang.cc.git

3. Navigate to the project directory:
   cd gridlang_public

#### Option 2: Download as ZIP

1. Go to the repository’s webpage (e.g., on GitHub).
2. Click the green **Code** button and select **Download ZIP**.
3. Extract the ZIP file to a folder (e.g., `gridlang_public`).
4. Navigate to the project directory in a terminal:
   cd path/to/gridlang_public

### 2. Set Up a Virtual Environment (Recommended)

To avoid conflicts with other Python packages, use a virtual environment:

python -m venv venv
venv\Scripts\activate # On Windows
source venv/bin/activate # On macOS/Linux

### 3. Install the Package

#### Option 1: Install with `setup.py`

Install the package and its dependencies using `setuptools`:

pip install .

This installs the `gridlang` package, its dependency (`pyarrow`), and creates the `grid` command in your Python environment’s `Scripts` directory (e.g., `C:\Users\johndoe\AppData\Local\Programs\Python\Python311\Scripts` on Windows).

#### Option 2: Create a Standalone Executable with PyInstaller

1. Install PyInstaller:
   pip install pyinstaller
2. Build the `grid.exe` executable:
   pyinstaller --add-data "main.py;." --add-data "compiler.py;." --add-data "array_handler.py;." --add-data "expression.py;." --name grid main.py
3. The executable will be created in `dist\grid\grid.exe`.

### 4. Verify Installation

#### For `setup.py` Installation

Check that the `grid` command is available:

grid --help or grid

This should display help content:

If you see a `command not found` error, add the Python `Scripts` directory to your PATH:

- On Windows:
  set PATH=%PATH%;C:\Users\johndoe\AppData\Local\Programs\Python\Python311\Scripts
- On macOS/Linux:
  export PATH=$PATH:$HOME/.local/bin

#### For PyInstaller

Test the executable:

dist\grid\grid.exe example.grid

## Usage

The `grid` command processes `.grid` files using the `GridLangCompiler`. It supports two modes:

- **Compile a `.grid` file**: Processes the input file and prints the results.
- **Debug mode**: Exports the results to a CSV file (requires implementation in `compiler.py`).

### Example Commands

1. **Compile a `.grid` file**:

   grid example.grid

   or, for PyInstaller:

   dist\grid\grid.exe example.grid

2. **Compile with debug mode**:

   grid example.grid --debug

3. **Run with `-r` flag**:
   grid -r example.grid

### Creating a Sample `.grid` File

Create a file named `example.grid` in the `gridlang_public` directory:

A1 = 42
B2 = "Hello"

Then run:

grid example.grid

### Expected Output

Imported gridlang.compiler successfully
Compiling example.grid...

A1 = 42
B2 = Hello

## Project Structure

gridlang_public/
├── setup.py # Package configuration
├── main.py # Entry point for the grid command
├── compiler.py # Contains the GridLangCompiler class
├── array_handler.py # Helper module for array operations
├── expression.py # Helper module for expression parsing
├── README.md # This file

## Troubleshooting

- **Error: `ModuleNotFoundError: No module named 'main'`**:

  - Ensure `main.py`, `compiler.py`, `array_handler.py`, and `expression.py` are in the `gridlang_public` directory.
  - For `setup.py`, reinstall:

    pip uninstall gridlang
    pip install .

  - For PyInstaller, rebuild:

    pyinstaller --add-data "main.py;." --add-data "compiler.py;." --add-data "array_handler.py;." --add-data "expression.py;." --name grid main.py

  - Check for erroneous `import main` in `compiler.py`, `array_handler.py`, or `expression.py`:

    findstr /s "import main" _.py # On Windows
    grep -r "import main" _.py # On macOS/Linux

- **Error: `Unable to create process ... The system cannot find the file specified`**:

  - Verify Python is installed at `C:\Users\johndoe\AppData\Local\Programs\Python\Python311` or `C:\Python313`.
  - Install PyInstaller:

    pip install pyinstaller

  - Add Python and Scripts to PATH:

    set PATH=%PATH%;C:\Users\johndoe\AppData\Local\Programs\Python\Python311;C:\Users\johndoe\AppData\Local\Programs\Python\Python311\Scripts

- **Error: `ModuleNotFoundError: No module named 'pyarrow'`**:

  - Install `pyarrow`:

    pip install pyarrow

## Development

1. Edit `main.py`, `compiler.py`, `array_handler.py`, or `expression.py`.
2. Reinstall (`pip install .`) or rebuild with PyInstaller.
3. Test with:

   python main.py example.grid

## License

MIT License

## Contact

Open an issue or pull request on the `gridlang_public` repository.
