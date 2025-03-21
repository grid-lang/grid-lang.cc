# Grid Language

A grid-based programming language interpreter that allows you to work with spatial and non-spatial data in a grid format.

## Installation

```bash
pip install -e .
```

## Usage

You can run .grid files directly from the command line:

```bash
grid example.grid
```

Or import and use the interpreter in your Python code:

```python
from grid_lang import run_grid

with open('example.grid', 'r') as file:
    code = file.read()
run_grid(code)
```

## Example

Here's a simple example of a .grid file:

```
# Spatial data
A1 = 10
B1 = 20
C1 = A1 + B1

# Array assignment
A2:B2 = [1, 2, 3]

# Non-spatial data
x = 5
y = 10
z = x * y

# Dynamic cell reference
D1 = A1 + x
```

## Development Roadmap

### 1. Expressions, spatial and non-spatial data

- [x] Basic arithmetic operations
- [x] Cell references
- [x] Variable declarations
- [x] Array assignments

### 2. Data types

- [ ] Integer
- [ ] Float
- [ ] String
- [ ] Boolean
- [ ] Date/Time

### 3. Data shape

- [ ] Matrix operations
- [ ] Reshape operations
- [ ] Transpose
- [ ] Concatenation

### 4. Private data and constraints

- [ ] Access modifiers
- [ ] Data validation
- [ ] Range constraints
- [ ] Type constraints

### 5. If condition and for loop

- [ ] Conditional statements
- [ ] Iteration over ranges
- [ ] While loops
- [ ] Break/Continue

### 6. Value updates

- [ ] Incremental updates
- [ ] Batch updates
- [ ] Transaction support
- [ ] Rollback capability

### 7. Subprocess

- [ ] Function definitions
- [ ] Module imports
- [ ] Process communication
- [ ] Error handling

## License

MIT License
