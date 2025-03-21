# Abstract Syntax Tree (AST) Node classes
# These classes represent the structure of the program after parsing
class Node:
    pass

# Represents an assignment statement (e.g., [A1] := 5 or :X = 10)
class Assignment(Node):
    def __init__(self, target, value, array_name=None):
        self.target = target  # Either CellReference or Variable
        self.value = value    # Expression to be assigned
        self.array_name = array_name  # Name for array assignments

# Represents a cell reference (e.g., [A1])
class CellReference(Node):
    def __init__(self, address):
        self.address = address  # Cell address (e.g., "A1")

# Represents a dynamic cell reference (e.g., A&{N})
class DynamicCellReference(Node):
    def __init__(self, prefix, dynamic_part):
        self.prefix = prefix      # Static part of the address (e.g., "A")
        self.dynamic_part = dynamic_part  # Variable or expression for the dynamic part

# Represents a variable reference (e.g., :X)
# Used for non-spatial data that is not placed on the grid
class Variable(Node):
    def __init__(self, name):
        self.name = name  # Variable name (without the ':' prefix)

# Represents a numeric literal (e.g., 5, 10, 15)
class Number(Node):
    def __init__(self, value):
        self.value = value  # Numeric value

# Represents a binary operation (e.g., A1 + B1)
class BinaryOp(Node):
    def __init__(self, left, op, right):
        self.left = left   # Left operand
        self.op = op       # Operator (+, -, *, /)
        self.right = right # Right operand

# Represents a function call (e.g., SUM[A1:A3])
class Function(Node):
    def __init__(self, name, range):
        self.name = name   # Function name (e.g., "SUM")
        self.range = range # CellRange object

# Represents a range of cells (e.g., A1:A3)
class CellRange(Node):
    def __init__(self, start, end):
        self.start = start  # Starting cell address
        self.end = end      # Ending cell address

# Represents an array of values
class ArrayValues(Node):
    def __init__(self, values):
        self.values = values  # List of numeric values 