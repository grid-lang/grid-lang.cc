class Node:
    pass

class Assignment(Node):
    def __init__(self, target, value, array_name=None):
        self.target = target  # Either CellReference or Variable
        self.value = value    # Expression to be assigned
        self.array_name = array_name  # Name for array assignments

class CellReference(Node):
    def __init__(self, address):
        self.address = address  # Cell address (e.g., "A1")

class DynamicCellReference(Node):
    def __init__(self, prefix, dynamic_part):
        self.prefix = prefix      # Static part of the address (e.g., "A")
        self.dynamic_part = dynamic_part  # Variable or expression for the dynamic part

class Variable(Node):
    def __init__(self, name):
        self.name = name  # Variable name (without the ':' prefix)

class Number(Node):
    def __init__(self, value):
        self.value = value  # Numeric value

class BinaryOp(Node):
    def __init__(self, left, op, right):
        self.left = left   # Left operand
        self.op = op       # Operator (+, -, *, /)
        self.right = right # Right operand

class Function(Node):
    def __init__(self, name, range):
        self.name = name   # Function name (e.g., "SUM")
        self.range = range # CellRange object

class CellRange(Node):
    def __init__(self, start, end):
        self.start = start  # Starting cell address
        self.end = end      # Ending cell address

class ArrayValues(Node):
    def __init__(self, values):
        self.values = values  # List of numeric values 