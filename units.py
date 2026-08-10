"""Unit-tracking value wrapper and sticky error values for GridLang.

Unit rules:
- Values carry an internal unit; plain numbers are unitless.
- A variable declared ``of A`` accepts a unitless value or a value of unit A;
  any other unit turns the variable into a sticky ``#UNIT`` error value.
  Variables without a unit constraint accept any unit.
- Unit violations produce a sticky ``#UNIT`` error value (never a raise). An
  If/For condition that evaluates to an error value takes the else branch.
- ``=``/``<>`` are symmetric: a unitless operand is allowed on either side and
  mismatched units count as a difference (never an error).
- ``<``/``<=``/``>``/``>=`` allow a unitless operand on either side; mismatched
  units on both sides produce a ``#UNIT`` error value.
- Arithmetic rules are implemented as operator overloads on UnitValue. Grid
  writes strip the unit wrapper (the grid stores plain numbers).

Error values:
- Errors are sticky: every operation on an error value yields another error
  value, comparisons on it are treated as an error (an If condition then takes
  the else branch), and it strips to its literal ``#...`` string at storage and
  grid boundaries.
- The error codes follow spreadsheet conventions:
    #UNIT   unit mismatch
    #DIV/0  divide (or mod/floor-div) by zero
    #NUM    number not representable (overflow/underflow, e.g. 1e10000)
    #VALUE  same variable bound to two different values
    #TYPE/I value does not match the declared type
    #DIM    value does not match the declared dimensions
    #N/A    value not available (e.g. reading an uninitialized variable)
    #REF    invalid address (index out of range, too many dimensions)
"""

import math

UNIT_ERROR = '#UNIT'
DIV0_ERROR = '#DIV/0'
NUM_ERROR = '#NUM'
VALUE_ERROR = '#VALUE'
TYPE_ERROR = '#TYPE/I'
DIM_ERROR = '#DIM'
NA_ERROR = '#N/A'
REF_ERROR = '#REF'

ERROR_CODES = frozenset({
    UNIT_ERROR, DIV0_ERROR, NUM_ERROR, VALUE_ERROR, TYPE_ERROR,
    DIM_ERROR, NA_ERROR, REF_ERROR,
})


def is_error_value(value):
    """True when ``value`` is a sticky error: a stored error string or a
    wrapped error value."""
    if isinstance(value, str):
        return value in ERROR_CODES
    if isinstance(value, UnitValue):
        return value.error_code is not None
    return False


def error_value(error_code):
    """Factory for a wrapped error value."""
    return UnitValue(None, None, error_code=error_code)


class ConstraintError(ValueError):
    """Raised internally when an assignment violates a value/type/dim/unit
    constraint. ``code`` names the sticky error value that should be stored."""

    def __init__(self, code, message=''):
        super().__init__(message)
        self.code = code


class UnitValue:
    """A value carrying an optional unit, or a sticky error value.

    ``error_code`` marks a sticky error: every operation on an error value
    yields another error value, comparisons on it are treated as an error (an
    If condition then takes the else branch), and it strips to the literal
    ``#...`` string at the grid boundary.
    """

    __slots__ = ('value', 'unit', 'error_code')

    def __init__(self, value, unit=None, error=False, error_code=None):
        self.value = value
        self.unit = unit
        if error:
            self.error_code = error_code or UNIT_ERROR
        else:
            self.error_code = error_code

    @property
    def error(self):
        return self.error_code is not None

    @staticmethod
    def _unit_of(other):
        return getattr(other, 'unit', None)

    @staticmethod
    def _is_error(other):
        return isinstance(other, UnitValue) and other.error_code is not None

    @staticmethod
    def _raw(other):
        return other.value if isinstance(other, UnitValue) else other

    def _same_unit(self, other_unit):
        if self.unit is None or other_unit is None:
            return True
        return str(other_unit).lower() == str(self.unit).lower()

    def _error_value(self, error_code=None):
        return UnitValue(None, None, error_code=error_code or self.error_code or UNIT_ERROR)

    def _check(self, other):
        """Return an error value when either operand is an error."""
        if self.error_code is not None:
            return self._error_value()
        if self._is_error(other):
            return other._error_value()
        return None

    def _result(self, result, unit):
        if isinstance(result, float) and (math.isinf(result) or math.isnan(result)):
            return self._error_value(NUM_ERROR)
        return UnitValue(result, unit) if unit is not None else result

    # ----- additive: same unit or unitless on one side keeps that unit -----

    def __add__(self, other):
        bad = self._check(other)
        if bad is not None:
            return bad
        other_unit = self._unit_of(other)
        if not self._same_unit(other_unit):
            return self._error_value()
        unit = self.unit if self.unit is not None else other_unit
        return self._result(self._raw(self) + self._raw(other), unit)

    __radd__ = __add__

    def __sub__(self, other):
        bad = self._check(other)
        if bad is not None:
            return bad
        other_unit = self._unit_of(other)
        if not self._same_unit(other_unit):
            return self._error_value()
        unit = self.unit if self.unit is not None else other_unit
        return self._result(self._raw(self) - self._raw(other), unit)

    __rsub__ = __sub__

    # ----- multiplicative: at least one operand must be unitless -----

    def __mul__(self, other):
        bad = self._check(other)
        if bad is not None:
            return bad
        other_unit = self._unit_of(other)
        if self.unit is not None and other_unit is not None:
            return self._error_value()
        unit = self.unit if self.unit is not None else other_unit
        return self._result(self._raw(self) * self._raw(other), unit)

    __rmul__ = __mul__

    # ----- division/mod: right operand unitless or same unit -----

    def _divide(self, other, op):
        bad = self._check(other)
        if bad is not None:
            return bad
        other_unit = self._unit_of(other)
        if other_unit is not None and not self._same_unit(other_unit):
            return self._error_value()
        # Same unit on both sides produces a unitless result; a unitless
        # right operand keeps the left operand's unit.
        unit = None if other_unit is not None else self.unit
        try:
            result = op(self._raw(self), self._raw(other))
        except ZeroDivisionError:
            return self._error_value(DIV0_ERROR)
        return self._result(result, unit)

    def __truediv__(self, other):
        return self._divide(other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        # Left operand (other) is unitless while self carries a unit -> error.
        bad = self._check(other)
        if bad is not None:
            return bad
        return self._error_value()

    def __floordiv__(self, other):
        return self._divide(other, lambda a, b: a // b)

    __rfloordiv__ = __rtruediv__

    def __mod__(self, other):
        return self._divide(other, lambda a, b: a % b)

    __rmod__ = __rtruediv__

    # ----- power: exponent must be unitless -----

    def __pow__(self, other):
        bad = self._check(other)
        if bad is not None:
            return bad
        if self._unit_of(other) is not None:
            return self._error_value()
        try:
            result = self._raw(self) ** self._raw(other)
        except ZeroDivisionError:
            return self._error_value(DIV0_ERROR)
        return self._result(result, self.unit)

    def __rpow__(self, other):
        # Base (other) is unitless while the exponent self carries a unit.
        bad = self._check(other)
        if bad is not None:
            return bad
        return self._error_value()

    # ----- unary -----

    def __neg__(self):
        if self.error_code is not None:
            return self._error_value()
        return UnitValue(-self.value, self.unit)

    def __pos__(self):
        if self.error_code is not None:
            return self._error_value()
        return UnitValue(self.value, self.unit)

    # ----- equality: symmetric; mismatch is a difference, never an error -----

    def __eq__(self, other):
        if self.error_code is not None:
            return self._error_value()
        if self._is_error(other):
            return other._error_value()
        other_unit = self._unit_of(other)
        if self.unit is not None and other_unit is not None and not self._same_unit(other_unit):
            return False
        return self._raw(self) == self._raw(other)

    def __ne__(self, other):
        if self.error_code is not None:
            return self._error_value()
        if self._is_error(other):
            return other._error_value()
        other_unit = self._unit_of(other)
        if self.unit is not None and other_unit is not None and not self._same_unit(other_unit):
            return True
        return self._raw(self) != self._raw(other)

    def __hash__(self):
        return hash((self.value, self.unit, self.error_code))

    # ----- ordering: unitless on either side is fine; mismatch is #UNIT -----

    def _ordered(self, other, op):
        if self.error_code is not None:
            return self._error_value()
        if self._is_error(other):
            return other._error_value()
        other_unit = self._unit_of(other)
        if self.unit is not None and other_unit is not None and not self._same_unit(other_unit):
            return self._error_value()
        try:
            return op(self._raw(self), self._raw(other))
        except (TypeError, ValueError):
            return self._error_value()

    def __lt__(self, other):
        return self._ordered(other, lambda a, b: a < b)

    def __le__(self, other):
        return self._ordered(other, lambda a, b: a <= b)

    def __gt__(self, other):
        return self._ordered(other, lambda a, b: a > b)

    def __ge__(self, other):
        return self._ordered(other, lambda a, b: a >= b)

    # ----- conversion -----

    def __bool__(self):
        return False if self.error_code is not None else bool(self.value)

    def __float__(self):
        return float('nan') if self.error_code is not None else float(self.value)

    def __int__(self):
        return 0 if self.error_code is not None else int(self.value)

    def __index__(self):
        return 0 if self.error_code is not None else int(self.value)

    def __str__(self):
        return self.error_code if self.error_code is not None else str(self.value)

    def __repr__(self):
        return (f'UnitValue({self.value!r}, {self.unit!r}, '
                f'error_code={self.error_code!r})')


def strip_units(value):
    """Recursively remove unit wrappers for storage/grid boundaries.

    An error value strips to its literal ``#...`` code.
    """
    if isinstance(value, UnitValue):
        return value.error_code if value.error_code is not None else value.value
    if isinstance(value, list):
        return [strip_units(v) for v in value]
    if isinstance(value, tuple):
        return tuple(strip_units(v) for v in value)
    if isinstance(value, dict):
        return {k: strip_units(v) for k, v in value.items()}
    return value
