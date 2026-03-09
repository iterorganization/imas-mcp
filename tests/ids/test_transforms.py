"""Tests for IDS transform execution engine."""

from __future__ import annotations

import pytest

from imas_codex.ids.transforms import convert_units, execute_transform, set_nested


class TestExecuteTransform:
    def test_identity_none(self):
        assert execute_transform(42, None) == 42

    def test_identity_value(self):
        assert execute_transform(42, "value") == 42

    def test_scale(self):
        assert execute_transform(5.0, "value * 1000.0") == 5000.0

    def test_negate(self):
        assert execute_transform(3.0, "-value") == -3.0

    def test_float_cast(self):
        assert execute_transform("42", "float(value)") == 42.0

    def test_str_cast(self):
        assert execute_transform(42, "str(value)") == "42"

    def test_math_sqrt(self):
        result = execute_transform(16.0, "math.sqrt(value)")
        assert result == pytest.approx(4.0)

    def test_abs_builtin(self):
        assert execute_transform(-5, "abs(value)") == 5

    def test_complex_expression(self):
        result = execute_transform(2.0, "value ** 2 + 1")
        assert result == pytest.approx(5.0)

    def test_no_builtins_access(self):
        with pytest.raises(NameError):
            execute_transform(1, "__import__('os')")


class TestSetNested:
    def test_simple_attr(self):
        class Obj:
            pass

        obj = Obj()
        set_nested(obj, "name", "test")
        assert obj.name == "test"

    def test_dotted_path(self):
        class Inner:
            pass

        class Mid:
            def __init__(self):
                self.inner = Inner()

        class Obj:
            def __init__(self):
                self.mid = Mid()

        obj = Obj()
        set_nested(obj, "mid.inner.value", 42)
        assert obj.mid.inner.value == 42


class TestConvertUnits:
    def test_meters_to_cm(self):
        result = convert_units(1.0, "m", "cm")
        assert result == pytest.approx(100.0)

    def test_same_units(self):
        result = convert_units(5.0, "m", "m")
        assert result == pytest.approx(5.0)
