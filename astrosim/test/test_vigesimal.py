"""
Tests for physics/vigesimal.py

Validates the Maya vigesimal (base-20) numeration system implementation.
Reference values are computed by hand and cross-checked against known
Maya astronomical constants.
"""

import pytest
from physics.vigesimal import (
    decimal_to_vigesimal,
    vigesimal_to_decimal,
    vigesimal_notation,
    vigesimal_glyph,
    full_vigesimal_display,
    astronomical_distance_vigesimal,
    solar_system_vigesimal_table,
    maya_long_count,
    orbital_period_maya,
)
from physics.constants import AU


class TestDecimalToVigesimal:

    def test_zero(self):
        assert decimal_to_vigesimal(0) == [0]

    def test_single_digit(self):
        """Numbers 1–19 are single-digit vigesimal."""
        for n in range(1, 20):
            assert decimal_to_vigesimal(n) == [n]

    def test_twenty(self):
        """20 in base-20 is [1, 0]."""
        assert decimal_to_vigesimal(20) == [1, 0]

    def test_four_hundred(self):
        """400 = 20² → [1, 0, 0]."""
        assert decimal_to_vigesimal(400) == [1, 0, 0]

    def test_solar_year(self):
        """365 days = 18×20 + 5 → [18, 5]."""
        assert decimal_to_vigesimal(365) == [18, 5]

    def test_venus_synodic_period(self):
        """584 days (Venus synodic period) = 1×400 + 9×20 + 4 → [1, 9, 4]."""
        assert decimal_to_vigesimal(584) == [1, 9, 4]

    def test_maya_round(self):
        """18,980 = 1×8000 + 18×400 + 19×20 + 0 — the Maya Calendar Round."""
        digits = decimal_to_vigesimal(18980)
        assert vigesimal_to_decimal(digits) == 18980

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            decimal_to_vigesimal(-1)

    def test_non_integer_raises(self):
        with pytest.raises((ValueError, TypeError)):
            decimal_to_vigesimal(3.14)


class TestVigesimalToDecimal:

    def test_zero(self):
        assert vigesimal_to_decimal([0]) == 0

    def test_single_digit(self):
        for n in range(20):
            assert vigesimal_to_decimal([n]) == n

    def test_two_digits(self):
        """[1, 5] = 1×20 + 5 = 25."""
        assert vigesimal_to_decimal([1, 5]) == 25

    def test_three_digits(self):
        """[2, 0, 0] = 2×400 = 800."""
        assert vigesimal_to_decimal([2, 0, 0]) == 800

    def test_invalid_digit_raises(self):
        with pytest.raises(ValueError):
            vigesimal_to_decimal([20])   # 20 is not a valid vigesimal digit

    def test_invalid_digit_negative_raises(self):
        with pytest.raises(ValueError):
            vigesimal_to_decimal([-1])


class TestRoundTrip:
    """decimal → vigesimal → decimal must be identity."""

    @pytest.mark.parametrize("n", [
        0, 1, 19, 20, 99, 365, 400, 584, 1000, 8000,
        18980, 144000, 1000000
    ])
    def test_roundtrip(self, n):
        assert vigesimal_to_decimal(decimal_to_vigesimal(n)) == n


class TestVigesimalNotation:

    def test_zero_notation(self):
        s = vigesimal_notation(0)
        assert "0" in s

    def test_contains_breakdown(self):
        """Notation should contain place-value breakdown."""
        s = vigesimal_notation(365)
        assert "20" in s
        assert "18" in s or "5" in s

    def test_maya_names_mode(self):
        """Maya name mode returns Yucatec name strings."""
        s = vigesimal_notation(365, use_names=True)
        assert "uaxaclahun" in s   # digit 18
        assert "ho" in s            # digit 5

    def test_single_digit(self):
        s = vigesimal_notation(7, use_names=True)
        assert "uuc" in s


class TestVigesimalGlyph:

    def test_zero_glyph(self):
        """Zero is represented by the shell symbol."""
        assert vigesimal_glyph(0) == "∅"

    def test_five_glyph(self):
        """5 is one bar."""
        assert "▬" in vigesimal_glyph(5)

    def test_one_glyph(self):
        assert "●" in vigesimal_glyph(1)

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            vigesimal_glyph(20)


class TestAstronomicalDistanceVigesimal:

    def test_returns_expected_keys(self):
        result = astronomical_distance_vigesimal(AU, unit="km")
        required = {"value_decimal", "unit", "value_int", "digits",
                    "notation", "glyph_digits", "maya_names",
                    "place_names", "full_display"}
        assert required <= set(result.keys())

    def test_au_in_km(self):
        """1 AU ≈ 149,597,870 km — check round-trip."""
        result = astronomical_distance_vigesimal(AU, unit="km")
        assert result["value_decimal"] == pytest.approx(149597870.7, rel=1e-4)
        # Verify vigesimal → decimal round-trip
        n_check = vigesimal_to_decimal(result["digits"])
        assert n_check == result["value_int"]

    def test_invalid_unit_raises(self):
        with pytest.raises(ValueError):
            astronomical_distance_vigesimal(AU, unit="furlongs")

    def test_digit_values_valid(self):
        """All digits must be in [0, 19]."""
        result = astronomical_distance_vigesimal(AU * 10, unit="km")
        for d in result["digits"]:
            assert 0 <= d <= 19


class TestSolarSystemTable:

    def test_returns_8_planets(self):
        table = solar_system_vigesimal_table()
        assert len(table) == 8

    def test_all_fields_present(self):
        table = solar_system_vigesimal_table()
        required = {"name", "distance_km", "semi_major_au",
                    "vigesimal_digits", "notation", "maya_names", "glyph_repr"}
        for row in table:
            assert required <= set(row.keys())

    def test_earth_au_approximately_one(self):
        table = solar_system_vigesimal_table()
        earth = next(r for r in table if r["name"] == "Earth")
        assert earth["semi_major_au"] == pytest.approx(1.0, abs=0.01)

    def test_distances_increasing(self):
        """Distances should increase from Mercury to Neptune."""
        table = solar_system_vigesimal_table()
        dists = [r["distance_km"] for r in table]
        assert dists == sorted(dists)


class TestMayaLongCount:
    """Test Maya Long Count calendar conversion."""

    def test_zero_days(self):
        lc = maya_long_count(0)
        assert lc["notation"] == "0.0.0.0.0"
        assert lc["total_days"] == 0

    def test_one_kin(self):
        lc = maya_long_count(1)
        assert lc["kin"] == 1
        assert lc["winal"] == 0

    def test_one_winal(self):
        """1 winal = 20 days."""
        lc = maya_long_count(20)
        assert lc["winal"] == 1
        assert lc["kin"] == 0

    def test_one_tun(self):
        """1 tun = 360 days."""
        lc = maya_long_count(360)
        assert lc["tun"] == 1
        assert lc["winal"] == 0

    def test_one_baktun(self):
        """1 baktun = 144,000 days."""
        lc = maya_long_count(144000)
        assert lc["baktun"] == 1
        assert lc["katun"] == 0

    def test_maya_creation_date(self):
        """Maya creation date: 13.0.0.0.0 = 13 baktuns = 1,872,000 days."""
        lc = maya_long_count(1_872_000)
        assert lc["baktun"] == 13
        assert lc["solar_years"] == pytest.approx(5125.36, rel=0.001)

    def test_solar_year_in_long_count(self):
        """365 days ≈ 1.0.5 in Long Count."""
        lc = maya_long_count(365)
        assert lc["tun"] == 1
        assert lc["winal"] == 0
        assert lc["kin"] == 5

    def test_venus_synodic_period(self):
        """Venus synodic period: 584 days = 1.12.4 in Long Count."""
        lc = maya_long_count(584)
        assert lc["total_days"] == 584
        assert lc["notation"] == "0.0.1.12.4"

    def test_orbital_period_maya_earth(self):
        """Earth orbital period in Maya Long Count."""
        s = orbital_period_maya(365.25)
        assert "1.0.5" in s or "365" in s
