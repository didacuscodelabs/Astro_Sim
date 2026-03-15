"""
Tests for physics/constants.py

Validates that all constants match IAU 2012 / CODATA 2018 reference values
to the precision stored in the module.
"""

import pytest
import math
from physics.constants import (
    G, AU, C, SOLAR_MASS, EARTH_MASS, EARTH_RADIUS,
    LIGHT_YEAR, PARSEC, DAY, JULIAN_YEAR,
    PLANETS,
    au_to_meters, meters_to_au,
    ly_to_meters, meters_to_ly,
    pc_to_meters, meters_to_pc,
    light_travel_time,
)


class TestFundamentalConstants:
    """Gravitational constant and speed of light — CODATA 2018."""

    def test_G_value(self):
        """G = 6.67430e-11 N·m²·kg⁻² (CODATA 2018)."""
        assert abs(G - 6.67430e-11) < 1e-16

    def test_C_value(self):
        """c = 2.99792458e8 m/s — exact SI definition."""
        assert C == pytest.approx(2.99792458e8, rel=1e-9)

    def test_AU_value(self):
        """1 AU = 1.495978707e11 m — IAU 2012 exact definition."""
        assert AU == pytest.approx(1.495978707e11, rel=1e-9)

    def test_SOLAR_MASS(self):
        """Solar mass ~ 1.989e30 kg."""
        assert SOLAR_MASS == pytest.approx(1.989e30, rel=1e-3)

    def test_EARTH_MASS(self):
        """Earth mass ~ 5.972e24 kg."""
        assert EARTH_MASS == pytest.approx(5.972e24, rel=1e-3)

    def test_EARTH_RADIUS(self):
        """Earth mean radius ~ 6.371e6 m."""
        assert EARTH_RADIUS == pytest.approx(6.371e6, rel=1e-3)

    def test_LIGHT_YEAR(self):
        """1 ly = 9.4607304725808e15 m."""
        assert LIGHT_YEAR == pytest.approx(9.4607304725808e15, rel=1e-9)

    def test_PARSEC(self):
        """1 pc ≈ 3.0857e16 m."""
        assert PARSEC == pytest.approx(3.085677581491367e16, rel=1e-9)

    def test_DAY_seconds(self):
        """1 solar day = 86400 s exactly."""
        assert DAY == 86400.0

    def test_JULIAN_YEAR(self):
        """1 Julian year = 365.25 days = 31,557,600 s."""
        assert JULIAN_YEAR == pytest.approx(365.25 * 86400, rel=1e-10)


class TestPlanetaryData:
    """All eight solar system planets have valid data."""

    EXPECTED_PLANETS = [
        "mercury", "venus", "earth", "mars",
        "jupiter", "saturn", "uranus", "neptune"
    ]

    def test_all_planets_present(self):
        for name in self.EXPECTED_PLANETS:
            assert name in PLANETS, f"Missing planet: {name}"

    def test_planet_fields(self):
        required = {"mass", "radius", "semi_major_axis_au",
                    "eccentricity", "orbital_period_days", "symbol"}
        for name, data in PLANETS.items():
            assert required <= set(data.keys()), \
                f"Planet '{name}' missing fields: {required - set(data.keys())}"

    def test_earth_params(self):
        e = PLANETS["earth"]
        assert e["semi_major_axis_au"] == pytest.approx(1.0, abs=0.01)
        assert e["mass"] == pytest.approx(5.972e24, rel=1e-2)
        assert e["eccentricity"] == pytest.approx(0.0167, abs=0.001)
        assert e["orbital_period_days"] == pytest.approx(365.25, abs=0.1)

    def test_masses_ordered(self):
        """Sun must be heavier than all planets; Jupiter heavier than Earth."""
        assert SOLAR_MASS > PLANETS["jupiter"]["mass"]
        assert PLANETS["jupiter"]["mass"] > PLANETS["earth"]["mass"]

    def test_semi_major_axes_ordered(self):
        """Planets must be ordered outward from Mercury to Neptune."""
        names = ["mercury", "venus", "earth", "mars",
                 "jupiter", "saturn", "uranus", "neptune"]
        axes = [PLANETS[n]["semi_major_axis_au"] for n in names]
        assert axes == sorted(axes), "Semi-major axes not in increasing order"

    def test_eccentricities_valid(self):
        for name, data in PLANETS.items():
            e = data["eccentricity"]
            assert 0 <= e < 1, f"{name}: eccentricity {e} not in [0, 1)"


class TestUnitConversions:
    """Round-trip and cross-check unit conversion functions."""

    def test_au_roundtrip(self):
        for val in [0.0, 1.0, 5.2, 30.07]:
            assert meters_to_au(au_to_meters(val)) == pytest.approx(val, rel=1e-10)

    def test_ly_roundtrip(self):
        for val in [0.0, 1.0, 4.243, 100.0]:
            assert meters_to_ly(ly_to_meters(val)) == pytest.approx(val, rel=1e-10)

    def test_pc_roundtrip(self):
        for val in [0.0, 1.0, 10.0, 1000.0]:
            assert meters_to_pc(pc_to_meters(val)) == pytest.approx(val, rel=1e-10)

    def test_1_au_in_km(self):
        """1 AU ≈ 149,597,870.7 km."""
        assert au_to_meters(1.0) / 1e3 == pytest.approx(149597870.7, rel=1e-6)

    def test_light_travel_time_sun_earth(self):
        """Light travel time from Sun to Earth ≈ 499 seconds (8.317 minutes)."""
        t = light_travel_time(AU)
        assert t == pytest.approx(499.0, abs=1.0)

    def test_parsec_in_ly(self):
        """1 parsec ≈ 3.2616 light-years."""
        pc_in_m = pc_to_meters(1.0)
        ly_val  = meters_to_ly(pc_in_m)
        assert ly_val == pytest.approx(3.2616, abs=0.001)
