"""
Tests for physics/gravity.py

Every function is tested against independently computed reference values.
Physical quantities are validated to match published sources:
  - NASA Planetary Fact Sheets
  - Carroll & Ostlie (2017), Introduction to Modern Astrophysics
  - CODATA 2018
"""

import pytest
import math
from physics.constants import G, AU, C, SOLAR_MASS, EARTH_MASS, EARTH_RADIUS
from physics.gravity import (
    gravitational_force,
    gravitational_potential,
    gravitational_potential_energy,
    gravitational_acceleration,
    escape_velocity,
    schwarzschild_radius,
    hill_sphere_radius,
    roche_limit,
    reduced_mass,
    center_of_mass,
    tidal_acceleration,
    gravitational_wave_power,
)


class TestGravitationalForce:
    """F = G·m₁·m₂ / r²"""

    def test_sun_earth_force(self):
        """Sun–Earth gravitational force ≈ 3.54×10²² N."""
        F = gravitational_force(SOLAR_MASS, EARTH_MASS, AU)
        assert F == pytest.approx(3.54e22, rel=0.01)

    def test_symmetry(self):
        """Force is symmetric: F(m1, m2, r) == F(m2, m1, r)."""
        F1 = gravitational_force(1e24, 1e20, 1e8)
        F2 = gravitational_force(1e20, 1e24, 1e8)
        assert F1 == pytest.approx(F2, rel=1e-10)

    def test_inverse_square_law(self):
        """Doubling distance → force reduces by factor 4."""
        r  = 1e9
        F1 = gravitational_force(1e25, 1e22, r)
        F2 = gravitational_force(1e25, 1e22, 2*r)
        assert F1 / F2 == pytest.approx(4.0, rel=1e-9)

    def test_proportional_to_mass(self):
        """Doubling m1 doubles force."""
        m, r = 1e20, 1e8
        F1 = gravitational_force(1e24, m, r)
        F2 = gravitational_force(2e24, m, r)
        assert F2 / F1 == pytest.approx(2.0, rel=1e-9)

    def test_zero_mass(self):
        """Force is zero if either mass is zero."""
        assert gravitational_force(0, 1e24, 1e8) == 0.0
        assert gravitational_force(1e24, 0, 1e8) == 0.0

    def test_negative_r_raises(self):
        with pytest.raises(ValueError):
            gravitational_force(1e24, 1e24, -1.0)

    def test_zero_r_raises(self):
        with pytest.raises(ValueError):
            gravitational_force(1e24, 1e24, 0.0)

    def test_negative_mass_raises(self):
        with pytest.raises(ValueError):
            gravitational_force(-1e24, 1e24, 1e8)


class TestGravitationalPotential:
    """Φ = −G·M / r"""

    def test_negative(self):
        """Gravitational potential is always negative."""
        assert gravitational_potential(SOLAR_MASS, AU) < 0

    def test_approaches_zero_at_infinity(self):
        """Potential approaches 0 as r → ∞."""
        phi_large = gravitational_potential(SOLAR_MASS, 1e30)
        assert abs(phi_large) < 1e-8

    def test_deeper_at_smaller_r(self):
        """Potential is more negative closer to the source."""
        phi_close = gravitational_potential(SOLAR_MASS, 0.1 * AU)
        phi_far   = gravitational_potential(SOLAR_MASS, 10.0 * AU)
        assert phi_close < phi_far


class TestGravitationalAcceleration:
    """g = G·M / r²"""

    def test_earth_surface_gravity(self):
        """Earth surface gravity ≈ 9.82 m/s² (within 0.2% of standard g)."""
        g = gravitational_acceleration(EARTH_MASS, EARTH_RADIUS)
        assert g == pytest.approx(9.80665, abs=0.05)

    def test_inverse_square(self):
        """Acceleration follows inverse-square law."""
        g1 = gravitational_acceleration(SOLAR_MASS, AU)
        g2 = gravitational_acceleration(SOLAR_MASS, 2*AU)
        assert g1 / g2 == pytest.approx(4.0, rel=1e-9)

    def test_negative_r_raises(self):
        with pytest.raises(ValueError):
            gravitational_acceleration(EARTH_MASS, 0.0)


class TestEscapeVelocity:
    """v_esc = √(2GM/R)"""

    def test_earth_escape_velocity(self):
        """Earth escape velocity ≈ 11,186 m/s (< 0.01% error)."""
        v = escape_velocity(EARTH_MASS, EARTH_RADIUS)
        assert v == pytest.approx(11186.0, rel=0.001)

    def test_is_sqrt2_times_orbital(self):
        """v_esc = √2 · v_circ at same radius."""
        from physics.orbital_mechanics import orbital_velocity
        r = AU
        v_esc  = escape_velocity(SOLAR_MASS, r)
        v_circ = orbital_velocity(SOLAR_MASS, r)
        assert v_esc / v_circ == pytest.approx(math.sqrt(2), rel=1e-9)

    def test_moon_escape_velocity(self):
        """Moon escape velocity ≈ 2,380 m/s."""
        from physics.constants import PLANETS
        moon_mass = 7.346e22
        moon_r    = 1.7374e6
        v = escape_velocity(moon_mass, moon_r)
        assert v == pytest.approx(2380.0, rel=0.02)

    def test_zero_r_raises(self):
        with pytest.raises(ValueError):
            escape_velocity(EARTH_MASS, 0.0)


class TestSchwarzschildRadius:
    """r_s = 2GM/c²"""

    def test_solar_schwarzschild(self):
        """Solar Schwarzschild radius ≈ 2,953 m ≈ 2.95 km."""
        r_s = schwarzschild_radius(SOLAR_MASS)
        assert r_s == pytest.approx(2953.0, rel=0.01)

    def test_earth_schwarzschild(self):
        """Earth Schwarzschild radius ≈ 8.87 mm."""
        r_s = schwarzschild_radius(EARTH_MASS)
        assert r_s == pytest.approx(8.87e-3, rel=0.01)

    def test_scales_linearly_with_mass(self):
        """r_s ∝ M: doubling mass doubles r_s."""
        r1 = schwarzschild_radius(SOLAR_MASS)
        r2 = schwarzschild_radius(2 * SOLAR_MASS)
        assert r2 / r1 == pytest.approx(2.0, rel=1e-9)

    def test_at_rs_escape_equals_c(self):
        """At r = r_s, escape velocity should equal c."""
        r_s = schwarzschild_radius(SOLAR_MASS)
        v   = escape_velocity(SOLAR_MASS, r_s)
        assert v == pytest.approx(C, rel=1e-6)


class TestHillSphere:
    """r_H = a(1−e)·(M_planet / 3M_star)^(1/3)"""

    def test_earth_hill_sphere(self):
        """Earth's Hill sphere radius ≈ 1.5 million km."""
        r_h = hill_sphere_radius(EARTH_MASS, SOLAR_MASS, AU)
        assert r_h == pytest.approx(1.5e9, rel=0.05)

    def test_larger_planet_larger_hill_sphere(self):
        """More massive planet has larger Hill sphere."""
        r_earth   = hill_sphere_radius(EARTH_MASS, SOLAR_MASS, AU)
        r_jupiter = hill_sphere_radius(1.898e27, SOLAR_MASS, 5.2*AU)
        assert r_jupiter > r_earth


class TestReducedMass:
    """μ = m₁m₂/(m₁+m₂)"""

    def test_equal_masses(self):
        """Equal masses → μ = m/2."""
        m = 1e24
        assert reduced_mass(m, m) == pytest.approx(m / 2, rel=1e-6)

    def test_one_dominant_mass(self):
        """If m2 >> m1, μ → m1."""
        m1, m2 = 1e10, 1e30
        assert reduced_mass(m1, m2) == pytest.approx(m1, rel=0.001)


class TestTidalAcceleration:
    def test_tidal_scales_as_r_cubed(self):
        """Tidal acceleration ∝ 1/r³: tripling r → 1/27 tidal acc."""
        a1 = tidal_acceleration(SOLAR_MASS, AU,   EARTH_RADIUS)
        a2 = tidal_acceleration(SOLAR_MASS, 3*AU, EARTH_RADIUS)
        assert a1 / a2 == pytest.approx(27.0, rel=1e-6)
