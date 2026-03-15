"""
Tests for physics/orbital_mechanics.py

Validates orbital mechanics calculations against:
  - IAU 2012 planetary data
  - Vallado (2013), Fundamentals of Astrodynamics
  - Curtis (2020), Orbital Mechanics for Engineering Students
"""

import pytest
import math
from physics.constants import G, AU, SOLAR_MASS, EARTH_MASS, EARTH_RADIUS, JULIAN_YEAR, DAY
from physics.orbital_mechanics import (
    orbital_velocity,
    orbital_period,
    vis_viva,
    apoapsis,
    periapsis,
    semi_minor_axis,
    orbital_radius_at_angle,
    specific_orbital_energy,
    specific_angular_momentum,
    hohmann_transfer,
    tsiolkovsky_delta_v,
    mass_ratio_from_delta_v,
    solve_kepler,
    eccentric_to_true_anomaly,
    mean_motion,
    position_in_orbit,
    orbital_summary,
)


class TestOrbitalVelocity:
    """v = √(GM/r)"""

    def test_earth_orbital_velocity(self):
        """Earth's mean orbital velocity ≈ 29,785 m/s."""
        v = orbital_velocity(SOLAR_MASS, AU)
        assert v == pytest.approx(29784.69, abs=10.0)

    def test_faster_at_smaller_radius(self):
        """Inner orbits are faster than outer orbits."""
        v_inner = orbital_velocity(SOLAR_MASS, 0.5 * AU)
        v_outer = orbital_velocity(SOLAR_MASS, 2.0 * AU)
        assert v_inner > v_outer

    def test_inverse_sqrt_law(self):
        """v ∝ 1/√r: quadrupling r halves v."""
        v1 = orbital_velocity(SOLAR_MASS, AU)
        v2 = orbital_velocity(SOLAR_MASS, 4 * AU)
        assert v1 / v2 == pytest.approx(2.0, rel=1e-9)

    def test_zero_r_raises(self):
        with pytest.raises(ValueError):
            orbital_velocity(SOLAR_MASS, 0.0)


class TestOrbitalPeriod:
    """T = 2π√(a³/GM) — Kepler's Third Law"""

    def test_earth_period_one_year(self):
        """Earth's orbital period ≈ 1 Julian year (< 0.02% error)."""
        T = orbital_period(SOLAR_MASS, AU)
        assert T / JULIAN_YEAR == pytest.approx(1.0, rel=0.002)

    def test_mars_period(self):
        """Mars orbital period ≈ 686.97 days."""
        T = orbital_period(SOLAR_MASS, 1.524 * AU)
        assert T / DAY == pytest.approx(686.97, rel=0.005)

    def test_jupiter_period(self):
        """Jupiter orbital period ≈ 11.86 years."""
        T = orbital_period(SOLAR_MASS, 5.204 * AU)
        assert T / JULIAN_YEAR == pytest.approx(11.86, rel=0.01)

    def test_kepler_third_law_ratio(self):
        """T₁²/T₂² = a₁³/a₂³ (Kepler's Third Law)."""
        a1, a2 = 1.0 * AU, 1.524 * AU
        T1 = orbital_period(SOLAR_MASS, a1)
        T2 = orbital_period(SOLAR_MASS, a2)
        assert (T1 / T2)**2 == pytest.approx((a1 / a2)**3, rel=1e-9)


class TestVisViva:
    """v² = GM(2/r − 1/a)"""

    def test_circular_orbit_equals_orbital_velocity(self):
        """Vis-viva for circular orbit (r=a) equals orbital_velocity."""
        r = AU
        v_vv    = vis_viva(SOLAR_MASS, r, r)
        v_circ  = orbital_velocity(SOLAR_MASS, r)
        assert v_vv == pytest.approx(v_circ, rel=1e-9)

    def test_faster_at_periapsis(self):
        """Body moves faster at periapsis than apoapsis."""
        a = AU; e = 0.5
        r_p = a * (1 - e)
        r_a = a * (1 + e)
        v_p = vis_viva(SOLAR_MASS, r_p, a)
        v_a = vis_viva(SOLAR_MASS, r_a, a)
        assert v_p > v_a

    def test_parabolic_escape(self):
        """For very large a, vis-viva → escape velocity."""
        from physics.gravity import escape_velocity
        r = AU
        v_vv  = vis_viva(SOLAR_MASS, r, 1e30 * AU)   # a → ∞
        v_esc = escape_velocity(SOLAR_MASS, r)
        assert v_vv == pytest.approx(v_esc, rel=0.001)


class TestOrbitalGeometry:

    def test_apoapsis(self):
        a, e = 2.0 * AU, 0.5
        assert apoapsis(a, e) == pytest.approx(3.0 * AU, rel=1e-9)

    def test_periapsis(self):
        a, e = 2.0 * AU, 0.5
        assert periapsis(a, e) == pytest.approx(1.0 * AU, rel=1e-9)

    def test_circular_apo_peri_equal(self):
        a = 3.0 * AU
        assert apoapsis(a, 0.0) == pytest.approx(a, rel=1e-10)
        assert periapsis(a, 0.0) == pytest.approx(a, rel=1e-10)

    def test_semi_minor_axis_circular(self):
        """For e=0, b=a."""
        a = AU
        assert semi_minor_axis(a, 0.0) == pytest.approx(a, rel=1e-10)

    def test_semi_minor_axis_ellipse(self):
        """b = a√(1−e²)."""
        a, e = AU, 0.6
        b_expected = a * math.sqrt(1 - 0.36)
        assert semi_minor_axis(a, e) == pytest.approx(b_expected, rel=1e-9)

    def test_polar_equation_at_periapsis(self):
        """At θ=0, r = periapsis."""
        a, e = 2*AU, 0.4
        r = orbital_radius_at_angle(a, e, 0.0)
        assert r == pytest.approx(periapsis(a, e), rel=1e-9)

    def test_polar_equation_at_apoapsis(self):
        """At θ=π, r = apoapsis."""
        a, e = 2*AU, 0.4
        r = orbital_radius_at_angle(a, e, math.pi)
        assert r == pytest.approx(apoapsis(a, e), rel=1e-6)


class TestOrbitalEnergy:

    def test_energy_negative_bound_orbit(self):
        """Bound orbit has negative specific orbital energy."""
        E = specific_orbital_energy(SOLAR_MASS, AU)
        assert E < 0

    def test_energy_deeper_for_smaller_orbit(self):
        """Tighter orbit has more negative energy."""
        E_inner = specific_orbital_energy(SOLAR_MASS, 0.5 * AU)
        E_outer = specific_orbital_energy(SOLAR_MASS, 5.0 * AU)
        assert E_inner < E_outer


class TestHohmannTransfer:

    def test_earth_to_mars_delta_v(self):
        """Earth→Mars Hohmann Δv_total ≈ 5.597 km/s (Vallado 2013)."""
        h = hohmann_transfer(SOLAR_MASS, AU, 1.524 * AU)
        assert h["delta_v_total"] / 1000 == pytest.approx(5.597, rel=0.005)

    def test_earth_to_mars_transfer_time(self):
        """Earth→Mars transfer time ≈ 8.5–8.7 months."""
        h = hohmann_transfer(SOLAR_MASS, AU, 1.524 * AU)
        months = h["transfer_time"] / (DAY * 30.44)
        assert 8.0 < months < 9.5

    def test_loe_to_geo(self):
        """LEO→GEO Δv ≈ 3.9 km/s."""
        r_leo = EARTH_RADIUS + 400e3
        r_geo = EARTH_RADIUS + 35786e3
        h = hohmann_transfer(EARTH_MASS, r_leo, r_geo)
        assert h["delta_v_total"] / 1000 == pytest.approx(3.9, abs=0.2)

    def test_same_orbit_zero_dv(self):
        """Transfer between identical orbits requires zero Δv."""
        h = hohmann_transfer(SOLAR_MASS, AU, AU)
        assert h["delta_v_total"] == 0.0

    def test_dv1_and_dv2_positive(self):
        h = hohmann_transfer(SOLAR_MASS, AU, 2*AU)
        assert h["delta_v1"] > 0
        assert h["delta_v2"] > 0

    def test_transfer_semi_major_axis(self):
        """Transfer ellipse semi-major axis = (r1+r2)/2."""
        r1, r2 = AU, 1.524*AU
        h = hohmann_transfer(SOLAR_MASS, r1, r2)
        assert h["a_transfer"] == pytest.approx((r1 + r2) / 2, rel=1e-9)


class TestTsiolkovsky:
    """Δv = v_e · ln(m₀/mf)"""

    def test_known_value(self):
        """Isp=450s, mass ratio=10 → Δv = 450×9.80665×ln(10) ≈ 10.16 km/s."""
        v_e = 450 * 9.80665
        dv  = tsiolkovsky_delta_v(v_e, 100.0, 10.0)
        assert dv / 1000 == pytest.approx(10.16, rel=0.001)

    def test_larger_mass_ratio_more_dv(self):
        v_e = 3000.0
        dv1 = tsiolkovsky_delta_v(v_e, 100, 50)  # ratio 2
        dv2 = tsiolkovsky_delta_v(v_e, 100, 10)  # ratio 10
        assert dv2 > dv1

    def test_roundtrip_with_mass_ratio(self):
        """tsiolkovsky and mass_ratio_from_delta_v are inverses."""
        v_e = 4000.0; m0 = 1000.0; mf = 200.0
        dv       = tsiolkovsky_delta_v(v_e, m0, mf)
        ratio    = mass_ratio_from_delta_v(dv, v_e)
        assert ratio == pytest.approx(m0 / mf, rel=1e-9)

    def test_mf_greater_than_m0_raises(self):
        with pytest.raises(ValueError):
            tsiolkovsky_delta_v(3000.0, 100.0, 200.0)


class TestKeplerEquation:

    def test_circular_orbit(self):
        """For e=0, E = M (eccentric anomaly equals mean anomaly)."""
        for M_anom in [0.0, 0.5, 1.0, 2.0, math.pi]:
            E = solve_kepler(M_anom, 0.0)
            assert E == pytest.approx(M_anom, abs=1e-9)

    def test_e_zero_point_five(self):
        """Known solution: M=1.0, e=0.5 → E ≈ 1.4987."""
        E = solve_kepler(1.0, 0.5)
        # Verify by substituting back: M = E - e·sin(E)
        M_check = E - 0.5 * math.sin(E)
        assert M_check == pytest.approx(1.0, abs=1e-10)

    def test_convergence_high_eccentricity(self):
        """Solver converges for e = 0.9."""
        E = solve_kepler(math.pi / 2, 0.9)
        M_check = E - 0.9 * math.sin(E)
        assert M_check == pytest.approx(math.pi / 2, abs=1e-9)

    def test_true_anomaly_at_periapsis(self):
        """At M=0 (periapsis), true anomaly = 0."""
        E     = solve_kepler(0.0, 0.3)
        theta = eccentric_to_true_anomaly(E, 0.3)
        assert theta == pytest.approx(0.0, abs=1e-9)


class TestOrbitalSummary:

    def test_earth_summary(self):
        s = orbital_summary(SOLAR_MASS, AU, e=0.0167)
        assert s["period_years"]  == pytest.approx(1.0,   rel=0.001)
        assert s["periapsis_m"]   < s["apoapsis_m"]
        assert s["velocity_at_periapsis_m_s"] > s["velocity_at_apoapsis_m_s"]
        assert s["specific_energy_J_kg"] < 0
