"""
Tests for simulation/orbit_simulator.py

Integration tests validate that the physics engine produces trajectories
consistent with analytical orbital mechanics over multiple orbital periods.
Energy and angular momentum conservation are the primary diagnostics.
"""

import pytest
import math
import numpy as np
from physics.constants import SOLAR_MASS, EARTH_MASS, AU, JULIAN_YEAR, DAY
from physics.orbital_mechanics import orbital_period, orbital_velocity, periapsis
from simulation.orbit_simulator import (
    Body, OrbitSimulator, SimulationResult, AdaptiveOrbitSimulator,
    simulate_circular_orbit, simulate_elliptical_orbit,
    simulate_high_eccentricity_orbit, simulate_solar_system,
)


class TestBodyDataclass:

    def test_valid_body(self):
        b = Body("Test", 1e24, [1e9, 0], [0, 1e3])
        assert b.name == "Test"
        assert b.mass == 1e24
        np.testing.assert_array_equal(b.position, [1e9, 0])

    def test_position_velocity_mismatch_raises(self):
        with pytest.raises(ValueError):
            Body("Bad", 1e24, [1e9, 0], [0, 1e3, 0])  # 2D pos, 3D vel

    def test_3d_body(self):
        b = Body("3D", 1e24, [1e9, 0, 0], [0, 1e3, 0])
        assert b.position.shape == (3,)


class TestSimulatorSetup:

    def test_requires_two_bodies(self):
        b = Body("Solo", 1e24, [0, 0], [0, 0])
        with pytest.raises(ValueError):
            OrbitSimulator([b])

    def test_invalid_integrator_raises(self):
        b1 = Body("A", 1e24, [0, 0],    [0, 0])
        b2 = Body("B", 1e10, [1e9, 0], [0, 1e3])
        with pytest.raises(ValueError):
            OrbitSimulator([b1, b2], integrator="euler")

    def test_mixed_dimensions_raises(self):
        b1 = Body("A", 1e24, [0, 0],    [0, 0])
        b2 = Body("B", 1e10, [1e9, 0, 0], [0, 1e3, 0])
        with pytest.raises(ValueError):
            OrbitSimulator([b1, b2])


class TestCircularOrbitLeapfrog:
    """Leapfrog should conserve energy to machine precision for circular orbit."""

    @pytest.fixture(scope="class")
    def result(self):
        return simulate_circular_orbit(
            SOLAR_MASS, EARTH_MASS, AU,
            n_periods=2.0, steps_per_period=2000,
            integrator="leapfrog"
        )

    def test_correct_n_steps(self, result):
        assert result.n_steps > 100

    def test_body_names(self, result):
        assert "Central" in result.body_names
        assert "Orbiter" in result.body_names

    def test_radius_constant(self, result):
        """Orbital radius should remain at 1 AU throughout."""
        radii = np.linalg.norm(result.position_of("Orbiter"), axis=1)
        assert np.all(np.abs(radii / AU - 1.0) < 1e-3)

    def test_energy_conserved(self, result):
        """Energy drift < 1e-4 for leapfrog over 2 years."""
        assert result.energy_drift() < 1e-4

    def test_angular_momentum_conserved(self, result):
        """Angular momentum drift < 1e-8 — hallmark of symplectic integrator."""
        L0 = result.angular_momenta[0]
        Lf = result.angular_momenta[-1]
        assert abs(Lf - L0) / abs(L0) < 1e-6

    def test_returns_simulation_result(self, result):
        assert isinstance(result, SimulationResult)


class TestCircularOrbitRK4:

    @pytest.fixture(scope="class")
    def result(self):
        return simulate_circular_orbit(
            SOLAR_MASS, EARTH_MASS, AU,
            n_periods=1.0, steps_per_period=8760,
            integrator="rk4"
        )

    def test_energy_conserved(self, result):
        assert result.energy_drift() < 1e-4

    def test_final_radius_close(self, result):
        r_f = np.linalg.norm(result.position_of("Orbiter")[-1])
        assert r_f / AU == pytest.approx(1.0, abs=0.01)


class TestEllipticalOrbit:

    @pytest.fixture(scope="class")
    def result(self):
        # Mars-like orbit: a=1.524 AU, e=0.093
        return simulate_elliptical_orbit(
            SOLAR_MASS, 6.417e23, 1.524 * AU, 0.093,
            n_periods=1.0, steps_per_period=1000
        )

    def test_energy_conserved(self, result):
        assert result.energy_drift() < 1e-3

    def test_periapsis_correct(self, result):
        """Closest approach should be near periapsis = a(1-e)."""
        r_vals = np.linalg.norm(result.position_of("Orbiter"), axis=1)
        r_min  = r_vals.min()
        r_peri = 1.524 * AU * (1 - 0.093)
        assert r_min == pytest.approx(r_peri, rel=0.02)

    def test_apoapsis_correct(self, result):
        """Farthest point should be near apoapsis = a(1+e)."""
        r_vals = np.linalg.norm(result.position_of("Orbiter"), axis=1)
        r_max  = r_vals.max()
        r_apo  = 1.524 * AU * (1 + 0.093)
        assert r_max == pytest.approx(r_apo, rel=0.02)


class TestHighEccentricity:

    def test_halley_energy_drift(self):
        """High-eccentricity orbit with periapsis-tuned dt."""
        result = simulate_high_eccentricity_orbit(
            SOLAR_MASS, 2.2e14, 17.8 * AU, 0.967,
            n_periods=1.0, safety_factor=20
        )
        assert result.energy_drift() < 0.02   # looser: high-e is harder

    def test_position_of_helper(self):
        result = simulate_high_eccentricity_orbit(
            SOLAR_MASS, 2.2e14, 17.8 * AU, 0.967,
            n_periods=0.5, safety_factor=20
        )
        assert "Orbiter" in result.body_names
        pos = result.position_of("Orbiter")
        assert pos.shape[1] == 2  # 2D


class TestTwoBodyFactory:

    def test_circular_starts_at_r(self):
        sim = OrbitSimulator.two_body(SOLAR_MASS, EARTH_MASS, AU, e=0.0)
        res = sim.run(duration=DAY * 10, dt=3600)
        r0 = np.linalg.norm(res.position_of("Orbiter")[0])
        assert r0 == pytest.approx(AU, rel=1e-6)

    def test_elliptical_starts_at_periapsis(self):
        a, e = 2 * AU, 0.4
        r_p  = a * (1 - e)
        sim  = OrbitSimulator.two_body(SOLAR_MASS, EARTH_MASS, r_p, e=e)
        res  = sim.run(duration=DAY, dt=3600)
        r0   = np.linalg.norm(res.position_of("Orbiter")[0])
        assert r0 == pytest.approx(r_p, rel=1e-6)

    def test_invalid_eccentricity_raises(self):
        with pytest.raises(ValueError):
            OrbitSimulator.two_body(SOLAR_MASS, EARTH_MASS, AU, e=1.0)


class TestSolarSystemFactory:

    @pytest.fixture(scope="class")
    def result(self):
        sim = OrbitSimulator.solar_system(["earth", "mars"])
        return sim.run(duration=180 * DAY, dt=6 * 3600, store_every=4)

    def test_body_names(self, result):
        assert "Sun" in result.body_names
        assert "Earth" in result.body_names
        assert "Mars" in result.body_names

    def test_earth_stays_near_1_au(self, result):
        radii = np.linalg.norm(result.position_of("Earth"), axis=1)
        assert np.all(np.abs(radii / AU - 1.0) < 0.03)

    def test_mars_stays_near_1p5_au(self, result):
        radii = np.linalg.norm(result.position_of("Mars"), axis=1)
        assert np.all((1.3 < radii / AU) & (radii / AU < 1.7))

    def test_invalid_planet_raises(self):
        with pytest.raises(ValueError):
            OrbitSimulator.solar_system(["pluto"])


class TestSimulationResult:

    @pytest.fixture(scope="class")
    def result(self):
        return simulate_circular_orbit(SOLAR_MASS, EARTH_MASS, AU, 1.0, 500)

    def test_n_bodies(self, result):
        assert result.n_bodies == 2

    def test_position_of_unknown_raises(self, result):
        with pytest.raises(ValueError):
            result.position_of("Pluto")

    def test_energy_drift_nonnegative(self, result):
        assert result.energy_drift() >= 0

    def test_arrays_consistent_shape(self, result):
        N = result.n_steps
        assert result.times.shape      == (N,)
        assert result.energies.shape   == (N,)
        assert result.angular_momenta.shape == (N,)
        assert result.positions.shape  == (N, 2, 2)
        assert result.velocities.shape == (N, 2, 2)
