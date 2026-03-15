"""
AstroSim — pytest configuration and shared fixtures.

Run all tests:
    pytest tests/ -v

Run with coverage:
    pytest tests/ -v --cov=physics --cov=simulation --cov-report=term-missing
"""

import sys
import os
import pytest
import numpy as np

# Ensure the project root is on the path regardless of where pytest is called
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Shared physical constants (imported fresh to avoid circular issues) ───────

@pytest.fixture(scope="session")
def constants():
    """Return a dict of key physical constants for use in tests."""
    from physics.constants import (
        G, AU, C, SOLAR_MASS, EARTH_MASS, EARTH_RADIUS,
        LIGHT_YEAR, PARSEC, DAY, JULIAN_YEAR, PLANETS
    )
    return dict(
        G=G, AU=AU, C=C,
        SOLAR_MASS=SOLAR_MASS,
        EARTH_MASS=EARTH_MASS,
        EARTH_RADIUS=EARTH_RADIUS,
        LIGHT_YEAR=LIGHT_YEAR,
        PARSEC=PARSEC,
        DAY=DAY,
        JULIAN_YEAR=JULIAN_YEAR,
        PLANETS=PLANETS,
    )


@pytest.fixture(scope="session")
def earth_orbit_params(constants):
    """Real Earth orbital parameters for use in integration tests."""
    return dict(
        M=constants["SOLAR_MASS"],
        m=constants["EARTH_MASS"],
        r=constants["AU"],
        e=0.01671,
    )
