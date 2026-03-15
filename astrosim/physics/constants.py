"""
AstroSim — Physical & Astronomical Constants
=============================================
All constants follow IAU 2012 / CODATA 2018 standards.
Values are in SI units unless explicitly noted.

References
----------
- IAU 2012 Resolution B2: Astronomical Unit
- CODATA 2018: Fundamental Physical Constants
- NASA/JPL Solar System Dynamics: planetary parameters
"""

# ── Fundamental physical constants ────────────────────────────────────────────

G = 6.67430e-11
"""Gravitational constant [N·m²·kg⁻²] — CODATA 2018."""

C = 2.99792458e8
"""Speed of light in vacuum [m·s⁻¹] — exact, SI definition."""

H_PLANCK = 6.62607015e-34
"""Planck constant [J·s] — exact, SI definition."""

K_BOLTZMANN = 1.380649e-23
"""Boltzmann constant [J·K⁻¹] — exact, SI definition."""

# ── Astronomical distance units ───────────────────────────────────────────────

AU = 1.495978707e11
"""Astronomical Unit [m] — IAU 2012, exact definition."""

LIGHT_YEAR = 9.4607304725808e15
"""Light-year [m] — defined as distance light travels in one Julian year."""

PARSEC = 3.085677581491367e16
"""Parsec [m] — derived from IAU 2012 AU definition."""

LIGHT_SECOND = 2.99792458e8
"""Light-second [m] — equals the speed of light."""

LIGHT_MINUTE = 1.798754748e10
"""Light-minute [m]."""

# ── Time units ────────────────────────────────────────────────────────────────

JULIAN_YEAR = 365.25 * 24 * 3600
"""Julian year [s] — exactly 365.25 days."""

SIDEREAL_YEAR = 365.256363004 * 24 * 3600
"""Sidereal year [s] — Earth's orbital period relative to fixed stars."""

TROPICAL_YEAR = 365.24219878 * 24 * 3600
"""Tropical year [s] — equinox to equinox."""

DAY = 86400.0
"""Solar day [s] — exactly 86400 seconds."""

# ── Solar properties ──────────────────────────────────────────────────────────

SOLAR_MASS = 1.989e30
"""Solar mass [kg] — M☉."""

SOLAR_RADIUS = 6.957e8
"""Solar radius [m] — R☉, IAU 2015 nominal value."""

SOLAR_LUMINOSITY = 3.828e26
"""Solar luminosity [W] — L☉, IAU 2015 nominal value."""

SOLAR_TEMPERATURE = 5778.0
"""Solar effective surface temperature [K]."""

# ── Earth properties ──────────────────────────────────────────────────────────

EARTH_MASS = 5.972167867791379e24
"""Earth mass [kg] — M⊕."""

EARTH_RADIUS_EQUATORIAL = 6.3781e6
"""Earth equatorial radius [m] — IAU 2015 nominal value."""

EARTH_RADIUS_POLAR = 6.3568e6
"""Earth polar radius [m]."""

EARTH_RADIUS = 6.371e6
"""Earth mean radius [m]."""

EARTH_SURFACE_GRAVITY = 9.80665
"""Standard surface gravitational acceleration [m·s⁻²] — ISO 80000-3."""

EARTH_ORBITAL_VELOCITY = 29784.69
"""Earth mean orbital velocity around the Sun [m·s⁻¹]."""

EARTH_ESCAPE_VELOCITY = 11186.0
"""Earth surface escape velocity [m·s⁻¹]."""

EARTH_ORBITAL_PERIOD = 365.25 * DAY
"""Earth orbital period [s] — Julian year approximation."""

# ── Planetary data: (mass [kg], radius [m], semi-major axis [AU], eccentricity)
# ── Source: NASA Planetary Fact Sheets ───────────────────────────────────────

PLANETS = {
    "mercury": {
        "mass": 3.301e23,
        "radius": 2.4397e6,
        "semi_major_axis_au": 0.387098,
        "eccentricity": 0.205630,
        "orbital_period_days": 87.969,
        "symbol": "☿",
    },
    "venus": {
        "mass": 4.8675e24,
        "radius": 6.0518e6,
        "semi_major_axis_au": 0.723332,
        "eccentricity": 0.006772,
        "orbital_period_days": 224.701,
        "symbol": "♀",
    },
    "earth": {
        "mass": 5.972e24,
        "radius": 6.371e6,
        "semi_major_axis_au": 1.000000,
        "eccentricity": 0.016710,
        "orbital_period_days": 365.256,
        "symbol": "⊕",
    },
    "mars": {
        "mass": 6.4171e23,
        "radius": 3.3895e6,
        "semi_major_axis_au": 1.523679,
        "eccentricity": 0.093400,
        "orbital_period_days": 686.971,
        "symbol": "♂",
    },
    "jupiter": {
        "mass": 1.8982e27,
        "radius": 6.9911e7,
        "semi_major_axis_au": 5.204267,
        "eccentricity": 0.048900,
        "orbital_period_days": 4332.589,
        "symbol": "♃",
    },
    "saturn": {
        "mass": 5.6834e26,
        "radius": 5.8232e7,
        "semi_major_axis_au": 9.537070,
        "eccentricity": 0.056500,
        "orbital_period_days": 10759.22,
        "symbol": "♄",
    },
    "uranus": {
        "mass": 8.6810e25,
        "radius": 2.5362e7,
        "semi_major_axis_au": 19.19126,
        "eccentricity": 0.046381,
        "orbital_period_days": 30685.4,
        "symbol": "⛢",
    },
    "neptune": {
        "mass": 1.02413e26,
        "radius": 2.4622e7,
        "semi_major_axis_au": 30.06896,
        "eccentricity": 0.009456,
        "orbital_period_days": 60189.0,
        "symbol": "♆",
    },
}

# ── Unit conversion helpers ───────────────────────────────────────────────────

def au_to_meters(au: float) -> float:
    """Convert Astronomical Units to meters."""
    return au * AU


def meters_to_au(m: float) -> float:
    """Convert meters to Astronomical Units."""
    return m / AU


def ly_to_meters(ly: float) -> float:
    """Convert light-years to meters."""
    return ly * LIGHT_YEAR


def meters_to_ly(m: float) -> float:
    """Convert meters to light-years."""
    return m / LIGHT_YEAR


def pc_to_meters(pc: float) -> float:
    """Convert parsecs to meters."""
    return pc * PARSEC


def meters_to_pc(m: float) -> float:
    """Convert meters to parsecs."""
    return m / PARSEC


def light_travel_time(distance_m: float) -> float:
    """
    Compute the light travel time for a given distance.

    Parameters
    ----------
    distance_m : float
        Distance in meters.

    Returns
    -------
    float
        Travel time in seconds.
    """
    return distance_m / C
