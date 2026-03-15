"""
AstroSim — Gravitational Physics Engine
========================================
Implements Newtonian gravitation: forces, potentials, accelerations,
tidal effects, and the Schwarzschild radius (GR boundary condition).

All functions operate in SI units unless documented otherwise.

Physical foundation
-------------------
Newton's Law of Universal Gravitation (Principia, 1687):

    F = G · m₁ · m₂ / r²

where G = 6.67430 × 10⁻¹¹ N·m²·kg⁻² is the gravitational constant.

References
----------
- Newton, I. (1687). Philosophiæ Naturalis Principia Mathematica.
- Schwarzschild, K. (1916). Über das Gravitationsfeld eines Massenpunktes.
- Carroll, B. W. & Ostlie, D. A. (2017). Introduction to Modern Astrophysics.
"""

import math
from physics.constants import G, C


# ── Core gravitational functions ──────────────────────────────────────────────

def gravitational_force(m1: float, m2: float, r: float) -> float:
    """
    Compute the gravitational force between two point masses.

    Uses Newton's Law of Universal Gravitation:
        F = G · m₁ · m₂ / r²

    Parameters
    ----------
    m1 : float
        Mass of the first body [kg].
    m2 : float
        Mass of the second body [kg].
    r : float
        Distance between the centres of mass [m]. Must be > 0.

    Returns
    -------
    float
        Magnitude of the gravitational force [N].

    Raises
    ------
    ValueError
        If r ≤ 0 or either mass is negative.

    Examples
    --------
    >>> from physics.constants import EARTH_MASS, SOLAR_MASS, AU
    >>> gravitational_force(SOLAR_MASS, EARTH_MASS, AU)
    3.543e+22  # ≈ 3.54 × 10²² N
    """
    if r <= 0:
        raise ValueError(f"Distance r must be positive, got r={r}")
    if m1 < 0 or m2 < 0:
        raise ValueError("Masses must be non-negative.")
    return G * m1 * m2 / r**2


def gravitational_potential(M: float, r: float) -> float:
    """
    Compute the gravitational potential at distance r from mass M.

    The potential (energy per unit mass) is:
        Φ = −G · M / r

    Negative because gravity is attractive; zero at infinity.

    Parameters
    ----------
    M : float
        Source mass [kg].
    r : float
        Distance from the mass centre [m]. Must be > 0.

    Returns
    -------
    float
        Gravitational potential [J·kg⁻¹].

    Raises
    ------
    ValueError
        If r ≤ 0 or M is negative.
    """
    if r <= 0:
        raise ValueError(f"Distance r must be positive, got r={r}")
    if M < 0:
        raise ValueError("Mass M must be non-negative.")
    return -G * M / r


def gravitational_potential_energy(m1: float, m2: float, r: float) -> float:
    """
    Compute the gravitational potential energy of a two-body system.

    Defined as the work done to assemble the system from infinity:
        U = −G · m₁ · m₂ / r

    Parameters
    ----------
    m1 : float
        Mass of the first body [kg].
    m2 : float
        Mass of the second body [kg].
    r : float
        Separation between the bodies [m]. Must be > 0.

    Returns
    -------
    float
        Gravitational potential energy [J]. Always negative (bound system).
    """
    if r <= 0:
        raise ValueError(f"Distance r must be positive, got r={r}")
    return -G * m1 * m2 / r


def gravitational_acceleration(M: float, r: float) -> float:
    """
    Compute the gravitational acceleration at distance r from mass M.

    Derived from Newton's second law and gravitational force:
        g = G · M / r²

    Parameters
    ----------
    M : float
        Source mass [kg].
    r : float
        Distance from the mass centre [m]. Must be > 0.

    Returns
    -------
    float
        Gravitational acceleration magnitude [m·s⁻²].

    Examples
    --------
    >>> from physics.constants import EARTH_MASS, EARTH_RADIUS
    >>> gravitational_acceleration(EARTH_MASS, EARTH_RADIUS)
    9.819...  # ≈ 9.82 m/s² (matches standard g)
    """
    if r <= 0:
        raise ValueError(f"Distance r must be positive, got r={r}")
    if M < 0:
        raise ValueError("Mass M must be non-negative.")
    return G * M / r**2


# ── Derived gravitational quantities ──────────────────────────────────────────

def escape_velocity(M: float, r: float) -> float:
    """
    Compute the escape velocity from the surface of a body.

    The minimum speed required to escape a gravitational field from
    radius r, reaching infinity with zero kinetic energy:
        v_esc = √(2GM / r)

    Parameters
    ----------
    M : float
        Mass of the body [kg].
    r : float
        Radius at which escape velocity is evaluated [m].

    Returns
    -------
    float
        Escape velocity [m·s⁻¹].

    Examples
    --------
    >>> from physics.constants import EARTH_MASS, EARTH_RADIUS
    >>> escape_velocity(EARTH_MASS, EARTH_RADIUS)
    11184.9...  # ≈ 11.185 km/s
    """
    if r <= 0:
        raise ValueError(f"Radius r must be positive, got r={r}")
    if M < 0:
        raise ValueError("Mass M must be non-negative.")
    return math.sqrt(2 * G * M / r)


def schwarzschild_radius(M: float) -> float:
    """
    Compute the Schwarzschild radius (gravitational radius) of a mass.

    The radius at which the escape velocity equals the speed of light —
    the event horizon of a non-rotating black hole (Schwarzschild 1916):
        r_s = 2GM / c²

    Parameters
    ----------
    M : float
        Mass of the body [kg].

    Returns
    -------
    float
        Schwarzschild radius [m].

    Notes
    -----
    For Earth:   r_s ≈ 8.87 mm
    For the Sun: r_s ≈ 2.95 km

    Examples
    --------
    >>> from physics.constants import SOLAR_MASS
    >>> schwarzschild_radius(SOLAR_MASS)
    2953.4...  # ≈ 2.95 km
    """
    if M < 0:
        raise ValueError("Mass M must be non-negative.")
    return 2 * G * M / C**2


def hill_sphere_radius(M_planet: float, M_star: float, a: float,
                       e: float = 0.0) -> float:
    """
    Compute the Hill sphere radius of a planet orbiting a star.

    The gravitational sphere of influence within which a planet can
    retain satellites against the star's tidal force:
        r_H = a(1−e) · (M_planet / 3M_star)^(1/3)

    Parameters
    ----------
    M_planet : float
        Mass of the planet [kg].
    M_star : float
        Mass of the host star [kg].
    a : float
        Semi-major axis of the planet's orbit [m].
    e : float, optional
        Orbital eccentricity (default: 0 for circular orbit).

    Returns
    -------
    float
        Hill sphere radius [m].
    """
    return a * (1 - e) * (M_planet / (3 * M_star)) ** (1 / 3)


def roche_limit(M_primary: float, R_primary: float,
                density_secondary: float) -> float:
    """
    Compute the Roche limit for a fluid secondary body.

    The distance within which tidal forces exceed the self-gravity of
    the secondary, causing it to disintegrate (Roche 1848):
        d = R_primary · (2 · ρ_primary / ρ_secondary)^(1/3)
    or equivalently using masses and radii directly.

    Parameters
    ----------
    M_primary : float
        Mass of the primary body [kg].
    R_primary : float
        Radius of the primary body [m].
    density_secondary : float
        Mean density of the secondary body [kg·m⁻³].

    Returns
    -------
    float
        Roche limit [m].
    """
    density_primary = M_primary / ((4 / 3) * math.pi * R_primary**3)
    return R_primary * (2 * density_primary / density_secondary) ** (1 / 3)


# ── Two-body problem utilities ────────────────────────────────────────────────

def reduced_mass(m1: float, m2: float) -> float:
    """
    Compute the reduced mass of a two-body system.

    Used to transform a two-body problem into an equivalent one-body
    problem:
        μ = m₁m₂ / (m₁ + m₂)

    Parameters
    ----------
    m1, m2 : float
        Masses of the two bodies [kg].

    Returns
    -------
    float
        Reduced mass [kg].
    """
    return (m1 * m2) / (m1 + m2)


def center_of_mass(m1: float, r1: float, m2: float, r2: float) -> float:
    """
    Compute the centre-of-mass position of a two-body system (1D).

    Parameters
    ----------
    m1, m2 : float
        Masses [kg].
    r1, r2 : float
        Positions [m] along a chosen axis.

    Returns
    -------
    float
        Centre-of-mass position [m].
    """
    return (m1 * r1 + m2 * r2) / (m1 + m2)


# ── Tidal effects ─────────────────────────────────────────────────────────────

def tidal_acceleration(M: float, r: float, dr: float) -> float:
    """
    Compute the differential (tidal) gravitational acceleration.

    The difference in gravitational acceleration across an extended
    body of size dr at distance r from mass M:
        a_tidal ≈ 2GM · dr / r³

    Parameters
    ----------
    M : float
        Tidal source mass [kg].
    r : float
        Distance to the source [m].
    dr : float
        Size of the test body [m].

    Returns
    -------
    float
        Tidal acceleration [m·s⁻²].
    """
    if r <= 0:
        raise ValueError(f"Distance r must be positive, got r={r}")
    return 2 * G * M * dr / r**3


# ── Gravitational wave (order-of-magnitude) ───────────────────────────────────

def gravitational_wave_power(m1: float, m2: float, r: float) -> float:
    """
    Estimate gravitational wave power radiated by two masses (Peters 1964).

    Quadrupole approximation for a circular binary:
        P = −32/5 · G⁴/c⁵ · (m₁m₂)²(m₁+m₂) / r⁵

    Parameters
    ----------
    m1, m2 : float
        Masses of the two bodies [kg].
    r : float
        Orbital separation [m].

    Returns
    -------
    float
        Power radiated as gravitational waves [W]. Positive value.

    Notes
    -----
    Valid for the Newtonian (weak-field, slow-motion) regime only.
    """
    if r <= 0:
        raise ValueError(f"Orbital separation must be positive, got r={r}")
    return (32 / 5) * (G**4 / C**5) * (m1 * m2)**2 * (m1 + m2) / r**5
