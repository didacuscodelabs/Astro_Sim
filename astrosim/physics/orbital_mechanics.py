"""
AstroSim — Orbital Mechanics Engine
=====================================
Implements classical orbital mechanics based on Kepler's laws and the
vis-viva equation. Covers circular and elliptical orbits, Hohmann
transfers, orbital energy, and angular momentum.

All functions operate in SI units unless documented otherwise.

Physical foundation
-------------------
Kepler's Three Laws (1609–1619):
  1. Orbits are ellipses with the central body at one focus.
  2. The radius vector sweeps equal areas in equal times (conservation
     of angular momentum).
  3. T² ∝ a³  →  T² = 4π²a³ / (GM)

Vis-Viva equation (energy conservation along an orbit):
    v² = GM(2/r − 1/a)

References
----------
- Kepler, J. (1609). Astronomia Nova.
- Curtis, H. D. (2020). Orbital Mechanics for Engineering Students, 4th ed.
- Vallado, D. A. (2013). Fundamentals of Astrodynamics and Applications, 4th ed.
- Bate, Mueller & White (1971). Fundamentals of Astrodynamics.
"""

import math
from typing import Tuple
from physics.constants import G, AU, SOLAR_MASS


# ── Circular orbit ────────────────────────────────────────────────────────────

def orbital_velocity(M: float, r: float) -> float:
    """
    Compute the circular orbital velocity at radius r around mass M.

    Derived from balancing centripetal acceleration with gravity:
        v = √(GM / r)

    Parameters
    ----------
    M : float
        Mass of the central body [kg].
    r : float
        Orbital radius (distance from centre) [m]. Must be > 0.

    Returns
    -------
    float
        Circular orbital velocity [m·s⁻¹].

    Examples
    --------
    >>> from physics.constants import SOLAR_MASS, AU
    >>> orbital_velocity(SOLAR_MASS, AU)
    29784.7...  # ≈ 29.78 km/s — Earth's orbital speed
    """
    if r <= 0:
        raise ValueError(f"Orbital radius must be positive, got r={r}")
    if M <= 0:
        raise ValueError("Central mass M must be positive.")
    return math.sqrt(G * M / r)


def orbital_period(M: float, a: float) -> float:
    """
    Compute the orbital period using Kepler's Third Law.

    Kepler's Third Law:
        T = 2π √(a³ / GM)

    Parameters
    ----------
    M : float
        Mass of the central body [kg].
    a : float
        Semi-major axis of the orbit [m]. Must be > 0.

    Returns
    -------
    float
        Orbital period [s].

    Examples
    --------
    >>> from physics.constants import SOLAR_MASS, AU
    >>> orbital_period(SOLAR_MASS, AU) / (365.25 * 86400)
    1.0000...  # Earth's period = 1 year
    """
    if a <= 0:
        raise ValueError(f"Semi-major axis must be positive, got a={a}")
    if M <= 0:
        raise ValueError("Central mass M must be positive.")
    return 2 * math.pi * math.sqrt(a**3 / (G * M))


# ── Elliptical orbit ──────────────────────────────────────────────────────────

def vis_viva(M: float, r: float, a: float) -> float:
    """
    Compute orbital velocity at any point using the vis-viva equation.

    The vis-viva equation is the fundamental energy conservation
    relation for any conic-section orbit:
        v² = GM(2/r − 1/a)

    Parameters
    ----------
    M : float
        Mass of the central body [kg].
    r : float
        Current distance from the central body [m].
    a : float
        Semi-major axis of the orbit [m].
        Pass a = ∞ (or a very large number) for a parabolic trajectory.

    Returns
    -------
    float
        Orbital speed at radius r [m·s⁻¹].

    Notes
    -----
    - Circular orbit:   r = a  →  v = √(GM/a)
    - Parabolic escape: a → ∞  →  v = √(2GM/r) = escape velocity
    """
    if r <= 0:
        raise ValueError(f"Distance r must be positive, got r={r}")
    if a <= 0:
        raise ValueError(f"Semi-major axis must be positive, got a={a}")
    return math.sqrt(G * M * (2 / r - 1 / a))


def apoapsis(a: float, e: float) -> float:
    """
    Compute the apoapsis distance (farthest point) of an elliptical orbit.

    r_apo = a(1 + e)

    Parameters
    ----------
    a : float
        Semi-major axis [m].
    e : float
        Orbital eccentricity (0 ≤ e < 1 for ellipse).

    Returns
    -------
    float
        Apoapsis distance [m].
    """
    if not (0 <= e < 1):
        raise ValueError(f"Eccentricity must satisfy 0 ≤ e < 1 for ellipse, got e={e}")
    return a * (1 + e)


def periapsis(a: float, e: float) -> float:
    """
    Compute the periapsis distance (closest point) of an elliptical orbit.

    r_peri = a(1 − e)

    Parameters
    ----------
    a : float
        Semi-major axis [m].
    e : float
        Orbital eccentricity (0 ≤ e < 1 for ellipse).

    Returns
    -------
    float
        Periapsis distance [m].
    """
    if not (0 <= e < 1):
        raise ValueError(f"Eccentricity must satisfy 0 ≤ e < 1 for ellipse, got e={e}")
    return a * (1 - e)


def semi_minor_axis(a: float, e: float) -> float:
    """
    Compute the semi-minor axis of an elliptical orbit.

    b = a √(1 − e²)

    Parameters
    ----------
    a : float
        Semi-major axis [m].
    e : float
        Orbital eccentricity (0 ≤ e < 1).

    Returns
    -------
    float
        Semi-minor axis [m].
    """
    if not (0 <= e < 1):
        raise ValueError(f"Eccentricity must be in [0, 1), got e={e}")
    return a * math.sqrt(1 - e**2)


def orbital_radius_at_angle(a: float, e: float, theta: float) -> float:
    """
    Compute the orbital radius at a given true anomaly (angle from periapsis).

    The polar equation of an ellipse with the focus at the origin:
        r = a(1 − e²) / (1 + e·cos θ)

    Parameters
    ----------
    a : float
        Semi-major axis [m].
    e : float
        Orbital eccentricity (0 ≤ e < 1).
    theta : float
        True anomaly [radians] — angle from periapsis.

    Returns
    -------
    float
        Orbital radius at true anomaly θ [m].
    """
    p = a * (1 - e**2)   # semi-latus rectum
    return p / (1 + e * math.cos(theta))


# ── Orbital energy & angular momentum ─────────────────────────────────────────

def specific_orbital_energy(M: float, a: float) -> float:
    """
    Compute the specific orbital energy (energy per unit mass) of an orbit.

    For a bound (elliptical) orbit:
        ε = −GM / (2a)

    Negative for bound orbits; zero for parabolic; positive for hyperbolic.

    Parameters
    ----------
    M : float
        Mass of the central body [kg].
    a : float
        Semi-major axis [m].

    Returns
    -------
    float
        Specific orbital energy [J·kg⁻¹].
    """
    return -G * M / (2 * a)


def specific_angular_momentum(M: float, a: float, e: float) -> float:
    """
    Compute the specific angular momentum of an orbit.

    h = √(GMp)  where p = a(1 − e²) is the semi-latus rectum.

    Parameters
    ----------
    M : float
        Mass of the central body [kg].
    a : float
        Semi-major axis [m].
    e : float
        Orbital eccentricity.

    Returns
    -------
    float
        Specific angular momentum [m²·s⁻¹].
    """
    p = a * (1 - e**2)   # semi-latus rectum
    return math.sqrt(G * M * p)


# ── Hohmann transfer orbit ────────────────────────────────────────────────────

def hohmann_transfer(M: float, r1: float, r2: float) -> dict:
    """
    Compute the parameters of a Hohmann transfer between two circular orbits.

    A Hohmann transfer is the most energy-efficient two-burn manoeuvre
    between coplanar circular orbits. The transfer ellipse has:
        periapsis = r1,  apoapsis = r2

    Parameters
    ----------
    M : float
        Mass of the central body [kg].
    r1 : float
        Radius of the departure (inner) circular orbit [m].
    r2 : float
        Radius of the arrival (outer) circular orbit [m].

    Returns
    -------
    dict with keys:
        delta_v1        : float — first burn Δv [m·s⁻¹]
        delta_v2        : float — second burn Δv [m·s⁻¹]
        delta_v_total   : float — total Δv [m·s⁻¹]
        transfer_time   : float — half-period of the transfer ellipse [s]
        a_transfer      : float — semi-major axis of transfer ellipse [m]
        v1_circular     : float — speed in departure orbit [m·s⁻¹]
        v2_circular     : float — speed in arrival orbit [m·s⁻¹]
        v_peri          : float — speed at transfer periapsis [m·s⁻¹]
        v_apo           : float — speed at transfer apoapsis [m·s⁻¹]

    Notes
    -----
    Works for r2 > r1 (ascent) and r2 < r1 (descent) — Δv values are
    always returned as positive magnitudes.

    Examples
    --------
    >>> from physics.constants import SOLAR_MASS, AU
    >>> h = hohmann_transfer(SOLAR_MASS, AU, 1.524 * AU)  # Earth → Mars
    >>> h['delta_v_total'] / 1000
    5.59...  # ≈ 5.59 km/s total Δv
    """
    if r1 <= 0 or r2 <= 0:
        raise ValueError("Both orbit radii must be positive.")
    if r1 == r2:
        return {k: 0.0 for k in
                ["delta_v1", "delta_v2", "delta_v_total", "transfer_time",
                 "a_transfer", "v1_circular", "v2_circular", "v_peri", "v_apo"]}

    a_t = (r1 + r2) / 2            # semi-major axis of transfer ellipse

    v1_c = orbital_velocity(M, r1)  # circular speed at r1
    v2_c = orbital_velocity(M, r2)  # circular speed at r2

    v_peri = vis_viva(M, r1, a_t)   # transfer orbit speed at periapsis
    v_apo  = vis_viva(M, r2, a_t)   # transfer orbit speed at apoapsis

    dv1 = abs(v_peri - v1_c)        # burn 1: inject into transfer orbit
    dv2 = abs(v2_c   - v_apo)       # burn 2: circularise at destination

    t_transfer = orbital_period(M, a_t) / 2  # half-period

    return {
        "delta_v1":      dv1,
        "delta_v2":      dv2,
        "delta_v_total": dv1 + dv2,
        "transfer_time": t_transfer,
        "a_transfer":    a_t,
        "v1_circular":   v1_c,
        "v2_circular":   v2_c,
        "v_peri":        v_peri,
        "v_apo":         v_apo,
    }


# ── Rocket equation (Tsiolkovsky) ─────────────────────────────────────────────

def tsiolkovsky_delta_v(v_exhaust: float, m0: float, mf: float) -> float:
    """
    Compute the ideal rocket Δv using the Tsiolkovsky rocket equation (1903).

    Δv = v_e · ln(m₀ / m_f)

    where v_e = I_sp · g₀ is the effective exhaust velocity.

    Parameters
    ----------
    v_exhaust : float
        Effective exhaust velocity [m·s⁻¹].
        Equivalently: v_e = specific_impulse_s × 9.80665
    m0 : float
        Initial (wet) mass of the rocket [kg].
    mf : float
        Final (dry) mass after propellant is expended [kg].

    Returns
    -------
    float
        Ideal Δv [m·s⁻¹].

    Raises
    ------
    ValueError
        If mf ≥ m0 (no propellant) or any mass is non-positive.
    """
    if m0 <= 0 or mf <= 0:
        raise ValueError("Masses must be positive.")
    if mf >= m0:
        raise ValueError("Final mass mf must be less than initial mass m0.")
    return v_exhaust * math.log(m0 / mf)


def mass_ratio_from_delta_v(delta_v: float, v_exhaust: float) -> float:
    """
    Compute the required mass ratio for a given Δv.

    Inverse of the Tsiolkovsky equation:
        m₀ / m_f = exp(Δv / v_e)

    Parameters
    ----------
    delta_v : float
        Required velocity change [m·s⁻¹].
    v_exhaust : float
        Effective exhaust velocity [m·s⁻¹].

    Returns
    -------
    float
        Required mass ratio m₀ / m_f (dimensionless).
    """
    if v_exhaust <= 0:
        raise ValueError("Exhaust velocity must be positive.")
    return math.exp(delta_v / v_exhaust)


# ── Kepler's equation (mean → eccentric → true anomaly) ──────────────────────

def solve_kepler(M_anomaly: float, e: float,
                 tol: float = 1e-10, max_iter: int = 100) -> float:
    """
    Solve Kepler's equation for the eccentric anomaly E.

    Kepler's equation:
        M = E − e·sin(E)

    where M is the mean anomaly (proportional to time since periapsis).
    Solved iteratively using Newton-Raphson.

    Parameters
    ----------
    M_anomaly : float
        Mean anomaly [radians].
    e : float
        Orbital eccentricity (0 ≤ e < 1).
    tol : float, optional
        Convergence tolerance (default: 1e-10).
    max_iter : int, optional
        Maximum Newton-Raphson iterations (default: 100).

    Returns
    -------
    float
        Eccentric anomaly E [radians].

    Raises
    ------
    RuntimeError
        If Newton-Raphson fails to converge within max_iter iterations.
    """
    # Initial guess: E ≈ M for small e, E ≈ π for e ≈ 1
    E = M_anomaly if e < 0.8 else math.pi

    for _ in range(max_iter):
        f  = E - e * math.sin(E) - M_anomaly
        fp = 1 - e * math.cos(E)
        dE = -f / fp
        E += dE
        if abs(dE) < tol:
            return E

    raise RuntimeError(
        f"Kepler's equation did not converge after {max_iter} iterations "
        f"(M={M_anomaly:.4f}, e={e:.4f}). Final ΔE={abs(dE):.2e}."
    )


def eccentric_to_true_anomaly(E: float, e: float) -> float:
    """
    Convert eccentric anomaly to true anomaly.

    tan(θ/2) = √((1+e)/(1−e)) · tan(E/2)

    Parameters
    ----------
    E : float
        Eccentric anomaly [radians].
    e : float
        Orbital eccentricity.

    Returns
    -------
    float
        True anomaly θ [radians].
    """
    return 2 * math.atan2(
        math.sqrt(1 + e) * math.sin(E / 2),
        math.sqrt(1 - e) * math.cos(E / 2)
    )


def mean_motion(M: float, a: float) -> float:
    """
    Compute the mean motion (average angular velocity) of an orbit.

    n = √(GM / a³) = 2π / T

    Parameters
    ----------
    M : float
        Mass of the central body [kg].
    a : float
        Semi-major axis [m].

    Returns
    -------
    float
        Mean motion [rad·s⁻¹].
    """
    return math.sqrt(G * M / a**3)


# ── Position in orbit ─────────────────────────────────────────────────────────

def position_in_orbit(M: float, a: float, e: float,
                      t: float, t0: float = 0.0) -> Tuple[float, float]:
    """
    Compute the (x, y) position in the orbital plane at time t.

    Uses Kepler's equation to convert time to true anomaly, then
    converts to Cartesian coordinates with the focus at the origin.

    Parameters
    ----------
    M : float
        Mass of the central body [kg].
    a : float
        Semi-major axis [m].
    e : float
        Orbital eccentricity (0 ≤ e < 1).
    t : float
        Current time [s].
    t0 : float, optional
        Time of periapsis passage [s] (default: 0).

    Returns
    -------
    Tuple[float, float]
        (x, y) position [m] in the orbital plane, with the central
        body at the origin and periapsis along the +x axis.
    """
    n = mean_motion(M, a)
    mean_anom = n * (t - t0) % (2 * math.pi)
    E = solve_kepler(mean_anom, e)
    theta = eccentric_to_true_anomaly(E, e)
    r = orbital_radius_at_angle(a, e, theta)
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return x, y


# ── Convenience: orbital summary ─────────────────────────────────────────────

def orbital_summary(M: float, a: float, e: float = 0.0) -> dict:
    """
    Return a comprehensive summary of orbital parameters.

    Parameters
    ----------
    M : float
        Mass of the central body [kg].
    a : float
        Semi-major axis [m].
    e : float, optional
        Eccentricity (default: 0 for circular orbit).

    Returns
    -------
    dict
        Orbital parameters including period, velocities, distances,
        energy, angular momentum.
    """
    T       = orbital_period(M, a)
    v_circ  = orbital_velocity(M, a)
    r_apo   = apoapsis(a, e)
    r_peri  = periapsis(a, e)
    b       = semi_minor_axis(a, e)
    energy  = specific_orbital_energy(M, a)
    h       = specific_angular_momentum(M, a, e)
    v_peri  = vis_viva(M, r_peri, a)
    v_apo   = vis_viva(M, r_apo,  a)
    n       = mean_motion(M, a)

    return {
        "semi_major_axis_m":          a,
        "semi_minor_axis_m":          b,
        "eccentricity":               e,
        "period_s":                   T,
        "period_days":                T / 86400,
        "period_years":               T / (365.25 * 86400),
        "mean_motion_rad_per_s":      n,
        "circular_velocity_m_s":      v_circ,
        "periapsis_m":                r_peri,
        "apoapsis_m":                 r_apo,
        "velocity_at_periapsis_m_s":  v_peri,
        "velocity_at_apoapsis_m_s":   v_apo,
        "specific_energy_J_kg":       energy,
        "specific_ang_momentum_m2_s": h,
    }
