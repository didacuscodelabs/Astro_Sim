"""
AstroSim — Numerical Orbit Simulator
======================================
Propagates orbital trajectories by numerically integrating Newton's equations
of motion. Supports both two-body and N-body gravitational systems.

Two integration schemes are provided:

  RK4 (Runge-Kutta 4th order)
    General-purpose, 4 force evaluations per step, O(h⁴) local error.
    Good for short integrations or when high accuracy is needed.

  Leapfrog / Störmer-Verlet
    Symplectic integrator, 1 force evaluation per step, O(h²) local error.
    Conserves energy and angular momentum over long integrations — the
    correct choice for orbital dynamics over many periods.

All units are SI throughout. Positions in metres, velocities in m/s,
masses in kg, time in seconds.

Physical foundation
-------------------
Newton's second law for body i in an N-body system:

    d²rᵢ/dt² = Σⱼ≠ᵢ  G·mⱼ·(rⱼ − rᵢ) / |rⱼ − rᵢ|³

The state vector for body i is [rᵢ, vᵢ] ∈ ℝ⁶.

References
----------
- Hairer, E., Lubich, C., & Wanner, G. (2006). Geometric Numerical
  Integration, 2nd ed. Springer. doi:10.1007/3-540-30666-8
- Leapfrog: Störmer (1907), Verlet (1967)
- RK4: Kutta (1901)
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable

from physics.constants import G


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Body:
    """
    A gravitating body in the simulation.

    Parameters
    ----------
    name : str
        Human-readable label (e.g. "Earth", "Star").
    mass : float
        Mass [kg].
    position : array-like, shape (2,) or (3,)
        Initial position [m]. 2D or 3D.
    velocity : array-like, shape (2,) or (3,)
        Initial velocity [m/s]. Must match position dimensionality.
    color : str, optional
        Matplotlib color string for visualisation (default: 'white').
    radius : float, optional
        Physical radius [m] for collision detection (default: 0).
    """
    name:     str
    mass:     float
    position: np.ndarray
    velocity: np.ndarray
    color:    str   = "white"
    radius:   float = 0.0

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)
        if self.position.shape != self.velocity.shape:
            raise ValueError(
                f"Body '{self.name}': position and velocity must have the "
                f"same shape, got {self.position.shape} and {self.velocity.shape}."
            )


@dataclass
class SimulationResult:
    """
    Container for the output of a completed simulation run.

    Attributes
    ----------
    times : np.ndarray, shape (N_steps,)
        Time values [s].
    positions : np.ndarray, shape (N_steps, N_bodies, N_dim)
        Position history for each body at each timestep [m].
    velocities : np.ndarray, shape (N_steps, N_bodies, N_dim)
        Velocity history [m/s].
    energies : np.ndarray, shape (N_steps,)
        Total mechanical energy of the system [J].
    angular_momenta : np.ndarray, shape (N_steps,)
        Magnitude of total angular momentum [kg·m²·s⁻¹].
    body_names : list of str
        Ordered list of body names.
    """
    times:            np.ndarray
    positions:        np.ndarray
    velocities:       np.ndarray
    energies:         np.ndarray
    angular_momenta:  np.ndarray
    body_names:       List[str]

    @property
    def n_steps(self) -> int:
        return len(self.times)

    @property
    def n_bodies(self) -> int:
        return self.positions.shape[1]

    def position_of(self, name: str) -> np.ndarray:
        """Return position history [N_steps, N_dim] for the named body."""
        idx = self.body_names.index(name)
        return self.positions[:, idx, :]

    def energy_drift(self) -> float:
        """
        Fractional energy drift over the full simulation.

        Returns |E_final - E_initial| / |E_initial|.
        A well-behaved integration should keep this below ~1e-4.
        """
        E0 = self.energies[0]
        if abs(E0) < 1e-100:
            return 0.0
        return abs(self.energies[-1] - E0) / abs(E0)


# ── Force computation ─────────────────────────────────────────────────────────

def _compute_accelerations(positions: np.ndarray,
                           masses:    np.ndarray,
                           softening: float = 0.0) -> np.ndarray:
    """
    Compute gravitational accelerations for all bodies.

    Uses vectorised pairwise computation for efficiency.

    Parameters
    ----------
    positions : np.ndarray, shape (N, D)
        Current positions of all N bodies in D dimensions [m].
    masses : np.ndarray, shape (N,)
        Masses [kg].
    softening : float
        Gravitational softening length [m] to avoid singularities
        when bodies approach closely (default: 0, exact Newtonian).

    Returns
    -------
    np.ndarray, shape (N, D)
        Acceleration of each body [m/s²].
    """
    N, D = positions.shape
    acc  = np.zeros((N, D), dtype=float)

    for i in range(N):
        for j in range(i + 1, N):
            dr   = positions[j] - positions[i]        # displacement vector
            r2   = np.dot(dr, dr) + softening**2       # softened r²
            r3   = r2 ** 1.5                            # r³
            fac  = G / r3

            ai =  fac * masses[j] * dr
            aj = -fac * masses[i] * dr

            acc[i] += ai
            acc[j] += aj

    return acc


def _compute_energy(positions: np.ndarray,
                    velocities: np.ndarray,
                    masses:     np.ndarray) -> float:
    """
    Compute the total mechanical energy (kinetic + potential) of the system.

    E = Σᵢ ½mᵢvᵢ² − Σᵢ<ⱼ G·mᵢ·mⱼ / rᵢⱼ

    Parameters
    ----------
    positions, velocities : np.ndarray, shape (N, D)
    masses : np.ndarray, shape (N,)

    Returns
    -------
    float
        Total mechanical energy [J].
    """
    N = len(masses)
    KE = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))
    PE = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            r = np.linalg.norm(positions[j] - positions[i])
            if r > 0:
                PE -= G * masses[i] * masses[j] / r
    return KE + PE


def _compute_angular_momentum(positions:  np.ndarray,
                              velocities: np.ndarray,
                              masses:     np.ndarray) -> float:
    """
    Compute the magnitude of the total angular momentum vector.

    L = Σᵢ mᵢ (rᵢ × vᵢ)

    For 2D systems: L = Σᵢ mᵢ (xᵢvyᵢ − yᵢvxᵢ)  [scalar z-component]

    Returns
    -------
    float
        |L| [kg·m²·s⁻¹].
    """
    N = len(masses)
    D = positions.shape[1]
    L = np.zeros(3)

    for i in range(N):
        r = positions[i]
        v = velocities[i]
        if D == 2:
            r3 = np.array([r[0], r[1], 0.0])
            v3 = np.array([v[0], v[1], 0.0])
        else:
            r3 = r
            v3 = v
        L += masses[i] * np.cross(r3, v3)

    return float(np.linalg.norm(L))


# ── RK4 integrator ─────────────────────────────────────────────────────────────

def _rk4_step(positions:  np.ndarray,
              velocities: np.ndarray,
              masses:     np.ndarray,
              dt:         float,
              softening:  float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Advance the state by one RK4 step.

    The state is [positions, velocities] ∈ ℝ^(2ND).
    The equation of motion is:
        dr/dt = v
        dv/dt = a(r)

    Four evaluations of a(r) per step; local truncation error O(dt⁵).

    Parameters
    ----------
    positions, velocities : np.ndarray, shape (N, D)
    masses : np.ndarray, shape (N,)
    dt : float  — timestep [s]
    softening : float — softening length [m]

    Returns
    -------
    (new_positions, new_velocities) : np.ndarray, shape (N, D)
    """
    def deriv(pos, vel):
        acc = _compute_accelerations(pos, masses, softening)
        return vel, acc   # d(pos)/dt = vel, d(vel)/dt = acc

    # k1
    dpos1, dvel1 = deriv(positions, velocities)

    # k2
    pos2 = positions  + 0.5 * dt * dpos1
    vel2 = velocities + 0.5 * dt * dvel1
    dpos2, dvel2 = deriv(pos2, vel2)

    # k3
    pos3 = positions  + 0.5 * dt * dpos2
    vel3 = velocities + 0.5 * dt * dvel2
    dpos3, dvel3 = deriv(pos3, vel3)

    # k4
    pos4 = positions  + dt * dpos3
    vel4 = velocities + dt * dvel3
    dpos4, dvel4 = deriv(pos4, vel4)

    # Weighted average
    new_pos = positions  + (dt / 6.0) * (dpos1 + 2*dpos2 + 2*dpos3 + dpos4)
    new_vel = velocities + (dt / 6.0) * (dvel1 + 2*dvel2 + 2*dvel3 + dvel4)

    return new_pos, new_vel


# ── Leapfrog (Störmer-Verlet) integrator ──────────────────────────────────────

def _leapfrog_step(positions:  np.ndarray,
                   velocities: np.ndarray,
                   acc_prev:   np.ndarray,
                   masses:     np.ndarray,
                   dt:         float,
                   softening:  float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Advance the state by one Leapfrog (velocity-Verlet) step.

    The velocity-Verlet variant is:
        r(t+dt) = r(t) + v(t)·dt + ½a(t)·dt²
        a(t+dt) = f(r(t+dt)) / m
        v(t+dt) = v(t) + ½(a(t) + a(t+dt))·dt

    This form is symplectic and time-reversible, conserving the
    symplectic structure of Hamiltonian mechanics.

    Parameters
    ----------
    acc_prev : np.ndarray, shape (N, D)
        Acceleration at the current timestep (passed in to avoid
        recomputing it on each step).

    Returns
    -------
    (new_positions, new_velocities, new_accelerations)
    """
    # Half-kick then drift (velocity-Verlet formulation)
    new_pos  = positions + velocities * dt + 0.5 * acc_prev * dt**2
    new_acc  = _compute_accelerations(new_pos, masses, softening)
    new_vel  = velocities + 0.5 * (acc_prev + new_acc) * dt

    return new_pos, new_vel, new_acc


# ── Main simulator class ──────────────────────────────────────────────────────

class OrbitSimulator:
    """
    Numerical orbit propagator for N gravitating bodies.

    Integrates Newton's equations of motion using either a 4th-order
    Runge-Kutta scheme (RK4) or a symplectic Leapfrog integrator.

    Parameters
    ----------
    bodies : list of Body
        The gravitating bodies to simulate.
    integrator : str
        Integration scheme: 'leapfrog' (default, recommended for orbits)
        or 'rk4' (higher accuracy for short runs).
    softening : float
        Gravitational softening length [m]. Set > 0 to prevent
        singularities in close-encounter scenarios (default: 0).

    Examples
    --------
    Simulate Earth orbiting the Sun for one year:

    >>> from physics.constants import SOLAR_MASS, EARTH_MASS, AU
    >>> from physics.orbital_mechanics import orbital_velocity
    >>> import numpy as np
    >>>
    >>> sun   = Body("Sun",   SOLAR_MASS,  [0, 0],        [0, 0])
    >>> earth = Body("Earth", EARTH_MASS,  [AU, 0],
    ...                                   [0, orbital_velocity(SOLAR_MASS, AU)])
    >>> sim   = OrbitSimulator([sun, earth])
    >>> result = sim.run(duration=365.25 * 86400, dt=3600)
    >>> print(result.energy_drift())   # should be < 1e-5
    """

    def __init__(self,
                 bodies:     List[Body],
                 integrator: str   = "leapfrog",
                 softening:  float = 0.0):

        if len(bodies) < 2:
            raise ValueError("Simulation requires at least 2 bodies.")
        if integrator not in ("leapfrog", "rk4"):
            raise ValueError(f"Unknown integrator '{integrator}'. "
                             f"Choose 'leapfrog' or 'rk4'.")

        self.bodies     = bodies
        self.integrator = integrator
        self.softening  = softening

        self._dim = bodies[0].position.shape[0]
        for b in bodies:
            if b.position.shape[0] != self._dim:
                raise ValueError(
                    "All bodies must have the same spatial dimensionality."
                )

    # ── Public API ──────────────────────────────────────────────────────────

    def run(self,
            duration:         float,
            dt:               float,
            store_every:      int   = 1,
            progress_callback: Optional[Callable[[float], None]] = None
            ) -> SimulationResult:
        """
        Run the simulation for a specified duration.

        Parameters
        ----------
        duration : float
            Total integration time [s].
        dt : float
            Timestep [s]. For stable orbits, use dt < T/1000 where T
            is the shortest orbital period in the system.
            Rule of thumb: dt = orbital_period / 500 for Leapfrog,
                           dt = orbital_period / 200 for RK4.
        store_every : int
            Store state every this many timesteps (default: 1 = all).
            Set to a larger value to reduce memory for long runs.
        progress_callback : callable, optional
            Called with the current simulation time fraction in [0, 1]
            every 1000 steps. Useful for progress bars.

        Returns
        -------
        SimulationResult
            Complete position/velocity/energy history.

        Raises
        ------
        ValueError
            If dt ≤ 0 or duration ≤ 0.
        """
        if dt <= 0:
            raise ValueError(f"Timestep dt must be positive, got {dt}.")
        if duration <= 0:
            raise ValueError(f"Duration must be positive, got {duration}.")

        N      = len(self.bodies)
        D      = self._dim
        n_step = int(math.ceil(duration / dt))
        n_stor = math.ceil(n_step / store_every)

        # Initialise state arrays
        pos = np.array([b.position for b in self.bodies], dtype=float)
        vel = np.array([b.velocity for b in self.bodies], dtype=float)
        mss = np.array([b.mass     for b in self.bodies], dtype=float)

        # Storage
        pos_hist = np.empty((n_stor, N, D), dtype=float)
        vel_hist = np.empty((n_stor, N, D), dtype=float)
        t_hist   = np.empty(n_stor,          dtype=float)
        E_hist   = np.empty(n_stor,          dtype=float)
        L_hist   = np.empty(n_stor,          dtype=float)

        t       = 0.0
        store_i = 0

        # Precompute initial acceleration for leapfrog (avoids extra eval at t=0)
        if self.integrator == "leapfrog":
            acc = _compute_accelerations(pos, mss, self.softening)

        # ── Main integration loop
        for step in range(n_step):

            # Store state
            if step % store_every == 0 and store_i < n_stor:
                pos_hist[store_i] = pos
                vel_hist[store_i] = vel
                t_hist[store_i]   = t
                E_hist[store_i]   = _compute_energy(pos, vel, mss)
                L_hist[store_i]   = _compute_angular_momentum(pos, vel, mss)
                store_i += 1

            # Progress callback every 1000 steps
            if progress_callback and step % 1000 == 0:
                progress_callback(t / duration)

            # Integrate one step
            if self.integrator == "leapfrog":
                pos, vel, acc = _leapfrog_step(
                    pos, vel, acc, mss, dt, self.softening
                )
            else:  # rk4
                pos, vel = _rk4_step(pos, vel, mss, dt, self.softening)

            t += dt

        # Store final state
        if store_i < n_stor:
            pos_hist[store_i] = pos
            vel_hist[store_i] = vel
            t_hist[store_i]   = t
            E_hist[store_i]   = _compute_energy(pos, vel, mss)
            L_hist[store_i]   = _compute_angular_momentum(pos, vel, mss)
            store_i += 1

        return SimulationResult(
            times           = t_hist[:store_i],
            positions       = pos_hist[:store_i],
            velocities      = vel_hist[:store_i],
            energies        = E_hist[:store_i],
            angular_momenta = L_hist[:store_i],
            body_names      = [b.name for b in self.bodies],
        )

    # ── Convenience factory methods ─────────────────────────────────────────

    @classmethod
    def solar_system(cls,
                     planets: List[str] = None,
                     integrator: str    = "leapfrog") -> "OrbitSimulator":
        """
        Create a simulator pre-loaded with real solar system bodies.

        Uses IAU 2012 semi-major axes and CODATA 2018 masses.
        Orbits are initialised as circular in the ecliptic plane.

        Parameters
        ----------
        planets : list of str, optional
            Planet names to include (lowercase). If None, includes
            ['mercury', 'venus', 'earth', 'mars'].
        integrator : str
            Integration scheme (default: 'leapfrog').

        Returns
        -------
        OrbitSimulator
        """
        from physics.constants import SOLAR_MASS, PLANETS, AU
        from physics.orbital_mechanics import orbital_velocity

        if planets is None:
            planets = ["mercury", "venus", "earth", "mars"]

        COLORS = {
            "mercury": "#b5b5b5",
            "venus":   "#e8c060",
            "earth":   "#4499ee",
            "mars":    "#cc4422",
            "jupiter": "#c8a060",
            "saturn":  "#d4b880",
            "uranus":  "#88cccc",
            "neptune": "#4466cc",
        }

        bodies = [Body("Sun", SOLAR_MASS, [0.0, 0.0], [0.0, 0.0],
                       color="#ffcc44", radius=6.957e8)]

        for name in planets:
            if name not in PLANETS:
                raise ValueError(f"Unknown planet '{name}'. "
                                 f"Choose from {list(PLANETS.keys())}.")
            p  = PLANETS[name]
            a  = p["semi_major_axis_au"] * AU
            v  = orbital_velocity(SOLAR_MASS, a)
            bodies.append(Body(
                name      = name.capitalize(),
                mass      = p["mass"],
                position  = np.array([a, 0.0]),
                velocity  = np.array([0.0, v]),
                color     = COLORS.get(name, "white"),
                radius    = p["radius"],
            ))

        return cls(bodies, integrator=integrator)

    @classmethod
    def two_body(cls,
                 M:          float,
                 m:          float,
                 r:          float,
                 e:          float = 0.0,
                 integrator: str   = "leapfrog") -> "OrbitSimulator":
        """
        Create a two-body simulator for a central mass M and orbiting mass m.

        The system is initialised at periapsis (closest approach) with
        the central body at the origin.

        Parameters
        ----------
        M : float
            Central body mass [kg].
        m : float
            Orbiting body mass [kg].
        r : float
            Initial distance from centre (= periapsis) [m].
        e : float
            Orbital eccentricity (0 = circular, default: 0).
        integrator : str
            Integration scheme (default: 'leapfrog').

        Returns
        -------
        OrbitSimulator
        """
        from physics.orbital_mechanics import vis_viva

        # Semi-major axis from periapsis and eccentricity: r_peri = a(1-e)
        if e >= 1.0:
            raise ValueError("Eccentricity must be < 1 for a bound orbit.")
        a   = r / (1 - e) if e > 0 else r
        v   = vis_viva(M, r, a)    # periapsis velocity

        central  = Body("Central", M, [0.0,  0.0],  [0.0,  0.0],  color="#ffcc44")
        orbiting = Body("Orbiter", m, [r,    0.0],  [0.0,  v],    color="#4499ee")

        return cls([central, orbiting], integrator=integrator)


# ── Adaptive timestep wrapper ─────────────────────────────────────────────────

class AdaptiveOrbitSimulator(OrbitSimulator):
    """
    Orbit simulator with simple adaptive timestep control.

    Adjusts dt at each step to keep the local energy error below a
    specified tolerance. Uses RK4 internally (required for error estimation).

    Parameters
    ----------
    bodies : list of Body
    tolerance : float
        Maximum allowed fractional energy change per step (default: 1e-6).
    dt_min, dt_max : float
        Minimum and maximum allowed timestep [s].

    Notes
    -----
    Adaptive control approximately doubles the computation cost compared
    to fixed-step RK4. For most orbital mechanics applications the fixed
    Leapfrog integrator is preferable.
    """

    def __init__(self,
                 bodies:    List[Body],
                 tolerance: float = 1e-6,
                 dt_min:    float = 1.0,
                 dt_max:    float = 1e8):
        super().__init__(bodies, integrator="rk4")
        self.tolerance = tolerance
        self.dt_min    = dt_min
        self.dt_max    = dt_max

    def run(self,
            duration:    float,
            dt:          float,
            store_every: int  = 1,
            progress_callback=None) -> SimulationResult:
        """
        Run with adaptive timestep.

        The initial dt is used as the starting guess.
        dt is multiplied by 0.5 if local energy error exceeds tolerance,
        and multiplied by 1.5 (capped at dt_max) if it is well below.
        """
        N      = len(self.bodies)
        D      = self._dim
        pos = np.array([b.position for b in self.bodies], dtype=float)
        vel = np.array([b.velocity for b in self.bodies], dtype=float)
        mss = np.array([b.mass     for b in self.bodies], dtype=float)

        t         = 0.0
        current_dt = dt
        store_i   = 0
        results   = []
        E_prev    = _compute_energy(pos, vel, mss)

        while t < duration:
            current_dt = min(current_dt, duration - t, self.dt_max)
            current_dt = max(current_dt, self.dt_min)

            new_pos, new_vel = _rk4_step(pos, vel, mss, current_dt, self.softening)
            E_new = _compute_energy(new_pos, new_vel, mss)

            # Local energy error estimate
            if abs(E_prev) > 1e-100:
                err = abs(E_new - E_prev) / abs(E_prev)
            else:
                err = 0.0

            if err > self.tolerance and current_dt > self.dt_min:
                current_dt *= 0.5
                continue

            # Accept step
            pos, vel = new_pos, new_vel
            t += current_dt
            E_prev = E_new

            if store_i % store_every == 0:
                results.append({
                    "t":   t,
                    "pos": pos.copy(),
                    "vel": vel.copy(),
                    "E":   E_new,
                    "L":   _compute_angular_momentum(pos, vel, mss),
                })

            if progress_callback and store_i % 500 == 0:
                progress_callback(t / duration)

            # Grow dt if error is comfortably below tolerance
            if err < self.tolerance * 0.1:
                current_dt = min(current_dt * 1.5, self.dt_max)

            store_i += 1

        n = len(results)
        t_hist   = np.array([r["t"]   for r in results])
        pos_hist = np.array([r["pos"] for r in results])
        vel_hist = np.array([r["vel"] for r in results])
        E_hist   = np.array([r["E"]   for r in results])
        L_hist   = np.array([r["L"]   for r in results])

        return SimulationResult(
            times           = t_hist,
            positions       = pos_hist,
            velocities      = vel_hist,
            energies        = E_hist,
            angular_momenta = L_hist,
            body_names      = [b.name for b in self.bodies],
        )


# ── Quick-run helpers ─────────────────────────────────────────────────────────

def simulate_circular_orbit(M: float,
                             m: float,
                             r: float,
                             n_periods: float = 2.0,
                             steps_per_period: int = 500,
                             integrator: str = "leapfrog") -> SimulationResult:
    """
    Simulate a circular two-body orbit for a specified number of periods.

    Parameters
    ----------
    M : float  — central body mass [kg]
    m : float  — orbiting body mass [kg]
    r : float  — orbital radius [m]
    n_periods : float  — number of orbital periods to simulate (default: 2)
    steps_per_period : int  — integration steps per period (default: 500)
    integrator : str  — 'leapfrog' or 'rk4'

    Returns
    -------
    SimulationResult
    """
    from physics.orbital_mechanics import orbital_period
    T  = orbital_period(M, r)
    dt = T / steps_per_period
    sim = OrbitSimulator.two_body(M, m, r, e=0.0, integrator=integrator)
    return sim.run(duration=n_periods * T, dt=dt)


def simulate_elliptical_orbit(M: float,
                               m: float,
                               a: float,
                               e: float,
                               n_periods: float = 2.0,
                               steps_per_period: int = 800,
                               integrator: str = "leapfrog") -> SimulationResult:
    """
    Simulate an elliptical two-body orbit.

    Parameters
    ----------
    M : float  — central body mass [kg]
    m : float  — orbiting body mass [kg]
    a : float  — semi-major axis [m]
    e : float  — eccentricity [0, 1)
    n_periods : float  — number of periods (default: 2)
    steps_per_period : int  — steps per period (default: 800;
        elliptical orbits need more steps near periapsis)
    integrator : str

    Returns
    -------
    SimulationResult
    """
    from physics.orbital_mechanics import orbital_period, periapsis
    r_peri = periapsis(a, e)
    T      = orbital_period(M, a)
    dt     = T / steps_per_period
    sim    = OrbitSimulator.two_body(M, m, r_peri, e=e, integrator=integrator)
    return sim.run(duration=n_periods * T, dt=dt)


def simulate_high_eccentricity_orbit(M: float,
                                     m: float,
                                     a: float,
                                     e: float,
                                     n_periods: float = 1.0,
                                     safety_factor: float = 10.0,
                                     integrator: str = "leapfrog"
                                     ) -> SimulationResult:
    """
    Simulate a high-eccentricity orbit with a timestep tuned for periapsis.

    For orbits with e > 0.5, the standard steps_per_period approach
    produces large energy errors because the body moves much faster at
    periapsis than at apoapsis. This helper sets dt based on the periapsis
    crossing timescale r_peri / v_peri, ensuring accuracy throughout.

    Parameters
    ----------
    M : float  — central body mass [kg]
    m : float  — orbiting body mass [kg]
    a : float  — semi-major axis [m]
    e : float  — eccentricity (0 ≤ e < 1). For e > 0.8, leapfrog
                 with periapsis-tuned dt is the correct approach.
    n_periods : float  — number of orbital periods (default: 1)
    safety_factor : float  — dt = (r_peri/v_peri) / safety_factor (default: 10)
    integrator : str

    Returns
    -------
    SimulationResult
    """
    from physics.orbital_mechanics import orbital_period, periapsis, vis_viva

    r_peri = periapsis(a, e)
    v_peri = vis_viva(M, r_peri, a)
    T      = orbital_period(M, a)

    dt = r_peri / v_peri / safety_factor
    store_every = max(1, int(T / dt / 2000))  # keep ~2000 stored points

    sim = OrbitSimulator.two_body(M, m, r_peri, e=e, integrator=integrator)
    return sim.run(duration=n_periods * T, dt=dt, store_every=store_every)


def simulate_solar_system(planets:    List[str] = None,
                           n_years:   float      = 2.0,
                           dt_hours:  float      = 6.0) -> SimulationResult:
    """
    Simulate selected solar system planets for n_years.

    Parameters
    ----------
    planets : list of str, optional
        Planet names. Default: ['mercury', 'venus', 'earth', 'mars'].
    n_years : float
        Simulation duration in years (default: 2).
    dt_hours : float
        Timestep in hours (default: 6 — good balance of speed and accuracy).

    Returns
    -------
    SimulationResult
    """
    from physics.constants import JULIAN_YEAR
    dt  = dt_hours * 3600.0
    dur = n_years  * JULIAN_YEAR
    sim = OrbitSimulator.solar_system(planets=planets)
    # Store every 24 hours to reduce memory for long runs
    store_every = max(1, int(24 / dt_hours))
    return sim.run(duration=dur, dt=dt, store_every=store_every)
