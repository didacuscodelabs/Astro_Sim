"""
AstroSim — Physics Package
===========================
Exposes the core physics modules for convenient import.

Usage
-----
    from physics.constants import G, AU, SOLAR_MASS
    from physics.gravity import gravitational_force, escape_velocity
    from physics.orbital_mechanics import orbital_velocity, hohmann_transfer
"""

from physics.constants import G, C, AU, SOLAR_MASS, EARTH_MASS, PLANETS
from physics.gravity import (
    gravitational_force,
    gravitational_potential_energy,
    gravitational_acceleration,
    escape_velocity,
    schwarzschild_radius,
    hill_sphere_radius,
)
from physics.orbital_mechanics import (
    orbital_velocity,
    orbital_period,
    vis_viva,
    hohmann_transfer,
    tsiolkovsky_delta_v,
    orbital_summary,
)

__all__ = [
    "G", "C", "AU", "SOLAR_MASS", "EARTH_MASS", "PLANETS",
    "gravitational_force", "gravitational_potential_energy",
    "gravitational_acceleration", "escape_velocity",
    "schwarzschild_radius", "hill_sphere_radius",
    "orbital_velocity", "orbital_period", "vis_viva",
    "hohmann_transfer", "tsiolkovsky_delta_v", "orbital_summary",
]
