# Diego Palencia

**Computational science developer** working at the intersection of astrophysics, orbital mechanics, and scientific visualization.

Building tools that make physics explorable.

---

## AstroSim — Interactive Astrophysics Simulation Lab

> An open-source platform for gravitational dynamics, orbital mechanics, and astronomical scale exploration. Physics engine calibrated to IAU 2012 and CODATA 2018 standards.

**[Live Demo →](https://your-astrosim.streamlit.app)** &nbsp;|&nbsp; **[Repository →](https://github.com/diegopalencia/astrosim)** &nbsp;|&nbsp; **[Paper (JOSS) →](https://doi.org/10.21105/joss.XXXXX)**

```
v² = GM(2/r − 1/a)     ← vis-viva equation: speed at any orbital point
T² = 4π²a³/GM          ← Kepler's third law: period from semi-major axis
v_esc = √(2GM/R)        ← escape velocity from any body
Δv = vₑ · ln(m₀/mf)    ← Tsiolkovsky: propellant for any mission
```

**What it simulates:**
- Keplerian orbits (circular, elliptical, any eccentricity)
- N-body gravitational dynamics — RK4 and symplectic Leapfrog integrators
- Hohmann transfer orbits with full Δv budget
- Escape velocity and Schwarzschild radius for any body
- Astronomical distances: meters → parsecs, 24 orders of magnitude
- Vigesimal (base-20) Maya notation for any astronomical distance — 
  orbital periods expressed in Long Count calendar (Venus: 0.0.1.11.4)

**Verified against:**
| Quantity | AstroSim | Reference | Error |
|---|---|---|---|
| Earth orbital period | 0.9999 yr | 1.0000 yr | <0.01% |
| Earth orbital speed | 29,789 m/s | 29,784.69 m/s | <0.02% |
| Earth→Mars Δv | 5.597 km/s | 5.59 km/s | <0.1% |
| Earth escape velocity | 11,186 m/s | 11,186 m/s | <0.001% |

---

## Stack

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=flat-square&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-1.11+-8CAAE6?style=flat-square&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

---

## Physics modules

```
astrosim/
├── app.py                 ← Streamlit entry point
├── requirements.txt       ← numpy, matplotlib, scipy, streamlit
├── README.md              ← your scientific portfolio face
├── physics/
│   ├── __init__.py
│   ├── constants.py       ← G, AU, solar mass, etc.
│   ├── gravity.py         ← Newton's law functions
│   └── orbital_mechanics.py ← orbital velocity, period, Kepler
    └── vigesimal.py
├── tests/
│   ├── conftest.py          
│   ├── test_constants.py     
│   ├── test_gravity.py    
│   ├── test_orbital_mechanics.py 
│   ├── test_vigesimal.py   
│   └── test_orbit_simulator.py  
├── simulation/
│   ├── __init__.py
│   └── orbit_simulator.py ← numerical integration
└── paper/
    └── astrosim_abstract.md ← your preprint seed
```

---

## Goals

Currently building toward research in **computational astrophysics** and **aerospace engineering**.
Interested in orbital dynamics, mission design, and scientific software.

Open to collaborations, research opportunities, and conversations about physics.

---

<sub>All physical constants: IAU 2012 · CODATA 2018 · NASA/JPL Planetary Fact Sheets</sub>
