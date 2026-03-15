# AstroSim: Preprint Seed

**Title:** AstroSim: An Open-Source Interactive Platform for Astrophysics Simulation and Orbital Mechanics Education

**Author:** Diego Palencia
**Affiliation:** Independent Researcher
**Date:** 2025
**Repository:** https://github.com/diegopalencia/astrosim
**License:** MIT

---

## Abstract

Astrophysics and orbital mechanics constitute foundational disciplines within
modern space science and aerospace engineering. However, the mathematical and
conceptual complexity of these subjects frequently creates significant barriers
for students, early-career researchers, and science communicators.

This paper presents **AstroSim**, an open-source interactive platform designed
to facilitate exploration of gravitational dynamics, orbital mechanics, and
astronomical scales through real-time computational simulation.

The platform integrates numerically accurate physical models grounded in
Newtonian gravitation and Keplerian orbital mechanics, calibrated against
IAU 2012 and CODATA 2018 standards. Core modules include:

- A **live orbital simulator** implementing the vis-viva equation and Kepler's
  laws, with support for circular, elliptical, and high-eccentricity orbits
- A **gravitational physics calculator** for force, potential energy, and
  surface gravity
- An **escape velocity and mission planner** with Hohmann transfer budgets
  and Tsiolkovsky rocket equation integration
- A **multi-unit astronomical converter** spanning 24 orders of magnitude,
  including an ethno-mathematical vigesimal (base-20) converter drawing from
  Maya positional notation
- A **numerical N-body integrator** supporting 4th-order Runge–Kutta and
  symplectic Leapfrog schemes

AstroSim is implemented in Python using Streamlit and Matplotlib, freely
deployable as a public web application, and includes a comprehensive test
suite (pytest) with 30+ physics-validated unit and integration tests.

The platform aims to support computational thinking, physics education, and
early exposure to astrophysical modelling in academic and outreach contexts.

---

## Keywords

astrophysics, orbital mechanics, gravitational dynamics, Kepler's laws,
scientific education, Python, Streamlit, computational simulation,
vigesimal numeration, ethno-mathematics, N-body integration

---

## Statement of Need

*(Required by JOSS — explains why this software is needed)*

Existing astrodynamics tools (REBOUND, poliastro, Astropy) prioritise
programmatic interfaces for expert users. AstroSim addresses the complementary
need: an interactive, browser-deployable platform requiring no programming
knowledge, exposing the same physical models through real-time sliders and
visualisations. It is designed for undergraduate students, science communicators,
and self-directed learners entering astrophysics and aerospace engineering.

The inclusion of a vigesimal (base-20) numeration module — inspired by Maya
positional mathematics — introduces an ethno-mathematical dimension absent
from existing scientific simulation software, providing a culturally grounded
entry point for learners from Mesoamerican heritage communities.

---

## Core Equations

```
Newton's law:      F = G·m₁·m₂ / r²
Vis-viva:          v² = GM(2/r − 1/a)
Kepler III:        T = 2π√(a³/GM)
Escape velocity:   v_esc = √(2GM/R)
Schwarzschild r:   r_s = 2GM/c²
Hohmann Δv₁:       Δv₁ = √(GM/r₁)·(√(2r₂/(r₁+r₂)) − 1)
Tsiolkovsky:       Δv = v_e · ln(m₀/mf)
```

---

## Validation Results

| Quantity | AstroSim | Reference | Error |
|---|---|---|---|
| Earth orbital period | 0.9999 yr | 1.0000 yr | < 0.01% |
| Earth orbital speed | 29,789 m/s | 29,784.69 m/s | < 0.02% |
| Earth→Mars Δv (Hohmann) | 5.597 km/s | ~5.59 km/s | < 0.1% |
| Earth escape velocity | 11,186.1 m/s | 11,186 m/s | < 0.001% |
| Solar Schwarzschild radius | 2,954 m | ~2,953 m | < 0.05% |

---

## Target Journals

1. **JOSS** (Journal of Open Source Software) — primary target
   - https://joss.theoj.org
   - Software paper, peer-reviewed, open access, free to publish, issues DOI
   - Section: Astronomy / Physics Education

2. **arXiv preprint** — upload simultaneously
   - Section: `cs.MS` (Mathematical Software)
   - Or: `physics.ed-ph` (Physics Education)

3. **Future:** Astronomy & Computing, European Journal of Physics

---

## Submission Checklist (JOSS)

- [ ] Public GitHub repository with MIT license
- [ ] `requirements.txt` present
- [ ] `paper.tex` + `references.bib` in `paper/` folder
- [ ] `tests/` folder with passing pytest suite
- [ ] Live demo URL (Streamlit deployment)
- [ ] ORCID registered: https://orcid.org/register
- [ ] Statement of need written (see above)
- [ ] Software archive on Zenodo (GitHub → Zenodo integration)

---

## Notes on the Vigesimal Module

The Maya vigesimal (base-20) numeration system is one of the earliest
positional notation systems in the Americas, pre-dating European contact by
over a millennium. The Maya used it for astronomical calendrical calculations
— tracking Venus cycles, solar years, and long-count periods with remarkable
precision.

Including vigesimal representation in an astrophysics tool is not merely
decorative. It:

1. Demonstrates that numerical representation is a **cultural choice**, not a
   physical necessity — the same distance exists regardless of the base used
2. Connects computational physics to **indigenous scientific heritage**
3. Provides a **non-trivial programming challenge** (positional base conversion
   with Maya glyph names) that demonstrates mathematical programming skill
4. Creates a unique **authorial signature** that distinguishes AstroSim from
   every other orbital mechanics tool in existence

In the paper, cite this as an application of ethno-mathematics to scientific
education software. Reference: Ascher, M. (2002). *Mathematics Elsewhere:
An Exploration of Ideas Across Cultures*. Princeton University Press.
