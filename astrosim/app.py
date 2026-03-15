"""
AstroSim — Interactive Astrophysics Simulation Lab
====================================================
A Streamlit application for exploring gravitational physics,
orbital mechanics, and astronomical scales through real computation.

Run:
    streamlit run app.py

Author : Diego Palencia
Version: 1.0.0
License: MIT
"""

import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import streamlit as st

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from physics.constants import (
    G, AU, C, SOLAR_MASS, EARTH_MASS, EARTH_RADIUS,
    LIGHT_YEAR, PARSEC, DAY, PLANETS
)
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
    apoapsis,
    periapsis,
    orbital_radius_at_angle,
    hohmann_transfer,
    tsiolkovsky_delta_v,
    orbital_summary,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AstroSim Lab",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global style ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Base */
html, body, [class*="css"] {
    font-family: 'IBM Plex Mono', 'Courier New', monospace;
}
[data-testid="stAppViewContainer"] {
    background-color: #080b10;
}
[data-testid="stSidebar"] {
    background-color: #0d1117;
    border-right: 1px solid #1a2030;
}
[data-testid="stSidebar"] * {
    color: #8b9ab0 !important;
}
[data-testid="stSidebar"] .stRadio label {
    color: #8b9ab0 !important;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #0f1520;
    border: 1px solid #1a2535;
    border-radius: 6px;
    padding: 0.6rem 0.8rem;
}
[data-testid="stMetricLabel"] {
    font-size: 0.65rem !important;
    letter-spacing: 0.15em;
    color: #4a6080 !important;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.05rem !important;
    color: #2dd4bf !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.7rem !important;
}

/* Headers */
h1 { color: #e2d9c8 !important; letter-spacing: 0.04em; font-size: 1.3rem !important; }
h2 { color: #b8c4d4 !important; letter-spacing: 0.06em; font-size: 0.95rem !important;
     text-transform: uppercase; border-bottom: 1px solid #1a2535; padding-bottom: 6px; }
h3 { color: #8b9ab0 !important; font-size: 0.85rem !important; letter-spacing: 0.06em; }

/* Sliders */
[data-testid="stSlider"] {
    padding: 0.2rem 0;
}

/* Expanders */
[data-testid="stExpander"] {
    background: #0d1420;
    border: 1px solid #1a2535 !important;
    border-radius: 6px;
}

/* Select boxes */
[data-testid="stSelectbox"] > div > div {
    background: #0d1420;
    border: 1px solid #1a2535;
    color: #8b9ab0;
    font-size: 0.82rem;
}

/* Number inputs */
[data-testid="stNumberInput"] input {
    background: #0d1420;
    border: 1px solid #1a2535;
    color: #e2d9c8;
    font-family: 'IBM Plex Mono', monospace;
}

/* Formula box */
.formula-box {
    background: #0a1018;
    border: 1px solid #1e3050;
    border-left: 3px solid #e8a020;
    border-radius: 0 4px 4px 0;
    padding: 10px 14px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: #e8a020;
    margin: 8px 0 16px 0;
    line-height: 1.9;
}
.formula-label {
    font-size: 0.65rem;
    color: #3a5070;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 4px;
}

/* Result card */
.result-card {
    background: #0a0f18;
    border: 1px solid #1a2535;
    border-radius: 6px;
    padding: 12px 14px;
    margin: 8px 0;
}
.result-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 5px 0;
    border-bottom: 1px solid #111820;
    font-size: 0.78rem;
}
.result-row:last-child { border-bottom: none; }
.result-label { color: #3a5070; letter-spacing: 0.1em; text-transform: uppercase; font-size: 0.65rem; }
.result-value { color: #2dd4bf; font-family: 'IBM Plex Mono', monospace; font-weight: 500; }

/* Section divider */
.section-divider {
    border: none;
    border-top: 1px solid #111820;
    margin: 18px 0;
}

/* Status bar */
.status-bar {
    background: #0a0d12;
    border-bottom: 1px solid #1a2030;
    padding: 5px 0 8px 0;
    margin-bottom: 20px;
    font-size: 0.68rem;
    color: #2a4060;
    letter-spacing: 0.12em;
}
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ─────────────────────────────────────────────────────

PLOT_BG    = "#08090e"
PLOT_FG    = "#1a2030"
AXIS_COLOR = "#2a3a50"
TEXT_COLOR = "#6a7a90"
AMBER      = "#e8a020"
TEAL       = "#2dd4bf"
BLUE       = "#60a5fa"
CORAL      = "#f87171"
GRID_COLOR = "#111820"

def apply_dark_style(fig, ax):
    fig.patch.set_facecolor(PLOT_BG)
    ax.set_facecolor(PLOT_BG)
    ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(AXIS_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.4, alpha=0.8)
    ax.set_axisbelow(True)

def apply_dark_style_multi(fig, axes):
    fig.patch.set_facecolor(PLOT_BG)
    for ax in axes:
        ax.set_facecolor(PLOT_BG)
        ax.tick_params(colors=TEXT_COLOR, labelsize=7)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(AXIS_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.4, alpha=0.8)
        ax.set_axisbelow(True)

# ── Number formatting helpers ─────────────────────────────────────────────────

def fmt_sci(val, decimals=3):
    """Format a float in compact scientific notation."""
    if val == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(val))))
    if abs(exp) < 4:
        return f"{val:.{decimals}g}"
    mantissa = val / 10**exp
    return f"{mantissa:.{decimals-1}f}×10{_sup(exp)}"

def _sup(n):
    table = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    return str(n).translate(table)

def fmt_velocity(v):
    if v >= 1e6:
        return f"{v/1e3:,.0f} km/s"
    elif v >= 1000:
        return f"{v/1e3:,.3f} km/s"
    return f"{v:,.1f} m/s"

def fmt_distance(m):
    if m >= PARSEC:
        return f"{m/PARSEC:.4f} pc"
    elif m >= LIGHT_YEAR * 0.1:
        return f"{m/LIGHT_YEAR:.4e} ly"
    elif m >= AU * 0.01:
        return f"{m/AU:.4f} AU"
    elif m >= 1e6:
        return f"{m/1e3:,.0f} km"
    return f"{m:,.2f} m"

def fmt_time(s):
    years = s / (365.25 * DAY)
    if years >= 1:
        return f"{years:.4f} yr"
    days = s / DAY
    if days >= 1:
        return f"{days:.2f} days"
    hours = s / 3600
    if hours >= 1:
        return f"{hours:.2f} hr"
    return f"{s:.1f} s"

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
<div style='padding: 10px 0 18px 0;'>
  <div style='font-size:1.1rem; font-weight:500; color:#e2d9c8; letter-spacing:0.1em;'>ASTROSIM</div>
  <div style='font-size:0.62rem; color:#e8a020; letter-spacing:0.2em; margin-top:2px;'>ASTROPHYSICS LAB v1.0</div>
</div>
""", unsafe_allow_html=True)

    MODULE = st.radio(
        "MODULE",
        options=[
            "01 — Orbital Simulator",
            "02 — Gravity Calculator",
            "03 — Escape Velocity",
            "04 — Unit Converter",
        ],
        label_visibility="visible",
    )

    st.markdown("<hr style='border-color:#1a2030; margin: 18px 0;'>", unsafe_allow_html=True)

    st.markdown("""
<div style='font-size:0.62rem; color:#2a4060; letter-spacing:0.15em; line-height:2;'>
G = 6.67430×10⁻¹¹ N·m²·kg⁻²<br>
1 AU = 1.496×10¹¹ m<br>
M☉ = 1.989×10³⁰ kg<br>
c = 2.998×10⁸ m/s
</div>
""", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1a2030; margin: 18px 0;'>", unsafe_allow_html=True)
    st.markdown("""
<div style='font-size:0.6rem; color:#1e3048; letter-spacing:0.1em; line-height:1.8;'>
Newton · Kepler · Tsiolkovsky<br>
IAU 2012 · CODATA 2018
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 01 — ORBITAL SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════

if MODULE == "01 — Orbital Simulator":

    st.markdown("## Orbital Simulator")
    st.markdown("""
<div class='status-bar'>KEPLER · NEWTON · VIS-VIVA &nbsp;|&nbsp; NEWTONIAN MECHANICS &nbsp;|&nbsp; CONIC SECTION ORBITS</div>
""", unsafe_allow_html=True)

    col_ctrl, col_plot = st.columns([1, 2], gap="medium")

    with col_ctrl:

        preset = st.selectbox(
            "PRESET",
            ["Earth (real)", "Mars (real)", "Jupiter (real)",
             "Halley's Comet", "Custom"],
        )
        presets = {
            "Earth (real)":    dict(star_mass=1.0, a_au=1.000, ecc=0.0167),
            "Mars (real)":     dict(star_mass=1.0, a_au=1.524, ecc=0.0934),
            "Jupiter (real)":  dict(star_mass=1.0, a_au=5.204, ecc=0.0489),
            "Halley's Comet":  dict(star_mass=1.0, a_au=17.8,  ecc=0.967),
            "Custom":          dict(star_mass=1.0, a_au=2.5,   ecc=0.3),
        }
        p = presets[preset]

        st.markdown("#### Parameters")

        star_mass_solar = st.slider(
            "Star mass (M☉)", 0.1, 5.0, float(p["star_mass"]), 0.01,
            help="Mass of the central star in solar masses."
        )
        a_au = st.slider(
            "Semi-major axis (AU)", 0.2, 30.0, float(p["a_au"]), 0.01,
            help="Semi-major axis of the orbit in Astronomical Units."
        )
        ecc = st.slider(
            "Eccentricity (e)", 0.0, 0.97, float(p["ecc"]), 0.001,
            help="0 = circle, approaching 1 = very elongated."
        )
        show_elements = st.checkbox("Show orbital elements", value=True)
        show_velocity = st.checkbox("Show velocity vector", value=True)
        show_second_planet = st.checkbox("Add comparison orbit", value=False)
        if show_second_planet:
            a2_au = st.slider("Comparison orbit (AU)", 0.2, 30.0, a_au * 1.5, 0.01)

        # ── Compute
        M_star  = star_mass_solar * SOLAR_MASS
        a       = a_au * AU
        summary = orbital_summary(M_star, a, ecc)

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.markdown("""
<div class='formula-label'>Physical law</div>
<div class='formula-box'>
v² = GM(2/r − 1/a)<br>
T² = 4π²a³/GM<br>
r = a(1−e²)/(1+e·cosθ)
</div>
""", unsafe_allow_html=True)

    with col_plot:

        # ── Draw orbit
        theta = np.linspace(0, 2 * np.pi, 1000)
        r_vals = np.array([orbital_radius_at_angle(a, ecc, t) for t in theta])
        x_vals = r_vals * np.cos(theta) / AU
        y_vals = r_vals * np.sin(theta) / AU

        fig, ax = plt.subplots(figsize=(7, 7))
        apply_dark_style(fig, ax)

        # Faint reference circles
        for ref_r in [1, 5, 10, 20]:
            circle = plt.Circle((0, 0), ref_r, color=GRID_COLOR,
                                 fill=False, linewidth=0.3, linestyle=":")
            ax.add_patch(circle)

        # Main orbit
        ax.plot(x_vals, y_vals, color=TEAL, linewidth=1.2, alpha=0.9, zorder=3)

        # Second orbit
        if show_second_planet:
            a2 = a2_au * AU
            r2 = np.array([orbital_radius_at_angle(a2, 0.0, t) for t in theta])
            ax.plot(r2 * np.cos(theta) / AU, r2 * np.sin(theta) / AU,
                    color=CORAL, linewidth=0.8, alpha=0.5, linestyle="--", zorder=2)

        # Star at focus
        ax.scatter([0], [0], color=AMBER, s=120, zorder=5)
        ax.scatter([0], [0], color=AMBER, s=600, alpha=0.12, zorder=4)
        ax.text(0.05, 0.05, "☀", color=AMBER, fontsize=9, alpha=0.7,
                transform=ax.transData, ha="left")

        # Planet at periapsis
        r_peri = periapsis(a, ecc) / AU
        ax.scatter([r_peri], [0], color=BLUE, s=60, zorder=6)

        # Velocity vector at periapsis
        if show_velocity:
            v_p = vis_viva(M_star, periapsis(a, ecc), a)
            v_scale = 0.5 * a_au / 30
            ax.annotate("", xy=(r_peri, v_scale),
                        xytext=(r_peri, 0),
                        arrowprops=dict(arrowstyle="->",
                                        color=TEAL, lw=1.2))
            ax.text(r_peri + 0.04 * a_au, v_scale * 0.5,
                    f"v = {v_p/1000:.1f} km/s",
                    color=TEAL, fontsize=7, alpha=0.8)

        # Apoapsis marker
        r_apo = apoapsis(a, ecc) / AU
        ax.scatter([-r_apo], [0], color=CORAL, s=25, zorder=5, alpha=0.6)

        # Periapsis and apoapsis labels
        if show_elements:
            ax.text(r_peri + 0.02 * a_au, -0.06 * a_au,
                    f"q = {r_peri:.2f} AU", color=TEXT_COLOR, fontsize=7)
            ax.text(-r_apo - 0.02 * a_au, -0.06 * a_au,
                    f"Q = {r_apo:.2f} AU", color=TEXT_COLOR, fontsize=7,
                    ha="right")
            # Semi-major axis annotation
            ax.annotate("", xy=(r_peri, 0), xytext=(-r_apo, 0),
                        arrowprops=dict(arrowstyle="<->",
                                        color=AXIS_COLOR, lw=0.6))
            mid = (r_peri - r_apo) / 2
            ax.text(mid, 0.04 * a_au, f"2a = {2*a_au:.2f} AU",
                    color=TEXT_COLOR, fontsize=7, ha="center")

        # Second focus
        c_focus = a_au * ecc
        ax.scatter([-c_focus * 2 + r_peri - r_apo], [0],
                   color=AXIS_COLOR, s=10, zorder=4, alpha=0.5)

        lim = r_apo * 1.12 if r_apo > 1 else 1.5
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("x  [AU]", fontsize=8)
        ax.set_ylabel("y  [AU]", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.tight_layout(pad=1.2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Metrics row
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Period",       fmt_time(summary["period_s"]))
    m2.metric("v circular",   fmt_velocity(summary["circular_velocity_m_s"]))
    m3.metric("v periapsis",  fmt_velocity(summary["velocity_at_periapsis_m_s"]))
    m4.metric("v apoapsis",   fmt_velocity(summary["velocity_at_apoapsis_m_s"]))
    m5.metric("Periapsis",    f"{summary['periapsis_m']/AU:.3f} AU")
    m6.metric("Sp. Energy",   fmt_sci(summary["specific_energy_J_kg"]) + " J/kg")

    # ── Velocity vs angle plot
    with st.expander("Velocity profile — full orbit"):
        thetas_deg = np.linspace(0, 360, 720)
        v_profile = [
            vis_viva(M_star, orbital_radius_at_angle(a, ecc, math.radians(th)), a)
            for th in thetas_deg
        ]
        fig2, ax2 = plt.subplots(figsize=(9, 2.8))
        apply_dark_style(fig2, ax2)
        ax2.plot(thetas_deg, [v / 1000 for v in v_profile],
                 color=TEAL, linewidth=1.2)
        ax2.fill_between(thetas_deg, [v / 1000 for v in v_profile],
                         alpha=0.08, color=TEAL)
        ax2.axhline(summary["circular_velocity_m_s"] / 1000,
                    color=AMBER, linewidth=0.7, linestyle="--", alpha=0.6,
                    label="circular velocity")
        ax2.set_xlabel("True anomaly θ  [degrees]", fontsize=8)
        ax2.set_ylabel("Orbital speed  [km/s]", fontsize=8)
        ax2.set_xlim(0, 360)
        ax2.set_xticks([0, 90, 180, 270, 360])
        ax2.legend(fontsize=7, facecolor=PLOT_BG, edgecolor=AXIS_COLOR,
                   labelcolor=TEXT_COLOR)
        fig2.tight_layout(pad=1.0)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 02 — GRAVITY CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════════

elif MODULE == "02 — Gravity Calculator":

    st.markdown("## Gravity Calculator")
    st.markdown("""
<div class='status-bar'>NEWTON 1687 &nbsp;|&nbsp; F = Gm₁m₂/r² &nbsp;|&nbsp; TWO-BODY GRAVITATIONAL INTERACTION</div>
""", unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 2], gap="medium")

    with col_a:

        system = st.selectbox(
            "SYSTEM PRESET",
            ["Earth–Moon", "Sun–Earth", "Earth–ISS", "Sun–Jupiter",
             "Neutron Star–Companion", "Custom"],
        )
        sys_presets = {
            "Earth–Moon":            dict(m1=5.972e24, m2=7.346e22, r=3.844e8),
            "Sun–Earth":             dict(m1=1.989e30, m2=5.972e24, r=AU),
            "Earth–ISS":             dict(m1=5.972e24, m2=4.5e5,    r=6.371e6 + 408e3),
            "Sun–Jupiter":           dict(m1=1.989e30, m2=1.898e27, r=5.204 * AU),
            "Neutron Star–Companion":dict(m1=2.8e30,   m2=1.4e30,   r=1.5e9),
            "Custom":                dict(m1=5.972e24, m2=7.346e22, r=3.844e8),
        }
        sp = sys_presets[system]

        st.markdown("#### Body 1")
        m1_exp = st.slider("Mass 1 — exponent (10^x kg)", 10, 32,
                           int(round(math.log10(sp["m1"]))), 1)
        m1_coeff = st.slider("Mass 1 — coefficient", 1.0, 9.99,
                             float(f"{sp['m1'] / 10**int(round(math.log10(sp['m1']))):.2f}"),
                             0.01)
        m1 = m1_coeff * 10**m1_exp

        st.markdown("#### Body 2")
        m2_exp = st.slider("Mass 2 — exponent (10^x kg)", 1, 32,
                           int(round(math.log10(sp["m2"]))), 1)
        m2_coeff = st.slider("Mass 2 — coefficient", 1.0, 9.99,
                             float(f"{sp['m2'] / 10**int(round(math.log10(sp['m2']))):.2f}"),
                             0.01)
        m2 = m2_coeff * 10**m2_exp

        st.markdown("#### Separation")
        r_exp = st.slider("Distance — exponent (10^x m)", 3, 14,
                          int(round(math.log10(sp["r"]))), 1)
        r_coeff = st.slider("Distance — coefficient", 1.0, 9.99,
                            float(f"{sp['r'] / 10**int(round(math.log10(sp['r']))):.2f}"),
                            0.01)
        r = r_coeff * 10**r_exp

        st.markdown("""
<div class='formula-label'>Physical law</div>
<div class='formula-box'>
F = G·m₁·m₂ / r²<br>
U = −G·m₁·m₂ / r<br>
a₁ = F/m₁ &nbsp; a₂ = F/m₂
</div>
""", unsafe_allow_html=True)

    with col_b:

        # ── Compute
        F  = gravitational_force(m1, m2, r)
        U  = gravitational_potential_energy(m1, m2, r)
        a1 = F / m1
        a2 = F / m2
        v_orb_12 = orbital_velocity(m1, r)

        # ── Metrics
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Force",       fmt_sci(F)  + " N")
        mc2.metric("Potential E", "−" + fmt_sci(abs(U)) + " J")
        mc3.metric("Acc. on m₁",  fmt_sci(a1) + " m/s²")
        mc4.metric("Acc. on m₂",  fmt_sci(a2) + " m/s²")

        # ── Force vs distance plot
        st.markdown("#### Force vs separation")
        r_range = np.logspace(math.log10(r * 0.1), math.log10(r * 10), 500)
        F_range = [gravitational_force(m1, m2, ri) for ri in r_range]

        fig3, ax3 = plt.subplots(figsize=(8, 3.5))
        apply_dark_style(fig3, ax3)
        ax3.loglog(r_range, F_range, color=TEAL, linewidth=1.4)
        ax3.axvline(r, color=AMBER, linewidth=0.8, linestyle="--", alpha=0.8)
        ax3.scatter([r], [F], color=AMBER, s=60, zorder=5)
        ax3.text(r * 1.05, F * 1.5,
                 f"r = {fmt_sci(r)} m\nF = {fmt_sci(F)} N",
                 color=AMBER, fontsize=7, alpha=0.9)
        ax3.set_xlabel("Separation r  [m]", fontsize=8)
        ax3.set_ylabel("Gravitational force F  [N]", fontsize=8)
        ax3.annotate("F ∝ 1/r²", xy=(r_range[-1] * 0.3, F_range[-1] * 8),
                     color=TEXT_COLOR, fontsize=7, alpha=0.6)
        fig3.tight_layout(pad=1.0)
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

        # ── Potential energy vs distance plot
        with st.expander("Potential energy well"):
            r_pe = np.logspace(math.log10(r * 0.2), math.log10(r * 6), 400)
            U_pe = [-G * m1 * m2 / ri for ri in r_pe]

            fig4, ax4 = plt.subplots(figsize=(8, 3.0))
            apply_dark_style(fig4, ax4)
            ax4.semilogx(r_pe, U_pe, color=CORAL, linewidth=1.2)
            ax4.fill_between(r_pe, U_pe, 0, alpha=0.06, color=CORAL)
            ax4.axvline(r, color=AMBER, linewidth=0.8, linestyle="--", alpha=0.6)
            ax4.axhline(0, color=AXIS_COLOR, linewidth=0.5)
            ax4.scatter([r], [U], color=AMBER, s=50, zorder=5)
            ax4.text(r * 1.05, U * 0.6,
                     f"U = {fmt_sci(U)} J",
                     color=AMBER, fontsize=7)
            ax4.set_xlabel("Separation r  [m]", fontsize=8)
            ax4.set_ylabel("Potential energy U  [J]", fontsize=8)
            fig4.tight_layout(pad=1.0)
            st.pyplot(fig4, use_container_width=True)
            plt.close(fig4)

        # ── Comparison table across solar system bodies
        with st.expander("Solar system comparison — force from m₁"):
            planet_names = list(PLANETS.keys())
            distances = [p["semi_major_axis_au"] * AU for p in PLANETS.values()]
            forces_ss  = [gravitational_force(m1, SOLAR_MASS, d) for d in distances]

            fig5, ax5 = plt.subplots(figsize=(8, 3.0))
            apply_dark_style(fig5, ax5)
            colors_bar = [TEAL if n != "earth" else AMBER for n in planet_names]
            bars = ax5.bar(planet_names, forces_ss, color=colors_bar, alpha=0.8,
                           width=0.6, edgecolor=PLOT_BG, linewidth=0.5)
            ax5.set_yscale("log")
            ax5.set_ylabel("F on Sun from m₁ at orbit  [N]", fontsize=7)
            ax5.tick_params(axis="x", labelsize=7)
            fig5.tight_layout(pad=1.0)
            st.pyplot(fig5, use_container_width=True)
            plt.close(fig5)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 03 — ESCAPE VELOCITY & MISSION PLANNER
# ═══════════════════════════════════════════════════════════════════════════════

elif MODULE == "03 — Escape Velocity":

    st.markdown("## Escape Velocity & Mission Planner")
    st.markdown("""
<div class='status-bar'>ESCAPE DYNAMICS &nbsp;|&nbsp; HOHMANN TRANSFERS &nbsp;|&nbsp; TSIOLKOVSKY ROCKET EQUATION</div>
""", unsafe_allow_html=True)

    tab_esc, tab_mission, tab_rocket = st.tabs([
        "Escape Velocity", "Hohmann Transfer", "Rocket Equation"
    ])

    # ── Tab 1: Escape Velocity
    with tab_esc:
        col_e1, col_e2 = st.columns([1, 2], gap="medium")

        with col_e1:
            body_preset = st.selectbox(
                "BODY PRESET",
                ["Earth", "Moon", "Mars", "Venus", "Jupiter", "Sun",
                 "White Dwarf", "Neutron Star", "Custom"],
            )
            body_params = {
                "Earth":        dict(mass=5.972e24, radius=6.371e6),
                "Moon":         dict(mass=7.346e22, radius=1.7374e6),
                "Mars":         dict(mass=6.417e23, radius=3.3895e6),
                "Venus":        dict(mass=4.867e24, radius=6.0518e6),
                "Jupiter":      dict(mass=1.898e27, radius=6.9911e7),
                "Sun":          dict(mass=1.989e30, radius=6.957e8),
                "White Dwarf":  dict(mass=1.2e30,   radius=7e6),
                "Neutron Star": dict(mass=2.8e30,   radius=1.2e4),
                "Custom":       dict(mass=5.972e24, radius=6.371e6),
            }
            bp = body_params[body_preset]

            mass_exp = st.slider("Mass — exponent (10^x kg)", 15, 32,
                                 int(round(math.log10(bp["mass"]))), 1)
            mass_co  = st.slider("Mass — coefficient", 1.0, 9.99,
                                 round(bp["mass"] / 10**int(round(math.log10(bp["mass"]))), 2),
                                 0.01)
            M_body   = mass_co * 10**mass_exp

            rad_exp  = st.slider("Radius — exponent (10^x m)", 3, 11,
                                 int(round(math.log10(bp["radius"]))), 1)
            rad_co   = st.slider("Radius — coefficient", 1.0, 9.99,
                                 round(bp["radius"] / 10**int(round(math.log10(bp["radius"]))), 2),
                                 0.01)
            R_body   = rad_co * 10**rad_exp

            v_esc  = escape_velocity(M_body, R_body)
            v_orb  = orbital_velocity(M_body, R_body)
            g_surf = gravitational_acceleration(M_body, R_body)
            r_sch  = schwarzschild_radius(M_body)
            r_hill = hill_sphere_radius(M_body, SOLAR_MASS, AU)
            compact = R_body / r_sch  # compactness ratio

            st.markdown("""
<div class='formula-label'>Physical law</div>
<div class='formula-box'>
v_esc = √(2GM/R)<br>
r_s = 2GM/c²<br>
g = GM/R²
</div>
""", unsafe_allow_html=True)

        with col_e2:
            ec1, ec2, ec3, ec4 = st.columns(4)
            ec1.metric("Escape velocity", fmt_velocity(v_esc))
            ec2.metric("Orbital velocity", fmt_velocity(v_orb))
            ec3.metric("Surface gravity", f"{g_surf:.2f} m/s²")
            ec4.metric("Schwarzschild r", fmt_distance(r_sch))

            # ── Escape vs radius plot
            st.markdown("#### Escape velocity vs radius")
            r_range = np.logspace(math.log10(R_body * 0.1),
                                  math.log10(R_body * 20), 500)
            v_esc_range = [escape_velocity(M_body, ri) for ri in r_range]
            v_orb_range = [orbital_velocity(M_body, ri) for ri in r_range]

            fig6, ax6 = plt.subplots(figsize=(8, 3.5))
            apply_dark_style(fig6, ax6)
            ax6.loglog(r_range, [v / 1000 for v in v_esc_range],
                       color=AMBER, linewidth=1.3, label="v_escape")
            ax6.loglog(r_range, [v / 1000 for v in v_orb_range],
                       color=TEAL, linewidth=1.0, linestyle="--",
                       label="v_orbital (LEO)", alpha=0.8)
            ax6.axvline(R_body, color=CORAL, linewidth=0.8,
                        linestyle=":", alpha=0.7, label="surface radius")
            ax6.axhline(C / 1000, color="#f472b6", linewidth=0.6,
                        linestyle="--", alpha=0.4, label="speed of light")
            ax6.scatter([R_body], [v_esc / 1000], color=AMBER, s=60, zorder=5)
            ax6.set_xlabel("Radius  [m]", fontsize=8)
            ax6.set_ylabel("Velocity  [km/s]", fontsize=8)
            ax6.legend(fontsize=7, facecolor=PLOT_BG, edgecolor=AXIS_COLOR,
                       labelcolor=TEXT_COLOR)
            fig6.tight_layout(pad=1.0)
            st.pyplot(fig6, use_container_width=True)
            plt.close(fig6)

            # Compactness note
            if compact < 2.5:
                st.warning(f"⚠ Compactness ratio R/r_s = {compact:.1f} — extreme object, GR corrections needed.")
            elif compact < 10:
                st.info(f"ℹ Compactness R/r_s = {compact:.1f} — dense object.")

    # ── Tab 2: Hohmann Transfer
    with tab_mission:
        col_h1, col_h2 = st.columns([1, 2], gap="medium")

        with col_h1:
            mission = st.selectbox(
                "MISSION PRESET",
                ["Earth → Mars", "Earth → Venus", "Earth → Jupiter",
                 "LEO → GEO", "Custom"],
            )
            mission_params = {
                "Earth → Mars":    dict(r1=1.000, r2=1.524),
                "Earth → Venus":   dict(r1=1.000, r2=0.723),
                "Earth → Jupiter": dict(r1=1.000, r2=5.204),
                "LEO → GEO":       dict(r1=6371e3 + 400e3,   r2=6371e3 + 35786e3,
                                        unit="m"),
                "Custom":          dict(r1=1.000, r2=2.500),
            }
            mp = mission_params[mission]
            unit = mp.get("unit", "AU")

            if unit == "m":
                r1_m = mp["r1"]
                r2_m = mp["r2"]
                label1 = f"{mp['r1']/1e6:.0f}×10⁶ m"
                label2 = f"{mp['r2']/1e6:.0f}×10⁶ m"
                M_central = EARTH_MASS
                r1_disp = r1_m / 1e6
                r2_disp = r2_m / 1e6
                r1_m = st.slider("Departure orbit (×10⁶ m)", 6.4, 50.0,
                                  float(r1_disp), 0.1) * 1e6
                r2_m = st.slider("Arrival orbit (×10⁶ m)", 6.5, 50.0,
                                  float(r2_disp), 0.1) * 1e6
            else:
                M_central = SOLAR_MASS
                r1_au = st.slider("Departure orbit (AU)", 0.3, 10.0,
                                   float(mp["r1"]), 0.01)
                r2_au = st.slider("Arrival orbit (AU)", 0.3, 10.0,
                                   float(mp["r2"]), 0.01)
                r1_m = r1_au * AU
                r2_m = r2_au * AU

            h = hohmann_transfer(M_central, r1_m, r2_m)

            st.markdown("""
<div class='formula-label'>Physical law</div>
<div class='formula-box'>
a_t = (r₁+r₂)/2<br>
Δv₁ = v_peri − v_circ1<br>
Δv₂ = v_circ2 − v_apo<br>
t = π√(a³/GM)
</div>
""", unsafe_allow_html=True)

        with col_h2:
            hc1, hc2, hc3, hc4 = st.columns(4)
            hc1.metric("Δv₁", fmt_velocity(h["delta_v1"]))
            hc2.metric("Δv₂", fmt_velocity(h["delta_v2"]))
            hc3.metric("Δv total", fmt_velocity(h["delta_v_total"]))
            hc4.metric("Transfer time", fmt_time(h["transfer_time"]))

            # ── Draw Hohmann diagram
            fig7, ax7 = plt.subplots(figsize=(7, 7))
            apply_dark_style(fig7, ax7)

            theta_full = np.linspace(0, 2 * np.pi, 500)
            r1_au_val  = r1_m / AU if unit != "m" else r1_m / 1e7
            r2_au_val  = r2_m / AU if unit != "m" else r2_m / 1e7
            scale      = r2_au_val if r2_au_val > r1_au_val else r1_au_val

            # Orbits
            ax7.plot(r1_au_val * np.cos(theta_full),
                     r1_au_val * np.sin(theta_full),
                     color=BLUE, linewidth=0.8, alpha=0.6, linestyle="--")
            ax7.plot(r2_au_val * np.cos(theta_full),
                     r2_au_val * np.sin(theta_full),
                     color=CORAL, linewidth=0.8, alpha=0.6, linestyle="--")

            # Transfer ellipse (half)
            a_t_disp = h["a_transfer"] / AU if unit != "m" else h["a_transfer"] / 1e7
            e_t      = abs(r2_au_val - r1_au_val) / (r2_au_val + r1_au_val)
            theta_t  = np.linspace(0, math.pi, 300)
            r_t      = np.array([
                orbital_radius_at_angle(a_t_disp, e_t, th) for th in theta_t
            ])
            offset   = -(a_t_disp * e_t)
            ax7.plot(offset + r_t * np.cos(theta_t),
                     r_t * np.sin(theta_t),
                     color=AMBER, linewidth=1.6, zorder=3)

            # Bodies
            ax7.scatter([0], [0], color=AMBER, s=180, zorder=5)
            ax7.scatter([0], [0], color=AMBER, s=900, alpha=0.1, zorder=4)
            r1_body = r1_au_val if unit != "m" else r1_m / 1e7
            r2_body = r2_au_val if unit != "m" else r2_m / 1e7
            ax7.scatter([r1_body], [0], color=BLUE, s=80, zorder=6, label="Departure")
            ax7.scatter([-r2_body], [0], color=CORAL, s=80, zorder=6, label="Arrival")

            # Burn markers
            ax7.annotate(f"Burn 1\n+{h['delta_v1']/1000:.2f} km/s",
                         xy=(r1_body, 0), xytext=(r1_body + 0.08 * scale, 0.12 * scale),
                         color=TEAL, fontsize=7,
                         arrowprops=dict(arrowstyle="->", color=TEAL, lw=0.8))
            ax7.annotate(f"Burn 2\n+{h['delta_v2']/1000:.2f} km/s",
                         xy=(-r2_body, 0), xytext=(-r2_body - 0.12 * scale, 0.12 * scale),
                         color=TEAL, fontsize=7,
                         arrowprops=dict(arrowstyle="->", color=TEAL, lw=0.8))

            label_unit = "AU" if unit != "m" else "×10⁷ m"
            ax7.legend(fontsize=7, facecolor=PLOT_BG, edgecolor=AXIS_COLOR,
                       labelcolor=TEXT_COLOR)
            lim_h = scale * 1.15
            ax7.set_xlim(-lim_h, lim_h)
            ax7.set_ylim(-lim_h, lim_h)
            ax7.set_aspect("equal")
            ax7.set_xlabel(f"x  [{label_unit}]", fontsize=8)
            ax7.set_ylabel(f"y  [{label_unit}]", fontsize=8)
            fig7.tight_layout(pad=1.2)
            st.pyplot(fig7, use_container_width=True)
            plt.close(fig7)

    # ── Tab 3: Rocket Equation
    with tab_rocket:
        col_r1, col_r2 = st.columns([1, 2], gap="medium")

        with col_r1:
            st.markdown("#### Propellant")
            isp = st.slider("Specific impulse Isp [s]", 200, 4500, 450, 10,
                            help="Chemical: 250–460 s. Ion thruster: 1000–10000 s.")
            m0_tons  = st.slider("Initial (wet) mass [tonnes]", 1.0, 5000.0, 500.0, 1.0)
            mf_tons  = st.slider("Final (dry) mass [tonnes]",  0.1, 500.0, 70.0, 0.1)

            g0 = 9.80665
            v_exhaust = isp * g0
            m0 = m0_tons * 1e3
            mf = min(mf_tons * 1e3, m0 * 0.999)

            dv = tsiolkovsky_delta_v(v_exhaust, m0, mf)
            mass_ratio = m0 / mf
            prop_fraction = (m0 - mf) / m0

            st.markdown("""
<div class='formula-label'>Tsiolkovsky 1903</div>
<div class='formula-box'>
Δv = v_e · ln(m₀/m_f)<br>
v_e = Isp · g₀
</div>
""", unsafe_allow_html=True)

        with col_r2:
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("Δv",            fmt_velocity(dv))
            rc2.metric("Exhaust vel.",  fmt_velocity(v_exhaust))
            rc3.metric("Mass ratio",    f"{mass_ratio:.2f}")
            rc4.metric("Propellant %",  f"{prop_fraction*100:.1f}%")

            # ── Δv vs mass ratio
            ratios = np.linspace(1.01, 50, 400)
            dvs    = [v_exhaust * math.log(r) / 1000 for r in ratios]

            fig8, ax8 = plt.subplots(figsize=(8, 3.5))
            apply_dark_style(fig8, ax8)
            ax8.plot(ratios, dvs, color=TEAL, linewidth=1.3)
            ax8.axvline(mass_ratio, color=AMBER, linewidth=0.8,
                        linestyle="--", alpha=0.8)
            ax8.scatter([mass_ratio], [dv / 1000], color=AMBER, s=60, zorder=5)
            ax8.text(mass_ratio + 0.3, dv / 1000 + 0.3,
                     f"Δv = {dv/1000:.2f} km/s\nm₀/m_f = {mass_ratio:.2f}",
                     color=AMBER, fontsize=7)

            # Reference lines
            for ref_dv, label in [(7.9, "LEO"), (11.2, "escape"), (5.6, "Mars TLI")]:
                ax8.axhline(ref_dv, color=AXIS_COLOR, linewidth=0.5,
                            linestyle=":", alpha=0.6)
                ax8.text(49, ref_dv + 0.1, label, color=TEXT_COLOR,
                         fontsize=6, ha="right", alpha=0.7)

            ax8.set_xlabel("Mass ratio  m₀/m_f", fontsize=8)
            ax8.set_ylabel("Δv  [km/s]", fontsize=8)
            ax8.set_xlim(1, 50)
            fig8.tight_layout(pad=1.0)
            st.pyplot(fig8, use_container_width=True)
            plt.close(fig8)

            # ── Δv budget comparison
            with st.expander("Mission Δv budget comparison"):
                missions_dv = {
                    "LEO orbit":     7.9,
                    "Earth escape":  11.2,
                    "Earth→Mars":    5.6,
                    "Earth→Moon":    3.2,
                    "Moon landing":  1.9,
                    "Your rocket":   dv / 1000,
                }
                fig9, ax9 = plt.subplots(figsize=(8, 3.0))
                apply_dark_style(fig9, ax9)
                names  = list(missions_dv.keys())
                values = list(missions_dv.values())
                colors_m = [AMBER if n == "Your rocket" else TEAL for n in names]
                bars9 = ax9.barh(names, values, color=colors_m, alpha=0.8,
                                  height=0.55, edgecolor=PLOT_BG, linewidth=0.5)
                ax9.set_xlabel("Δv  [km/s]", fontsize=8)
                ax9.tick_params(labelsize=7)
                for bar, val in zip(bars9, values):
                    ax9.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                             f"{val:.1f}", va="center", ha="left",
                             color=TEXT_COLOR, fontsize=7)
                fig9.tight_layout(pad=1.0)
                st.pyplot(fig9, use_container_width=True)
                plt.close(fig9)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 04 — ASTRONOMICAL UNIT CONVERTER
# ═══════════════════════════════════════════════════════════════════════════════

elif MODULE == "04 — Unit Converter":

    st.markdown("## Astronomical Unit Converter")
    st.markdown("""
<div class='status-bar'>IAU 2012 · BIPM · CODATA 2018 &nbsp;|&nbsp; DISTANCE · MASS · VELOCITY · TIME</div>
""", unsafe_allow_html=True)

    tab_dist, tab_mass, tab_scale, tab_viges = st.tabs([
        "Distance", "Mass & Velocity", "Universe Scale", "Vigesimal / Maya"
    ])

    # ── Distance tab
    with tab_dist:
        col_cv1, col_cv2 = st.columns([1, 2], gap="medium")

        with col_cv1:
            value    = st.number_input("Input value", value=1.0,
                                        format="%.6g", step=0.1)
            from_unit = st.selectbox(
                "From unit",
                ["Meters", "Kilometers", "Astronomical Units (AU)",
                 "Light-seconds", "Light-minutes", "Light-hours",
                 "Light-years", "Parsecs", "Kiloparsecs", "Megaparsecs"],
            )
            UNIT_TO_M = {
                "Meters":                     1.0,
                "Kilometers":                 1e3,
                "Astronomical Units (AU)":    AU,
                "Light-seconds":              C,
                "Light-minutes":              C * 60,
                "Light-hours":                C * 3600,
                "Light-years":                LIGHT_YEAR,
                "Parsecs":                    PARSEC,
                "Kiloparsecs":                PARSEC * 1e3,
                "Megaparsecs":                PARSEC * 1e6,
            }
            val_m = value * UNIT_TO_M[from_unit]

            st.markdown("""
<div class='formula-label'>Reference values</div>
<div class='formula-box'>
1 AU = 1.496×10¹¹ m<br>
1 ly = 9.461×10¹⁵ m<br>
1 pc = 3.086×10¹⁶ m<br>
c = 2.998×10⁸ m/s
</div>
""", unsafe_allow_html=True)

        with col_cv2:
            st.markdown("#### Conversion results")

            results = [
                ("Meters",                   val_m,                "m"),
                ("Kilometers",               val_m / 1e3,          "km"),
                ("Astronomical Units",        val_m / AU,           "AU"),
                ("Light-seconds",            val_m / C,            "ls"),
                ("Light-minutes",            val_m / (C * 60),     "lmin"),
                ("Light-hours",              val_m / (C * 3600),   "lhr"),
                ("Light-years",              val_m / LIGHT_YEAR,   "ly"),
                ("Parsecs",                  val_m / PARSEC,       "pc"),
                ("Kiloparsecs",              val_m / (PARSEC*1e3), "kpc"),
                ("Megaparsecs",              val_m / (PARSEC*1e6), "Mpc"),
            ]

            html_rows = "".join([
                f"<div class='result-row'>"
                f"<span class='result-label'>{name}</span>"
                f"<span class='result-value'>{fmt_sci(val)} {unit}</span>"
                f"</div>"
                for name, val, unit in results
            ])
            st.markdown(
                f"<div class='result-card'>{html_rows}</div>",
                unsafe_allow_html=True,
            )

            st.markdown("#### Context — how far is that?")
            contexts = [
                ("Earth–Moon",        3.844e8,  val_m),
                ("Earth–Sun (1 AU)",  AU,        val_m),
                ("Sun–Neptune",       30.07*AU,  val_m),
                ("To Proxima Cen.",   4.243*LIGHT_YEAR, val_m),
                ("Milky Way diameter",100e3*LIGHT_YEAR, val_m),
                ("To Andromeda",      2.537e6*LIGHT_YEAR, val_m),
            ]
            ctx_html = "".join([
                f"<div class='result-row'>"
                f"<span class='result-label'>{name}</span>"
                f"<span class='result-value'>{fmt_sci(val_m / ref)}×</span>"
                f"</div>"
                for name, ref, _ in contexts
            ])
            st.markdown(
                f"<div class='result-card'>{ctx_html}</div>",
                unsafe_allow_html=True,
            )

    # ── Mass & velocity tab
    with tab_mass:
        col_mv1, col_mv2 = st.columns([1, 1], gap="medium")

        with col_mv1:
            st.markdown("#### Mass converter")
            mass_val  = st.number_input("Mass value", value=1.0,
                                         format="%.4g", step=0.1, key="mv")
            mass_from = st.selectbox("From unit", [
                "Kilograms", "Solar masses (M☉)",
                "Earth masses (M⊕)", "Jupiter masses"
            ], key="mf")
            MASS_TO_KG = {
                "Kilograms":          1.0,
                "Solar masses (M☉)":  SOLAR_MASS,
                "Earth masses (M⊕)":  EARTH_MASS,
                "Jupiter masses":     1.898e27,
            }
            mass_kg = mass_val * MASS_TO_KG[mass_from]

            mass_results = [
                ("Kilograms",       mass_kg,                  "kg"),
                ("Solar masses",    mass_kg / SOLAR_MASS,     "M☉"),
                ("Earth masses",    mass_kg / EARTH_MASS,     "M⊕"),
                ("Jupiter masses",  mass_kg / 1.898e27,       "M♃"),
                ("Electron masses", mass_kg / 9.109e-31,      "mₑ"),
            ]
            html_m = "".join([
                f"<div class='result-row'>"
                f"<span class='result-label'>{n}</span>"
                f"<span class='result-value'>{fmt_sci(v)} {u}</span>"
                f"</div>"
                for n, v, u in mass_results
            ])
            st.markdown(f"<div class='result-card'>{html_m}</div>",
                        unsafe_allow_html=True)

        with col_mv2:
            st.markdown("#### Velocity converter")
            vel_val  = st.number_input("Velocity value", value=29.784,
                                        format="%.4g", step=0.1, key="vv")
            vel_from = st.selectbox("From unit", [
                "km/s", "m/s", "c (speed of light)", "AU/year", "km/h"
            ], key="vf")
            VEL_TO_MS = {
                "km/s":              1e3,
                "m/s":               1.0,
                "c (speed of light)": C,
                "AU/year":           AU / (365.25 * DAY),
                "km/h":              1 / 3.6,
            }
            vel_ms = vel_val * VEL_TO_MS[vel_from]

            vel_results = [
                ("m/s",             vel_ms,          "m/s"),
                ("km/s",            vel_ms / 1e3,    "km/s"),
                ("km/h",            vel_ms * 3.6,    "km/h"),
                ("AU/year",         vel_ms / (AU / (365.25 * DAY)), "AU/yr"),
                ("% speed of light",vel_ms / C * 100, "% c"),
            ]
            html_v = "".join([
                f"<div class='result-row'>"
                f"<span class='result-label'>{n}</span>"
                f"<span class='result-value'>{fmt_sci(v)} {u}</span>"
                f"</div>"
                for n, v, u in vel_results
            ])
            st.markdown(f"<div class='result-card'>{html_v}</div>",
                        unsafe_allow_html=True)

    # ── Universe scale tab
    with tab_scale:
        st.markdown("#### Orders of magnitude — from quarks to the observable universe")

        scale_objects = [
            ("Quark",               1e-18),
            ("Proton",              1e-15),
            ("Atom",                1e-10),
            ("Virus",               1e-7),
            ("Human",               1.7),
            ("Eiffel Tower",        324.0),
            ("Everest",             8848.0),
            ("Earth radius",        EARTH_RADIUS),
            ("Moon distance",       3.844e8),
            ("Sun radius",          6.957e8),
            ("1 AU",                AU),
            ("Jupiter orbit",       5.2 * AU),
            ("Neptune orbit",       30.1 * AU),
            ("Proxima Centauri",    4.243 * LIGHT_YEAR),
            ("Galactic centre",     2.7e4 * LIGHT_YEAR),
            ("Milky Way",           1e5 * LIGHT_YEAR),
            ("Andromeda galaxy",    2.537e6 * LIGHT_YEAR),
            ("Observable universe", 93e9 * LIGHT_YEAR),
        ]

        names  = [o[0] for o in scale_objects]
        sizes  = [math.log10(o[1]) for o in scale_objects]

        fig10, ax10 = plt.subplots(figsize=(10, 5))
        apply_dark_style(fig10, ax10)
        bar_colors = [TEAL] * len(names)
        highlight_idx = [4, 7, 10, 14, 17]  # human, earth, AU, MW, universe
        for i in highlight_idx:
            bar_colors[i] = AMBER

        bars10 = ax10.barh(names, sizes, color=bar_colors, alpha=0.75,
                            height=0.65, edgecolor=PLOT_BG, linewidth=0.4)

        for bar, (name, size) in zip(bars10, scale_objects):
            exp = int(round(math.log10(size)))
            ax10.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                      f"10^{exp} m", va="center", ha="left",
                      color=TEXT_COLOR, fontsize=6.5)

        ax10.set_xlabel("log₁₀(size)  [log meters]", fontsize=8)
        ax10.tick_params(axis="y", labelsize=7)
        ax10.tick_params(axis="x", labelsize=7)
        ax10.set_xlim(min(sizes) - 1, max(sizes) + 3.5)
        fig10.tight_layout(pad=1.2)
        st.pyplot(fig10, use_container_width=True)
        plt.close(fig10)

        span = max([o[1] for o in scale_objects]) / min([o[1] for o in scale_objects])
        st.markdown(f"""
<div style='font-size:0.72rem; color:#2a4060; letter-spacing:0.1em; padding: 8px 0;'>
Observable universe / quark ≈ 10^{int(round(math.log10(span)))} — the full dynamic range displayed above.
</div>
""", unsafe_allow_html=True)
# ── Vigesimal / Maya tab
    with tab_viges:
        from physics.vigesimal import (
            astronomical_distance_vigesimal,
            full_vigesimal_display,
            maya_long_count,
            solar_system_vigesimal_table,
            orbital_period_maya,
        )

        st.markdown("#### Vigesimal (base-20) — Maya astronomical notation")
        st.markdown("""
<div class='status-bar'>SISTEMA VIGESIMAL MAYA &nbsp;|&nbsp; BASE-20 &nbsp;|&nbsp; ETNO-MATEMÁTICA</div>
""", unsafe_allow_html=True)

        col_v1, col_v2 = st.columns([1, 2], gap="medium")

        with col_v1:
            vunit = st.selectbox("Input unit", [
                "km", "m", "light_seconds", "light_minutes"
            ], key="vunit")
            vval = st.number_input("Distance value", value=149597870.7,
                                    format="%.2f", key="vval")

            dist_m_v = vval * {
                "m": 1.0, "km": 1e3,
                "light_seconds": 2.998e8,
                "light_minutes": 2.998e8 * 60,
            }[vunit]

            st.markdown("""
<div class='formula-box'>
Maya digits: ● = 1 &nbsp; ▬ = 5 &nbsp; ∅ = 0<br>
Base: 20⁰=1, 20¹=20, 20²=400, 20³=8000<br>
Long Count: kin · winal · tun · katun · baktun
</div>
""", unsafe_allow_html=True)

        with col_v2:
            try:
                r = astronomical_distance_vigesimal(dist_m_v, unit="km")
                st.markdown("#### Result")
                st.code(r["full_display"], language=None)

                st.markdown("#### Maya digit glyphs")
                glyph_str = "   |   ".join(
                    f"{r['place_names'][i]} ({r['digits'][i]}): {r['glyph_digits'][i]}"
                    for i in range(len(r["digits"]))
                )
                st.code(glyph_str, language=None)

                st.markdown("#### Yucatec Maya names")
                st.markdown(" · ".join(r["maya_names"]))
            except Exception as e:
                st.error(f"Error: {e}")

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.markdown("#### Solar system distances in vigesimal")
        table = solar_system_vigesimal_table()
        for row in table:
            digits_str = "–".join(str(d) for d in row["vigesimal_digits"])
            st.markdown(
                f"`{row['name']:<10}` &nbsp; "
                f"{row['semi_major_au']:.3f} AU &nbsp;|&nbsp; "
                f"**{digits_str}**₂₀ &nbsp;|&nbsp; "
                f"{row['glyph_repr']}",
                unsafe_allow_html=True
            )

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.markdown("#### Orbital periods in Maya Long Count")
        from physics.constants import PLANETS
        for name, data in PLANETS.items():
            lc_str = orbital_period_maya(data["orbital_period_days"])
            st.markdown(f"`{name.capitalize():<10}` &nbsp; {lc_str}")
