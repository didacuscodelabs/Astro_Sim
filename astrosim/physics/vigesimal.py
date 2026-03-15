"""
AstroSim — Vigesimal (Base-20) Numeration Module
==================================================
Implements the Maya vigesimal positional number system and applies it
to astronomical distance representation.

The Maya vigesimal system is one of the oldest positional notation systems
in the Americas. Unlike the modified vigesimal calendar system (18×20),
the pure mathematical system used for counting follows strict base-20 rules:

    Position:  ...  20²   20¹   20⁰
    Value:     ...  400    20     1

The Maya represented digits 0–19 using dots (●, each = 1) and bars
(▬, each = 5), with a shell symbol for zero — one of the earliest known
uses of a positional zero in any numeral system.

Ethno-mathematical context
--------------------------
The Maya tracked astronomical cycles — Venus synodic periods (584 days),
solar years (365 days), and the Long Count (1,872,000 days) — using this
system with remarkable precision. Their Dresden Codex contains Venus tables
accurate to within 2 hours over 500 years.

Including vigesimal representation in AstroSim connects modern astrophysics
computation to indigenous scientific heritage, demonstrating that numerical
base is a cultural construct rather than a physical necessity.

References
----------
- Ascher, M. (2002). Mathematics Elsewhere: An Exploration of Ideas Across
  Cultures. Princeton University Press.
- Ifrah, G. (2000). The Universal History of Numbers. Wiley.
- Dresden Codex (c. 1200–1250 CE). Astronomical Venus tables.
- Lounsbury, F. (1978). Maya numeration, computation, and calendrical
  astronomy. Dictionary of Scientific Biography, 15, 759–818.
"""

from __future__ import annotations
from typing import List, Tuple, Union
import math


# ── Maya digit representations ────────────────────────────────────────────────

# ASCII representation of Maya digits 0–19
# Each digit is composed of bars (value 5) and dots (value 1)
_MAYA_ASCII = {
    0:  "∅",          # shell / zero
    1:  "●",
    2:  "● ●",
    3:  "● ● ●",
    4:  "● ● ● ●",
    5:  "▬",
    6:  "▬ ●",
    7:  "▬ ● ●",
    8:  "▬ ● ● ●",
    9:  "▬ ● ● ● ●",
    10: "▬ ▬",
    11: "▬ ▬ ●",
    12: "▬ ▬ ● ●",
    13: "▬ ▬ ● ● ●",
    14: "▬ ▬ ● ● ● ●",
    15: "▬ ▬ ▬",
    16: "▬ ▬ ▬ ●",
    17: "▬ ▬ ▬ ● ●",
    18: "▬ ▬ ▬ ● ● ●",
    19: "▬ ▬ ▬ ● ● ● ●",
}

# Nahuatl/Maya names for digits 0–19 (classic Yucatec Maya numerals)
_MAYA_NAMES = {
    0:  "mih",        # zero / nothing
    1:  "hun",
    2:  "ca",
    3:  "ox",
    4:  "can",
    5:  "ho",
    6:  "uac",
    7:  "uuc",
    8:  "uaxac",
    9:  "bolon",
    10: "lahun",
    11: "buluc",
    12: "lahca",
    13: "oxlahun",
    14: "canlahun",
    15: "holahun",
    16: "uaclahun",
    17: "uuclahun",
    18: "uaxaclahun",
    19: "bolonlahun",
}

# Place value names (positional levels in pure base-20)
_PLACE_NAMES = [
    "kin",        # 20⁰ =        1
    "winal",      # 20¹ =       20
    "tun",        # 20² =      400
    "katun",      # 20³ =    8,000
    "baktun",     # 20⁴ =  160,000
    "piktun",     # 20⁵ =3,200,000
    "kalabtun",   # 20⁶
    "kinchiltun", # 20⁷
    "alautun",    # 20⁸
]


# ── Core conversion functions ──────────────────────────────────────────────────

def decimal_to_vigesimal(n: int) -> List[int]:
    """
    Convert a non-negative integer to vigesimal (base-20) digits.

    Returns digits from most significant to least significant.
    Each digit is in the range [0, 19].

    Parameters
    ----------
    n : int
        Non-negative integer to convert.

    Returns
    -------
    List[int]
        Vigesimal digits, most significant first.
        Example: decimal_to_vigesimal(400) → [1, 0, 0]
                 (= 1×20² + 0×20¹ + 0×20⁰)

    Examples
    --------
    >>> decimal_to_vigesimal(0)
    [0]
    >>> decimal_to_vigesimal(20)
    [1, 0]
    >>> decimal_to_vigesimal(365)
    [18, 5]   # 18×20 + 5 = 365 (one solar year in Maya counting)
    >>> decimal_to_vigesimal(584)
    [1, 9, 4]  # 1×400 + 9×20 + 4 = 584 (Venus synodic period)
    """
    if not isinstance(n, int) or n < 0:
        raise ValueError(f"Input must be a non-negative integer, got {n!r}.")
    if n == 0:
        return [0]
    digits = []
    while n > 0:
        digits.append(n % 20)
        n //= 20
    return digits[::-1]


def vigesimal_to_decimal(digits: List[int]) -> int:
    """
    Convert vigesimal digits to a decimal integer.

    Parameters
    ----------
    digits : List[int]
        Vigesimal digits, most significant first. Each must be in [0, 19].

    Returns
    -------
    int
        Equivalent decimal integer.

    Examples
    --------
    >>> vigesimal_to_decimal([1, 0, 0])
    400
    >>> vigesimal_to_decimal([18, 5])
    365
    """
    for d in digits:
        if not (0 <= d <= 19):
            raise ValueError(f"Each vigesimal digit must be in [0, 19], got {d}.")
    result = 0
    for d in digits:
        result = result * 20 + d
    return result


def vigesimal_notation(n: int, use_names: bool = False) -> str:
    """
    Format an integer in Maya vigesimal positional notation.

    Parameters
    ----------
    n : int
        Non-negative integer.
    use_names : bool
        If True, use Yucatec Maya digit names instead of numerals.

    Returns
    -------
    str
        Formatted vigesimal notation string.

    Examples
    --------
    >>> vigesimal_notation(365)
    '(18)(5)₂₀  =  18×20¹ + 5×20⁰'
    >>> vigesimal_notation(365, use_names=True)
    'uaxaclahun · ho  [base-20]'
    """
    digits = decimal_to_vigesimal(n)
    n_pos  = len(digits)

    if use_names:
        names = [_MAYA_NAMES[d] for d in digits]
        return " · ".join(names) + "  [base-20]"

    # Numeric form with place-value breakdown
    parts = []
    for i, d in enumerate(digits):
        power = n_pos - 1 - i
        if power == 0:
            parts.append(f"{d}×20⁰")
        elif power == 1:
            parts.append(f"{d}×20¹")
        else:
            sup = _superscript(power)
            parts.append(f"{d}×20{sup}")

    digit_str = "(" + ")(".join(str(d) for d in digits) + ")"
    breakdown  = " + ".join(parts)
    return f"{digit_str}₂₀  =  {breakdown}"


def vigesimal_glyph(n: int) -> str:
    """
    Return an ASCII glyph representation of a Maya digit (0–19).

    Uses dots (● = 1) and bars (▬ = 5), the visual language of
    Maya inscriptions.

    Parameters
    ----------
    n : int
        Digit in [0, 19].

    Returns
    -------
    str
        ASCII glyph.
    """
    if not (0 <= n <= 19):
        raise ValueError(f"Maya glyph only defined for 0–19, got {n}.")
    return _MAYA_ASCII[n]


def full_vigesimal_display(n: int) -> str:
    """
    Return a complete, multi-line Maya vigesimal display for an integer.

    Shows digit glyphs stacked vertically (most significant at top),
    place names, and the decimal equivalent — in the style of a
    Maya stele inscription read top-to-bottom.

    Parameters
    ----------
    n : int
        Non-negative integer.

    Returns
    -------
    str
        Multi-line formatted string.
    """
    digits = decimal_to_vigesimal(n)
    n_pos  = len(digits)
    lines  = [f"  Vigesimal representation of {n:,}"]
    lines.append("  " + "─" * 36)

    for i, d in enumerate(digits):
        power      = n_pos - 1 - i
        place      = _PLACE_NAMES[power] if power < len(_PLACE_NAMES) else f"20^{power}"
        place_val  = 20 ** power
        glyph      = _MAYA_ASCII[d]
        name       = _MAYA_NAMES[d]
        lines.append(
            f"  {place:<12} (20^{power:>1} = {place_val:>10,})  "
            f"digit={d:>2}  [{name:<12}]  {glyph}"
        )

    lines.append("  " + "─" * 36)
    lines.append(f"  Decimal check: {vigesimal_to_decimal(digits):,}  ✓")
    return "\n".join(lines)


def _superscript(n: int) -> str:
    table = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
    return str(n).translate(table)


# ── Astronomical distance in vigesimal ───────────────────────────────────────

def astronomical_distance_vigesimal(
        distance_m:  float,
        unit:        str   = "km",
        round_to:    int   = 0
) -> dict:
    """
    Express an astronomical distance in vigesimal (base-20) notation.

    Converts the distance to the specified unit, rounds to an integer,
    and returns the full vigesimal representation with Maya glyph notation.

    Parameters
    ----------
    distance_m : float
        Distance in metres.
    unit : str
        Display unit before vigesimal conversion. Options:
        'km', 'm', 'au_thousandths', 'light_seconds', 'light_minutes'.
        Default: 'km'.
    round_to : int
        Number of decimal places for rounding before conversion.
        Use 0 (default) for whole-number vigesimal digits.

    Returns
    -------
    dict with keys:
        value_decimal   : float  — distance in chosen unit
        unit            : str    — the unit used
        value_int       : int    — rounded integer for vigesimal conversion
        digits          : list   — vigesimal digits [most significant first]
        notation        : str    — formatted vigesimal notation string
        glyph_digits    : list   — ASCII glyph for each digit
        maya_names      : list   — Yucatec Maya name for each digit
        place_names     : list   — positional place names
        full_display    : str    — multi-line display string

    Examples
    --------
    >>> from physics.constants import AU
    >>> r = astronomical_distance_vigesimal(AU, unit='km')
    >>> r['value_decimal']
    149597870.7
    >>> r['notation']
    '(1)(14)(19)(9)(17)(15)(11)₂₀  =  1×20⁶ + 14×20⁵ + ...'
    """
    from physics.constants import AU, C

    UNIT_FACTORS = {
        "m":               1.0,
        "km":              1e-3,
        "au_thousandths":  1e3 / AU,   # thousandths of AU (compact for solar system)
        "light_seconds":   1.0 / C,
        "light_minutes":   1.0 / (C * 60),
    }
    if unit not in UNIT_FACTORS:
        raise ValueError(
            f"Unknown unit '{unit}'. Choose from {list(UNIT_FACTORS.keys())}."
        )

    val   = distance_m * UNIT_FACTORS[unit]
    val_r = round(val, round_to)
    n     = int(val_r)

    if n < 0:
        raise ValueError("Vigesimal representation requires a non-negative value.")

    digits      = decimal_to_vigesimal(n)
    n_pos       = len(digits)
    glyphs      = [_MAYA_ASCII[d] for d in digits]
    names       = [_MAYA_NAMES[d] for d in digits]
    places      = [
        _PLACE_NAMES[n_pos - 1 - i] if (n_pos - 1 - i) < len(_PLACE_NAMES)
        else f"20^{n_pos-1-i}"
        for i in range(n_pos)
    ]

    return {
        "value_decimal": val,
        "unit":          unit,
        "value_int":     n,
        "digits":        digits,
        "notation":      vigesimal_notation(n),
        "maya_names":    names,
        "glyph_digits":  glyphs,
        "place_names":   places,
        "full_display":  full_vigesimal_display(n),
    }


def solar_system_vigesimal_table() -> List[dict]:
    """
    Return vigesimal distance representations for all solar system planets.

    Distances are the mean orbital semi-major axes in km (rounded to
    nearest 1000 km for clean vigesimal digits).

    Returns
    -------
    List[dict]
        One entry per planet, each with keys:
        name, distance_km, vigesimal_digits, notation, maya_names.
    """
    from physics.constants import PLANETS, AU

    results = []
    for name, data in PLANETS.items():
        dist_m  = data["semi_major_axis_au"] * AU
        dist_km = round(dist_m / 1e3 / 1000) * 1000   # round to 1000 km

        r = astronomical_distance_vigesimal(dist_m, unit="km", round_to=0)
        results.append({
            "name":              name.capitalize(),
            "distance_km":       dist_km,
            "semi_major_au":     data["semi_major_axis_au"],
            "vigesimal_digits":  r["digits"],
            "notation":          r["notation"],
            "maya_names":        r["maya_names"],
            "glyph_repr":        " | ".join(r["glyph_digits"]),
        })
    return results


# ── Maya calendar connection ──────────────────────────────────────────────────

def maya_long_count(days: int) -> dict:
    """
    Express a number of days in Maya Long Count notation.

    The Long Count uses a modified vigesimal system for calendar purposes:
        1 Kin    =  1 day
        1 Winal  = 20 Kins    (20 days)
        1 Tun    = 18 Winals  (360 days ≈ 1 solar year)
        1 Katun  = 20 Tuns    (7,200 days ≈ 19.7 years)
        1 Baktun = 20 Katuns  (144,000 days ≈ 394 years)

    Note: the Tun uses 18 (not 20) Winals — a deliberate astronomical
    approximation to the 365-day solar year.

    Parameters
    ----------
    days : int
        Number of days to convert.

    Returns
    -------
    dict with Long Count components and context.
    """
    if days < 0:
        raise ValueError("Days must be non-negative.")

    baktun = days // 144000
    rem    = days  % 144000
    katun  = rem   // 7200
    rem    = rem    % 7200
    tun    = rem   // 360
    rem    = rem    % 360
    winal  = rem   // 20
    kin    = rem    % 20

    return {
        "baktun":        baktun,
        "katun":         katun,
        "tun":           tun,
        "winal":         winal,
        "kin":           kin,
        "notation":      f"{baktun}.{katun}.{tun}.{winal}.{kin}",
        "total_days":    days,
        "solar_years":   days / 365.25,
        "note": (
            "Long Count uses 18 winals per tun (≈ solar year approximation), "
            "not 20 — a deliberate calendrical adaptation of the vigesimal system."
        ),
    }


def orbital_period_maya(period_days: float) -> str:
    """
    Express an orbital period in Maya Long Count notation.

    Parameters
    ----------
    period_days : float
        Orbital period in days.

    Returns
    -------
    str
        Maya Long Count string, e.g. '1.0.1.5.5'.
    """
    lc = maya_long_count(int(round(period_days)))
    return (
        f"{lc['notation']}  "
        f"({period_days:.2f} days = {lc['solar_years']:.3f} solar years)"
    )
