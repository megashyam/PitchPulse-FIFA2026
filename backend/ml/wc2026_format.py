"""
FIFA World Cup 2026 knockout format (Regulations, matches 73-104).

Static; verified against the realised 2026 bracket (tests/test_wc2026_data.py).

Slot codes: "1A" winner of group A, "2B" runner-up of B, "3:ABCDF" the
third-placed team from one of those groups, resolved per combination of
qualifying thirds by FIFA's Annex C table
(data/wc2026/third_place_table.json).
"""

from __future__ import annotations

GROUP_LETTERS = "ABCDEFGHIJKL"

R32_SLOTS: dict[int, tuple[str, str]] = {
    73: ("2A", "2B"),
    74: ("1E", "3:ABCDF"),
    75: ("1F", "2C"),
    76: ("1C", "2F"),
    77: ("1I", "3:CDFGH"),
    78: ("2E", "2I"),
    79: ("1A", "3:CEFHI"),
    80: ("1L", "3:EHIJK"),
    81: ("1D", "3:BEFIJ"),
    82: ("1G", "3:AEHIJ"),
    83: ("2K", "2L"),
    84: ("1H", "2J"),
    85: ("1B", "3:EFGIJ"),
    86: ("1J", "2H"),
    87: ("1K", "3:DEIJL"),
    88: ("2D", "2G"),
}

# match → (feeder, feeder); winners advance. 103 is the third-place match.
KO_TREE: dict[int, tuple[int, int]] = {
    89: (74, 77),
    90: (73, 75),
    91: (76, 78),
    92: (79, 80),
    93: (83, 84),
    94: (81, 82),
    95: (86, 88),
    96: (85, 87),
    97: (89, 90),
    98: (93, 94),
    99: (91, 92),
    100: (95, 96),
    101: (97, 98),
    102: (99, 100),
    104: (101, 102),
}
FINAL = 104


def _leaves(m: int) -> list[int]:
    if m in R32_SLOTS:
        return [m]
    a, b = KO_TREE[m]
    return _leaves(a) + _leaves(b)


# R32 matches in bracket order: adjacent winners meet in every later round.
R32_ORDER: list[int] = _leaves(FINAL)

# Column order of Annex C: which group winner each qualifying third meets.
THIRD_SLOT_WINNERS = ["1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L"]


def third_slot_of(winner: str) -> tuple[int, str]:
    """"1A" → (79, "CEFHI"): the R32 match and eligible third-place groups."""
    for m, (a, b) in R32_SLOTS.items():
        if a == winner and b.startswith("3:"):
            return m, b[2:]
    raise KeyError(winner)
