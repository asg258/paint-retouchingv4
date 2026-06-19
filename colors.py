"""
colors.py — Valspar / Sherwin-Williams official color database.

Loads the official Valspar color dataset (1,596 colors with full RGB values)
and provides fast lookups by color code, color name, or hex value.

Source file: colors_valspar.csv
Data:  Valspar BulkPaint Lowe's Digital Data 2025

Usage examples:
    from colors import get_color, search_colors, list_families

    # Exact lookup by color code
    color = get_color("4002-9A")

    # Exact lookup by name (case-insensitive)
    color = get_color("Purple Mist")

    # Fuzzy search when you're not sure of the exact name
    results = search_colors("dusty blue")

    # Browse all colors in a family
    grays = list_colors(family="Grays")

Every lookup returns a ColorEntry (or None if not found):
    color.name      → "Purple Mist"
    color.code      → "4002-9A"
    color.rgb       → (227, 218, 232)
    color.hex       → "#E3DAE8"
    color.family    → "Purples"
    color.lrv       → 72        (Light Reflectance Value, 0–100)
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Iterator


# Path to the bundled CSV — lives next to this file.
_CSV_PATH = Path(__file__).parent / "colors_valspar.csv"

# Column names exactly as they appear in the CSV header.
_COL_CODE  = "Color Number"
_COL_NAME  = "Color Name"
_COL_R     = "RGB-1 Red Color Value"
_COL_G     = "RGB-2 Green Color Value"
_COL_B     = "RGB-3 Blue Color Value"
_COL_HEX   = "Hex Value"
_COL_LRV   = "LRV Notation Number"
_COL_FAMILY = "CBG NA Color Family Name"
_COL_COLLECTION = "Color Collection"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ColorEntry:
    """One paint color from the official Valspar dataset."""
    name:       str              # e.g. "Purple Mist"
    code:       str              # e.g. "4002-9A"
    rgb:        tuple[int, int, int]   # (R, G, B), each 0–255
    hex:        str              # e.g. "#E3DAE8"
    family:     str              # e.g. "Purples"
    lrv:        float            # Light Reflectance Value 0–100
    collection: str              # e.g. "Valspar Reserve"

    def __str__(self) -> str:
        r, g, b = self.rgb
        return (
            f"{self.code}  {self.name!r}  "
            f"RGB({r},{g},{b})  #{self.hex}  "
            f"{self.family}  LRV={self.lrv}"
        )


# ---------------------------------------------------------------------------
# Internal database (loaded once at import time)
# ---------------------------------------------------------------------------

class _ColorDatabase:
    """
    In-memory index of all colors, built once when colors.py is first imported.

    Three indexes are kept for O(1) lookup:
        _by_code  — color code → ColorEntry
        _by_name  — lowercase color name → ColorEntry
        _by_hex   — uppercase hex (no #) → ColorEntry

    A sorted list of all names is also kept for fuzzy matching.
    """

    def __init__(self, csv_path: Path) -> None:
        self._by_code:  dict[str, ColorEntry] = {}
        self._by_name:  dict[str, ColorEntry] = {}
        self._by_hex:   dict[str, ColorEntry] = {}
        self._all_names: list[str] = []          # lowercase, for fuzzy search
        self._all:       list[ColorEntry] = []   # every color in load order

        self._load(csv_path)

    def _load(self, csv_path: Path) -> None:
        """Parse the CSV and populate all indexes."""
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Color database not found: {csv_path}\n"
                "Make sure colors_valspar.csv is in the same folder as colors.py."
            )

        skipped = 0
        with open(csv_path, encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                code = row[_COL_CODE].strip()
                name = row[_COL_NAME].strip()

                # Skip blank / header-repeat rows.
                if not code or not name:
                    skipped += 1
                    continue

                # Skip rows without RGB values — we can't recolor without them.
                r_raw = row[_COL_R].strip()
                g_raw = row[_COL_G].strip()
                b_raw = row[_COL_B].strip()
                if not r_raw or not g_raw or not b_raw:
                    skipped += 1
                    continue

                try:
                    rgb = (int(r_raw), int(g_raw), int(b_raw))
                except ValueError:
                    skipped += 1
                    continue

                # Hex — normalise to uppercase without #.
                raw_hex = row[_COL_HEX].strip().lstrip("#").upper()

                # LRV — may be empty or non-numeric.
                try:
                    lrv = float(row[_COL_LRV].strip())
                except (ValueError, KeyError):
                    lrv = 0.0

                entry = ColorEntry(
                    name=name,
                    code=code,
                    rgb=rgb,
                    hex=raw_hex,
                    family=row[_COL_FAMILY].strip(),
                    lrv=lrv,
                    collection=row[_COL_COLLECTION].strip(),
                )

                # Register in all three indexes.
                self._by_code[code.upper()] = entry
                self._by_name[name.lower()] = entry
                if raw_hex:
                    self._by_hex[raw_hex] = entry

                self._all.append(entry)
                self._all_names.append(name.lower())

        print(
            f"[colors] Loaded {len(self._all)} colors "
            f"({skipped} rows skipped — blank or missing RGB)."
        )

    # ------------------------------------------------------------------ #
    # Lookup methods                                                       #
    # ------------------------------------------------------------------ #

    def by_code(self, code: str) -> ColorEntry | None:
        """Look up by exact color code, e.g. '4002-9A'."""
        return self._by_code.get(code.strip().upper())

    def by_name(self, name: str) -> ColorEntry | None:
        """Look up by exact color name (case-insensitive), e.g. 'Purple Mist'."""
        return self._by_name.get(name.strip().lower())

    def by_hex(self, hex_value: str) -> ColorEntry | None:
        """Look up by hex value, with or without #, e.g. 'E3DAE8'."""
        return self._by_hex.get(hex_value.strip().lstrip("#").upper())

    def fuzzy(self, query: str, n: int = 5, cutoff: float = 0.4) -> list[ColorEntry]:
        """
        Return up to n colors whose names are closest to the query string.

        Uses Python's difflib SequenceMatcher — good for typos and partial
        words, not full semantic search. For best results, use a distinctive
        part of the name (e.g. "dusty blue" rather than just "blue").

        Args:
            query:   Free-text search string.
            n:       Maximum number of results to return.
            cutoff:  Similarity threshold 0–1. Lower = more results, less precise.

        Returns:
            List of ColorEntry sorted by similarity (best match first).
        """
        q = query.strip().lower()
        matches = get_close_matches(q, self._all_names, n=n, cutoff=cutoff)
        return [self._by_name[m] for m in matches]

    def by_family(self, family: str) -> list[ColorEntry]:
        """Return all colors belonging to the given family (case-insensitive)."""
        f = family.strip().lower()
        return [c for c in self._all if c.family.lower() == f]

    def families(self) -> list[str]:
        """Return sorted list of all unique color family names."""
        return sorted({c.family for c in self._all if c.family})

    def all_colors(self) -> list[ColorEntry]:
        """Return every color in the database."""
        return list(self._all)

    def __len__(self) -> int:
        return len(self._all)


# Load the database once when this module is imported.
_DB = _ColorDatabase(_CSV_PATH)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_color(query: str) -> ColorEntry | None:
    """
    Look up a color by code, name, or hex. Returns None if not found.

    Try in this order:
        1. Exact color code match   (e.g. "4002-9A")
        2. Exact name match         (case-insensitive, e.g. "Purple Mist")
        3. Exact hex match          (e.g. "#E3DAE8" or "E3DAE8")

    For approximate / fuzzy matching use search_colors() instead.

    Examples:
        get_color("4002-9A")       → ColorEntry for Purple Mist
        get_color("purple mist")   → same
        get_color("#E3DAE8")       → same
    """
    result = (
        _DB.by_code(query)
        or _DB.by_name(query)
        or _DB.by_hex(query)
    )
    return result


def search_colors(query: str, n: int = 5, cutoff: float = 0.35) -> list[ColorEntry]:
    """
    Fuzzy search by color name. Good for partial names or unsure spelling.

    Returns up to n best matches sorted by similarity.

    Examples:
        search_colors("dusty blue")    → closest blue color names
        search_colors("warm beige")    → closest neutral/brown names
        search_colors("knitting")      → finds "Knitting Needles" etc.
    """
    return _DB.fuzzy(query, n=n, cutoff=cutoff)


def list_colors(family: str | None = None) -> list[ColorEntry]:
    """
    Return all colors, optionally filtered by family name.

    Args:
        family: One of the values returned by list_families().
                Pass None to get every color.

    Examples:
        list_colors()              → all 1596 colors
        list_colors("Grays")       → only gray colors
        list_colors("Neutrals")    → only neutral colors
    """
    if family is None:
        return _DB.all_colors()
    return _DB.by_family(family)


def list_families() -> list[str]:
    """
    Return all color family names available in the database.

    Example output:
        ['Blacks', 'Blues', 'Browns', 'Grays', 'Greens',
         'Neutrals', 'Oranges', 'Pinks', 'Purples',
         'Reds', 'Teals', 'Whites', 'Yellows']
    """
    return _DB.families()


def total_colors() -> int:
    """Return the total number of colors with valid RGB data."""
    return len(_DB)


# ---------------------------------------------------------------------------
# Quick test / interactive lookup — python colors.py [query]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print(f"Database: {total_colors()} colors loaded")
    print(f"Families: {list_families()}\n")

    if len(sys.argv) < 2:
        # No argument — show a few colors from each family as a sample.
        for fam in list_families():
            sample = list_colors(fam)[:3]
            print(f"  {fam} ({len(list_colors(fam))} colors):")
            for c in sample:
                print(f"    {c}")
        print("\nUsage: python colors.py <color name or code>")
        sys.exit(0)

    query = " ".join(sys.argv[1:])
    print(f"Looking up: '{query}'\n")

    # Exact match first.
    exact = get_color(query)
    if exact:
        print(f"Exact match found:")
        print(f"  {exact}")
    else:
        print("No exact match found.")

    # Always show fuzzy results too — helpful for nearby suggestions.
    fuzzy = search_colors(query, n=5)
    if fuzzy:
        print(f"\nClosest matches:")
        for c in fuzzy:
            print(f"  {c}")
