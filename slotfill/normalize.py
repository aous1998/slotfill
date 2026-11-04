"""Turn SROIE's verbatim labels into the canonical form the schema expects.

SROIE annotators copied values exactly as printed ("25/12/2018", "RM 29.70",
"05 MAR 2018"). The target system wants ISO dates and bare 2-decimal totals, so
every gold label goes through these functions once, and the same text
normalisation is applied to predictions before comparing.

Malaysian receipts print dates day-first, so ambiguous dates are read as DD/MM.
"""

from __future__ import annotations

import datetime as dt
import re
from decimal import Decimal, InvalidOperation

# Three-letter month abbreviation -> month number: {"JAN": 1, ..., "DEC": 12}.
MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}

# Plausible year range for these receipts. Anything outside is treated as a misread.
MIN_YEAR, MAX_YEAR = 2000, 2030


def _year(y: str) -> int:
    # Two-digit years ("18") mean 2000s ("2018"); four-digit years are used as is.
    n = int(y)
    return 2000 + n if len(y) == 2 else n


def _make(y: int, m: int, d: int) -> str | None:
    # Build an ISO date string, or None if the year is implausible or the date doesn't
    # exist (e.g. 31 February). Returning None lets callers try another reading.
    if not (MIN_YEAR <= y <= MAX_YEAR):
        return None
    try:
        return dt.date(y, m, d).isoformat()
    except ValueError:
        return None


def normalize_date(raw: str) -> str | None:
    """Return YYYY-MM-DD, or None if the label cannot be read unambiguously."""
    # Remove surrounding spaces and brackets, and upper-case so "Mar" == "MAR".
    s = raw.strip().strip("()").upper()

    # Each branch below handles one printed date style, tried in order.
    if m := re.fullmatch(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s):  # 2018-03-23
        return _make(int(m[1]), int(m[2]), int(m[3]))
    if re.fullmatch(r"\d{8}", s):  # 20180304 (YYYYMMDD), else 25032018 (DDMMYYYY)
        return _make(int(s[:4]), int(s[4:6]), int(s[6:])) or _make(int(s[4:]), int(s[2:4]), int(s[:2]))
    if m := re.fullmatch(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{2}|\d{4})", s):  # 25/12/2018, day first
        a, b, y = int(m[1]), int(m[2]), _year(m[3])
        # Day-first; fall back to month-first only when day-first is impossible (12/28/2017).
        return _make(y, b, a) or (_make(y, a, b) if b > 12 else None)
    if m := re.fullmatch(r"(\d{1,2})[-/ ]([A-Z]{3})[A-Z]*[-/ ](\d{2}|\d{4})", s):  # 05 MAR 2018
        mo = MONTHS.get(m[2])  # only the first 3 letters matter, so "MARCH" works too
        return _make(_year(m[3]), mo, int(m[1])) if mo else None
    if m := re.fullmatch(r"([A-Z]{3})[A-Z]* (\d{1,2}),? (\d{4})", s):  # OCT 3, 2016
        mo = MONTHS.get(m[1])
        return _make(int(m[3]), mo, int(m[2])) if mo else None
    # No known format matched: refuse to guess.
    return None


def normalize_total(raw: str) -> str | None:
    """'RM 1,007.5' -> '1007.50'. Negative or unparsable totals return None."""
    # Strip a leading currency (RM = Malaysian ringgit) and thousands separators.
    s = re.sub(r"^(RM|MYR|\$)\s*", "", raw.strip().upper()).replace(",", "")
    # What's left must be a plain non-negative number like "1007" or "1007.5".
    if not re.fullmatch(r"\d+(\.\d+)?", s):
        return None
    try:
        # Decimal (not float) avoids rounding surprises; quantize forces exactly 2 decimals.
        return str(Decimal(s).quantize(Decimal("0.01")))
    except InvalidOperation:
        return None


def normalize_text(raw: str | None) -> str | None:
    """Whitespace-collapse a free-text field. Used for gold targets."""
    if raw is None:
        return None
    # Any run of spaces/tabs/newlines becomes a single space.
    s = re.sub(r"\s+", " ", raw).strip()
    return s or None  # an all-whitespace value counts as missing


def text_key(raw: str | None) -> str:
    """Comparison key for free text: case-, whitespace- and edge-punctuation-insensitive."""
    # This is used only for *comparing* values (scoring, vendor grouping), never as output.
    if raw is None:
        return ""
    s = re.sub(r"\s+", " ", str(raw)).strip().upper()
    s = re.sub(r"\s*([,.:;&()/-])\s*", r"\1", s)  # "JALAN SAGU 18 , TAMAN" == "JALAN SAGU 18,TAMAN"
    # Ignore stray punctuation at either end, e.g. a trailing full stop.
    return s.strip(" .,;:")
