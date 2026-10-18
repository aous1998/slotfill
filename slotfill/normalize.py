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

MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}

MIN_YEAR, MAX_YEAR = 2000, 2030


def _year(y: str) -> int:
    n = int(y)
    return 2000 + n if len(y) == 2 else n


def _make(y: int, m: int, d: int) -> str | None:
    if not (MIN_YEAR <= y <= MAX_YEAR):
        return None
    try:
        return dt.date(y, m, d).isoformat()
    except ValueError:
        return None


def normalize_date(raw: str) -> str | None:
    """Return YYYY-MM-DD, or None if the label cannot be read unambiguously."""
    s = raw.strip().strip("()").upper()

    if m := re.fullmatch(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s):  # 2018-03-23
        return _make(int(m[1]), int(m[2]), int(m[3]))
    if re.fullmatch(r"\d{8}", s):  # 20180304 (YYYYMMDD), else 25032018 (DDMMYYYY)
        return _make(int(s[:4]), int(s[4:6]), int(s[6:])) or _make(int(s[4:]), int(s[2:4]), int(s[:2]))
    if m := re.fullmatch(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{2}|\d{4})", s):  # 25/12/2018, day first
        a, b, y = int(m[1]), int(m[2]), _year(m[3])
        # Day-first; fall back to month-first only when day-first is impossible (12/28/2017).
        return _make(y, b, a) or (_make(y, a, b) if b > 12 else None)
    if m := re.fullmatch(r"(\d{1,2})[-/ ]([A-Z]{3})[A-Z]*[-/ ](\d{2}|\d{4})", s):  # 05 MAR 2018
        mo = MONTHS.get(m[2])
        return _make(_year(m[3]), mo, int(m[1])) if mo else None
    if m := re.fullmatch(r"([A-Z]{3})[A-Z]* (\d{1,2}),? (\d{4})", s):  # OCT 3, 2016
        mo = MONTHS.get(m[1])
        return _make(int(m[3]), mo, int(m[2])) if mo else None
    return None


def normalize_total(raw: str) -> str | None:
    """'RM 1,007.5' -> '1007.50'. Negative or unparsable totals return None."""
    s = re.sub(r"^(RM|MYR|\$)\s*", "", raw.strip().upper()).replace(",", "")
    if not re.fullmatch(r"\d+(\.\d+)?", s):
        return None
    try:
        return str(Decimal(s).quantize(Decimal("0.01")))
    except InvalidOperation:
        return None


def normalize_text(raw: str | None) -> str | None:
    """Whitespace-collapse a free-text field. Used for gold targets."""
    if raw is None:
        return None
    s = re.sub(r"\s+", " ", raw).strip()
    return s or None


def text_key(raw: str | None) -> str:
    """Comparison key for free text: case-, whitespace- and edge-punctuation-insensitive."""
    if raw is None:
        return ""
    s = re.sub(r"\s+", " ", str(raw)).strip().upper()
    s = re.sub(r"\s*([,.:;&()/-])\s*", r"\1", s)  # "JALAN SAGU 18 , TAMAN" == "JALAN SAGU 18,TAMAN"
    return s.strip(" .,;:")
