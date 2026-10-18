"""The strict output schema and the pass/fail validity check.

A model output is *schema-valid* only if the raw text is a single JSON object
(no prose, no markdown fences) that passes `Receipt.model_validate`. This is the
same bar an accounting import would apply, so there is no partial credit.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"
TOTAL_PATTERN = r"^\d+\.\d{2}$"


class Receipt(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    company: str = Field(min_length=1, description="Seller's registered business name.")
    date: str = Field(pattern=DATE_PATTERN, description="Transaction date, ISO 8601 (YYYY-MM-DD).")
    address: str | None = Field(description="Seller's address on one line, or null if absent.")
    total: str = Field(pattern=TOTAL_PATTERN, description="Grand total, digits with 2 decimals, no currency.")

    @field_validator("date")
    @classmethod
    def _real_calendar_date(cls, v: str) -> str:
        dt.date.fromisoformat(v)  # rejects 2018-02-31 etc.
        return v

    @field_validator("company", "address")
    @classmethod
    def _no_blank(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("must not be blank")
        return v


FIELDS = ("company", "date", "address", "total")

# JSON Schema handed to providers that support constrained decoding. Kept
# deliberately simple (no regex) so every provider accepts it; the patterns are
# enforced afterwards by `validate_output`.
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "company": {"type": "string"},
        "date": {"type": "string"},
        "address": {"type": ["string", "null"]},
        "total": {"type": "string"},
    },
    "required": list(FIELDS),
    "additionalProperties": False,
}


@dataclass
class Validation:
    valid: bool
    data: dict | None  # parsed object if JSON parsed, even when schema failed
    error: str | None  # "json" | "not_object" | "schema" | None


def validate_output(text: str | None) -> Validation:
    if text is None:
        return Validation(False, None, "empty")
    try:
        obj = json.loads(text.strip())
    except json.JSONDecodeError:
        return Validation(False, None, "json")
    if not isinstance(obj, dict):
        return Validation(False, None, "not_object")
    try:
        Receipt.model_validate(obj)
    except ValidationError:
        return Validation(False, obj, "schema")
    return Validation(True, obj, None)
