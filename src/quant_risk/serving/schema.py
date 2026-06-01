"""Serving-edge schema (Pydantic v2): the contract's twin at the API boundary.

Same fields/ranges/domains as RAW_SCHEMA, but enforced per-request with helpful
422 errors. Domains are imported from schema.py so the two can't disagree.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from quant_risk.schema import GRADE_VALUES, HOME_VALUES, PURPOSE_VALUES, TERM_VALUES


class LoanApplication(BaseModel):
    model_config = {"extra": "forbid"}   # reject unknown fields -> catches schema drift early

    loan_amnt: float = Field(ge=500, le=40_000)
    term: Literal[tuple(TERM_VALUES)]            # type: ignore[valid-type]
    int_rate: float = Field(ge=0, le=35)
    installment: float = Field(gt=0)
    grade: Literal[tuple(GRADE_VALUES)]          # type: ignore[valid-type]
    emp_length: str | None = None
    home_ownership: Literal[tuple(HOME_VALUES)]  # type: ignore[valid-type]
    annual_inc: float = Field(ge=0, le=5_000_000)
    purpose: Literal[tuple(PURPOSE_VALUES)]      # type: ignore[valid-type]
    dti: float | None = Field(default=None, ge=0, le=100)
    delinq_2yrs: int = Field(ge=0)
    open_acc: int = Field(ge=0)
    revol_util: float | None = Field(default=None, ge=0, le=200)
    fico_range_low: int = Field(ge=300, le=850)
    fico_range_high: int = Field(ge=300, le=850)

    def to_record(self) -> dict:
        d = self.model_dump()
        d["loan_status"] = "Current"   # unused by features; keeps the frame shape consistent
        return d


class ScoreResponse(BaseModel):
    pd: float = Field(description="probability of default in [0,1]")
    risk_band: Literal["LOW", "MEDIUM", "HIGH"]
    model_version: str
    schema_version: str


class BatchScoreRequest(BaseModel):
    applications: list[LoanApplication] = Field(min_length=1, max_length=10_000)