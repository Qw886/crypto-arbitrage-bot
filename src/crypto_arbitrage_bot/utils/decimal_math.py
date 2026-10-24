"""Strict numeric and time primitives shared by the offline research model."""

from collections.abc import Callable
from datetime import UTC, datetime
from decimal import (
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DivisionByZero,
    FloatOperation,
    InvalidOperation,
    Overflow,
    localcontext,
)
from functools import wraps
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")
DECIMAL_PRECISION = 50


def decimal_calculation(function: Callable[P, R]) -> Callable[P, R]:
    """Make calculations independent of the caller's decimal context."""
    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        context = Context(
            prec=DECIMAL_PRECISION,
            rounding=ROUND_HALF_EVEN,
            traps=[InvalidOperation, DivisionByZero, Overflow, FloatOperation],
        )
        with localcontext(context):
            return function(*args, **kwargs)
    return wrapped


def require_decimal(
    value: Decimal,
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> Decimal:
    if not isinstance(value, Decimal):
        raise TypeError(f"{name} must be Decimal constructed from a string")
    if not value.is_finite():
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def utc_timestamp(value: datetime, name: str = "timestamp") -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)
