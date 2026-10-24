"""Explicit research asset identity, never an execution symbol authorization."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchInstrument:
    symbol: str
    base_asset: str

    def __post_init__(self) -> None:
        if (type(self.base_asset) is not str or not re.fullmatch(r"[A-Z0-9]{1,32}", self.base_asset)
                or type(self.symbol) is not str or self.symbol != self.base_asset + "USDT"):
            raise ValueError("research instrument requires one explicit USDT/base identity")
        if re.match(r"^(?:(?:10|100|1000|10000|100000|1000000)[A-Z]|[1-9][0-9]*M[A-Z])", self.base_asset):
            raise ValueError("scaled asset units require separate verification")

    def require_symbol(self, symbol: str) -> None:
        if type(symbol) is not str or symbol != self.symbol:
            raise ValueError("research instrument/symbol mismatch")
