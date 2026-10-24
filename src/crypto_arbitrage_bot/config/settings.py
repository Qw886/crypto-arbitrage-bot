"""Validated, non-sensitive offline settings. All fee values are configurable."""

import tomllib
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from crypto_arbitrage_bot.arbitrage.fees import FeeRates
from crypto_arbitrage_bot.config.research import ResearchSettings
from crypto_arbitrage_bot.config.testnet import TestnetSettings
from crypto_arbitrage_bot.utils.decimal_math import require_decimal


class FeeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    spot_entry: Decimal
    spot_exit: Decimal
    perp_entry: Decimal
    perp_exit: Decimal

    @field_validator("spot_entry", "spot_exit", "perp_entry", "perp_exit", mode="before")
    @classmethod
    def exact_rate_input(cls, value: Any) -> Any:
        if not isinstance(value, (str, Decimal)):
            raise ValueError("fee rates must be Decimal or decimal strings, never floats")
        return value

    @field_validator("spot_entry", "spot_exit", "perp_entry", "perp_exit")
    @classmethod
    def rate_range(cls, value: Decimal) -> Decimal:
        require_decimal(value, "fee rate", nonnegative=True)
        if value >= 1:
            raise ValueError("fee rates are fractions in [0, 1)")
        return value

    def as_rates(self) -> FeeRates:
        return FeeRates(self.spot_entry, self.spot_exit, self.perp_entry, self.perp_exit)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CAB_",
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
    )

    symbols: tuple[Literal["BTCUSDT", "ETHUSDT"], ...]
    poll_interval_seconds: int
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    log_dir: Path
    fees: FeeSettings
    research: ResearchSettings | None = None
    testnet: TestnetSettings | None = None

    @field_validator("symbols")
    @classmethod
    def nonempty_distinct_symbols(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(value) != len(set(value)):
            raise ValueError("symbols must be nonempty and unique")
        return value

    @field_validator("poll_interval_seconds")
    @classmethod
    def polling_bounds(cls, value: int) -> int:
        if not 5 <= value <= 30:
            raise ValueError("V1 polling interval must be 5 to 30 seconds")
        return value

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Environment > optional .env > TOML. Do not load the unused file source.
        return env_settings, dotenv_settings, init_settings


def load_settings(path: Path | None = None, *, env_file: Path | None = None) -> Settings:
    """Load explicit decimal strings from TOML; .env is opt-in via its path."""
    config_path = path or Path(__file__).with_name("defaults.toml")
    with config_path.open("rb") as source:
        data = tomllib.load(source)
    return Settings(_env_file=env_file, **data)
