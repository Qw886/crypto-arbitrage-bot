"""Minimal offline startup: validate local configuration and display its scope."""

import json
from pathlib import Path

from crypto_arbitrage_bot.config.settings import load_settings


def run(config_path: Path | None = None, *, env_file: Path | None = None) -> int:
    """Load settings, print an offline status summary, then exit immediately."""
    settings = load_settings(config_path, env_file=env_file)
    summary = {
        "project": "crypto-arbitrage-bot",
        "mode": "research/read-only",
        "status": "local_configuration_valid",
        "symbols": list(settings.symbols),
        "network": "not implemented",
        "scanner": "not implemented",
        "fee_rates": {
            name: str(getattr(settings.fees, name))
            for name in ("spot_entry", "spot_exit", "perp_entry", "perp_exit")
        },
        "fee_assumption": "configured research values; not verified account fees",
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0
