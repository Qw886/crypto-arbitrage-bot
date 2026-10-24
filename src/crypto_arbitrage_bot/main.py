"""Offline configuration check or an explicitly selected public research run."""

import argparse
import sys
from pathlib import Path

from crypto_arbitrage_bot.app import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="crypto-arbitrage-bot: research/read-only, offline configuration check"
    )
    parser.add_argument("--config", type=Path, help="optional local TOML configuration")
    parser.add_argument("--env-file", type=Path, help="optional non-sensitive local .env file")
    parser.add_argument("--version", action="version", version="crypto-arbitrage-bot 0.1.0")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="one public read-only research round")
    mode.add_argument("--forever", action="store_true", help="poll public research until interrupted")
    args = parser.parse_args(argv)
    try:
        if args.once or args.forever:
            from crypto_arbitrage_bot.config.settings import load_settings
            from crypto_arbitrage_bot.runner import CycleStatus, create_runner
            from crypto_arbitrage_bot.storage.opportunity_records import exact_json

            runner = create_runner(load_settings(args.config, env_file=args.env_file))
            if args.forever:
                runner.run_forever()
                return 0
            cycles = runner.run_once()
            print(exact_json(cycles))
            return 0 if all(cycle.status is CycleStatus.SAVED for cycle in cycles) else 1
        return run(args.config, env_file=args.env_file)
    except (OSError, ValueError) as error:
        print(f"Local configuration error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
