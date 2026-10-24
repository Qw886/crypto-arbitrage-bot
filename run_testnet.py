"""Local launch helper for a src-layout checkout without installing the package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from crypto_arbitrage_bot.execution.main import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
