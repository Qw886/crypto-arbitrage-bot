"""Explicitly configured rotating application logs, formatted in UTC."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from time import gmtime


def configure_logging(log_dir: Path, level: str = "INFO") -> logging.Logger:
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ValueError("unsupported log level")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("crypto_arbitrage_bot")
    logger.setLevel(level)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    handler = RotatingFileHandler(
        log_dir / "research.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(name)s %(message)s")
    formatter.converter = gmtime
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
