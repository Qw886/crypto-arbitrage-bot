"""Metadata-only testnet HTTP/exit timing; no URL queries, keys or bodies logged."""

import argparse
import os
from contextlib import contextmanager
from pathlib import Path
from time import monotonic_ns
from urllib.parse import urlsplit

from restart_acceptance import save
from testnet_continuous_acceptance import run_acceptance, safe_error

from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.execution.credential_store import (
    DEFAULT_CREDENTIAL_FILE,
    execution_environment,
)
from crypto_arbitrage_bot.execution.runner import ExecutionGuardError, create_testnet_bot
from crypto_arbitrage_bot.execution.testnet_client import _BASE_URLS, _ROUTES
from crypto_arbitrage_bot.market_data.quality import validate_market_data
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.logging import configure_logging


class Profile:
    def __init__(self, *, read_only=False):
        self.read_only, self.phase = read_only, "OTHER"
        self.requests, self.stages = [], []
        self.dropped_requests = 0

    @contextmanager
    def scope(self, name):
        prior, started = self.phase, monotonic_ns()
        record = {"stage": name, "started_at": utc_now()}
        self.phase = name
        try:
            yield
        except Exception as error:
            record["error_type"] = type(error).__name__
            raise
        finally:
            self.phase = prior
            record["elapsed_ms"] = (monotonic_ns() - started) // 1_000_000
            self.stages.append(record)


class TimedOpener:
    def __init__(self, opener, profile):
        self.opener, self.profile = opener, profile

    def open(self, request, data=None, timeout=None):
        parsed, method = urlsplit(request.full_url), request.get_method()
        allowed = {(urlsplit(_BASE_URLS[market]).hostname, verb, path)
                   for (market, _), (verb, path) in _ROUTES.items()}
        if (parsed.scheme != "https" or parsed.username or parsed.password
                or (parsed.hostname, method, parsed.path) not in allowed
                or self.profile.read_only and method != "GET"):
            raise ExecutionGuardError("TIMING_PROFILE_BLOCKED_ROUTE")
        record = {"stage": self.profile.phase, "host": parsed.hostname,
                  "path": parsed.path, "method": method, "started_at": utc_now()}
        if len(self.profile.requests) < 500:
            self.profile.requests.append(record)
        else:
            self.profile.dropped_requests += 1
        started = monotonic_ns()
        try:
            response = (self.opener.open(request, timeout=timeout) if data is None
                        else self.opener.open(request, data=data, timeout=timeout))
            record["http_status"] = getattr(response, "status", 200)
            headers = getattr(response, "headers", None) or {}
            encoding = headers.get("Content-Encoding", "identity").lower()
            record["content_encoding"] = encoding if encoding in ("identity", "gzip", "deflate", "br") else "other"
            return TimedResponse(response, record)
        except Exception as error:
            record["error_type"] = type(error).__name__
            raise
        finally:
            record["open_ms"] = (monotonic_ns() - started) // 1_000_000


class TimedResponse:
    def __init__(self, response, record):
        self.response, self.record = response, record

    def __getattr__(self, name):
        return getattr(self.response, name)

    def __enter__(self):
        self.response.__enter__()
        return self

    def __exit__(self, *args):
        return self.response.__exit__(*args)

    def read(self, limit=-1):
        started = monotonic_ns()
        try:
            body = self.response.read(limit)
            self.record["body_bytes"] = self.record.get("body_bytes", 0) + len(body)
            return body
        except Exception as error:
            self.record["read_error_type"] = type(error).__name__
            raise
        finally:
            self.record["read_ms"] = self.record.get("read_ms", 0) + (monotonic_ns() - started) // 1_000_000


def attach(bot, profile):
    for client in (bot.spot, bot.perp):
        client._opener = TimedOpener(client._opener, profile)
    def wrap(owner, attribute, phase):
        original = getattr(owner, attribute)
        def timed(*args, **kwargs):
            with profile.scope(phase):
                return original(*args, **kwargs)
        setattr(owner, attribute, timed)
    wrap(bot, "_plan", "ENTRY_PREFLIGHT")
    wrap(bot, "_close", "EXIT_PREFLIGHT")
    wrap(bot.coordinator, "open_pair", "ENTRY_SUBMISSION")
    wrap(bot.coordinator, "close_pair", "EXIT_SUBMISSION")


def preflight(bot, profile):
    with profile.scope("EXIT_ACCOUNTS"):
        bot.spot.get_account_state("ETHUSDT")
        bot.perp.get_account_state("ETHUSDT")
    with profile.scope("EXIT_RULES"):
        bot.spot.get_symbol_rules("ETHUSDT")
        bot.perp.get_symbol_rules("ETHUSDT")
    with profile.scope("EXIT_MARKET_DATA"):
        assembly = bot.collector.collect("ETHUSDT")
        result = validate_market_data(assembly, bot.research.data_quality, evaluated_at=bot.clock())
    return {"status": "PREFLIGHT_MEASURED", "data_quality": result.status,
            "quality_reason_codes": result.reason_codes, "orders_submitted": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("docs/testnet-funding-acceptance.toml"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute-sample", action="store_true")
    args = parser.parse_args()
    profile = Profile(read_only=args.preflight_only)
    outcome = {"started_at": utc_now(), "mode": "READ_ONLY" if args.preflight_only else "VIRTUAL_SAMPLE"}
    result = 1
    try:
        if args.report.exists():
            raise ExecutionGuardError("TIMING_PROFILE_REPORT_ALREADY_EXISTS")
        if args.execute_sample and os.environ.get("CAB_TESTNET_ORDERS_ENABLED") != "true":
            raise ExecutionGuardError("TESTNET_ORDERS_DISABLED")
        settings = load_settings(args.config)
        configure_logging(settings.log_dir / args.report.stem, settings.log_level)
        bot = create_testnet_bot(settings, environ=execution_environment(DEFAULT_CREDENTIAL_FILE, os.environ))
        attach(bot, profile)
        if args.preflight_only:
            outcome.update(preflight(bot, profile))
            result = 0
        else:
            case = args.report.with_suffix(".case.json")
            result = run_acceptance(bot, case, duration_seconds=120)
            outcome.update(status="PASS" if result == 0 else "NEEDS_ATTENTION", case_report=str(case))
    except Exception as error:
        outcome.update(status="FAILED", **safe_error(error))
    finally:
        outcome.update(finished_at=utc_now(), requests=profile.requests, stages=profile.stages,
                       dropped_requests=profile.dropped_requests)
        # Never overwrite an earlier profile to conceal a failure or repeat entry.
        target = args.report.with_suffix(".failure.json") if args.report.exists() else args.report
        save(target, outcome)
        print(exact_json(outcome), flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
