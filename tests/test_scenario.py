"""Offline exit hypotheses reuse the tested execution and cash-flow engine."""

from dataclasses import replace
from datetime import UTC, timedelta, timezone
from decimal import ROUND_DOWN, Decimal, Inexact, localcontext
from unittest.mock import Mock

import pytest
from test_offline_estimate import CLOSED, OPENED, SETTLED, assembly_for, book, books, config

from crypto_arbitrage_bot.arbitrage import scenario as engine
from crypto_arbitrage_bot.arbitrage.estimate import estimate_once
from crypto_arbitrage_bot.arbitrage.fees import FeeRates, FourLegFees
from crypto_arbitrage_bot.arbitrage.funding import FundingEvent
from crypto_arbitrage_bot.arbitrage.profit import (
    decomposed_net_profit,
    net_return,
    round_trip_from_fills,
)
from crypto_arbitrage_bot.market_data.snapshot import assemble_snapshot
from crypto_arbitrage_bot.models.estimate import EstimateStatus
from crypto_arbitrage_bot.models.market import Market
from crypto_arbitrage_bot.models.scenario import (
    ExitBasisAssumption,
    ScenarioDefinition,
    ScenarioType,
)

D = Decimal


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("scenarios must remain entirely offline"))
    monkeypatch.setattr("socket.create_connection", forbidden)
    monkeypatch.setattr("http.client.HTTPSConnection", forbidden)


def event(rate="0.01", settled_at=SETTLED, event_id="settlement", symbol="BTCUSDT"):
    return FundingEvent(event_id, settled_at, D("110"), D(rate), symbol)


def baseline(source_books, events=(), closed_at=CLOSED, name="Unchanged books"):
    return engine.reverse_execution_scenario(
        name=name, spot_entry_book=source_books[0], perp_entry_book=source_books[2],
        opened_at=OPENED, closed_at=closed_at, funding_events=events,
        description="Synthetic unchanged exit books; no automatic basis convergence.",
    )


def target(source_books, kind=ScenarioType.CONVERGENCE, pct="0.02", events=(), **changes):
    assumption = ExitBasisAssumption(D("100"), D(pct), source_books[0], source_books[2])
    values = dict(
        scenario_type=kind, name=kind.value, assumption=assumption,
        opened_at=OPENED, closed_at=CLOSED, funding_events=events,
        description="Explicit hypothetical exit mid and depth; not a price forecast.",
    )
    values.update(changes)
    return engine.target_basis_scenario(**values)


def evaluate(definition, source_books=None, **changes):
    source_books = books() if source_books is None else source_books
    return engine.evaluate_scenario(
        assembly_for((source_books[0], source_books[2])), config(**changes), definition,
    )


@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT"])
def test_reverse_execution_baseline_has_no_convergence_profit(symbol):
    source_books = books(symbol)
    definition = baseline(source_books)
    result = evaluate(definition, source_books)
    assert result.complete and result.scenario_type is ScenarioType.NO_CONVERGENCE
    assert result.name == "Unchanged books"
    assert result.definition is definition and result.assumptions == definition.exit_scenario.description
    assert result.entry_mid_basis == result.assumed_exit_mid_basis == D("10")
    assert result.estimate.mid_basis_pnl == D("0")
    assert result.entry_basis == D("6") and result.assumed_exit_basis == D("12")
    assert result.price_pnl == D("-12")  # Reverse execution pays both spreads and depth impact.
    assert result.funding_total == D("0") and result.matched_quantity == D("2")
    assert result.spot_fees == D(".600") and result.perp_fees == D(".2856")
    assert result.other_costs == D(".1") and result.net_profit == D("-12.9856")
    assert result.capital_occupied == D("420.9856")
    assert result.net_return == net_return(D("-12.9856"), D("420.9856"))


def test_explicit_convergence_matches_hand_calculation_and_existing_cashflow():
    """q=2: 198-204+216-206 +2.2 -.204-.396-.108-.1648 -.1 =5.2272."""
    source_books = books()
    definition = target(source_books, events=(event(),))
    result = evaluate(definition, source_books)
    assert result.entry_mid_basis == D("10") and result.assumed_exit_mid_basis == D("2")
    assert result.entry_basis == D("6") and result.assumed_exit_basis == D("4")
    assert result.price_pnl == D("4") and result.funding_total == D("2.2")
    assert result.estimate.fees == FourLegFees(D(".204"), D(".396"), D(".108"), D(".1648"))
    assert result.spot_fees == D(".600") and result.perp_fees == D(".2728")
    assert result.other_costs == D(".1") and result.net_profit == D("5.2272")
    assert result.capital_occupied == D("420.9728")
    assert result.net_return == net_return(D("5.2272"), D("420.9728"))
    estimate = result.estimate
    assert estimate.calculation == round_trip_from_fills(
        *estimate.fills, estimate.fees, D("2.2"), D(".1"),
    )
    assert result.net_profit == decomposed_net_profit(
        estimate.decomposition, D("2.2"), estimate.fees, D(".1"),
    )
    assert estimate.decomposition.mid_basis_pnl == D("16")
    assert estimate.decomposition.spread_costs == D("8")
    assert estimate.decomposition.slippage_costs == D("4")


def test_adverse_widening_can_lose_despite_positive_funding():
    definition = target(books(), ScenarioType.ADVERSE, "0.20", (event(),))
    result = evaluate(definition)
    assert result.complete and result.scenario_type is ScenarioType.ADVERSE
    assert result.entry_mid_basis == D("10") and result.assumed_exit_mid_basis == D("20")
    assert result.price_pnl == D("-32") and result.funding_total == D("2.2")
    assert result.net_profit == D("-30.8016") and result.net_return < D("0")


def test_explicit_unchanged_target_agrees_with_reverse_execution():
    source_books = books()
    explicit = evaluate(target(source_books, ScenarioType.NO_CONVERGENCE, "0.10"))
    reverse = evaluate(baseline(source_books))
    assert explicit.estimate.calculation == reverse.estimate.calculation
    assert explicit.net_return == reverse.net_return
    assert explicit.estimate.mid_basis_pnl == D("0")


def test_target_ratio_uses_explicit_exit_spot_mid_and_moves_both_books():
    source_books = books()
    assumption = ExitBasisAssumption(D("200"), D(".02"), source_books[0], source_books[2])
    definition = target(source_books, assumption=assumption)
    spot = definition.exit_scenario.spot_exit_book
    perp = definition.exit_scenario.perp_exit_book
    assert (spot.best_bid, spot.best_ask) == (D("199"), D("201"))
    assert (perp.best_bid, perp.best_ask) == (D("203"), D("205"))
    result = evaluate(definition)
    assert result.assumed_exit_mid_basis == D("4")  # .02 * EXIT spot mid 200.
    assert result.entry_basis == result.assumed_exit_basis == D("6")
    assert result.price_pnl == D("0") and result.funding_total == D("0")
    assert result.net_profit == D("-1.536")  # All four fees and other costs, once.


@pytest.mark.parametrize("rate,expected", [("0.01", "2.2"), ("-0.01", "-2.2"), ("0", "0")])
def test_each_scenario_uses_explicit_signed_settlements_once(rate, expected):
    source_books = books()
    for kind, pct in ((ScenarioType.NO_CONVERGENCE, "0.10"),
                      (ScenarioType.CONVERGENCE, "0.02"), (ScenarioType.ADVERSE, "0.20")):
        with_funding = evaluate(target(source_books, kind, pct, (event(rate),)))
        without_funding = evaluate(target(source_books, kind, pct))
        assert with_funding.funding_total == D(expected)
        assert with_funding.net_profit - without_funding.net_profit == D(expected)
        assert with_funding.spot_fees == without_funding.spot_fees
        assert with_funding.perp_fees == without_funding.perp_fees
        assert with_funding.capital_occupied == without_funding.capital_occupied


@pytest.mark.parametrize("settled_at,expected", [
    (OPENED - timedelta(microseconds=1), "0"), (OPENED, "0"),
    (OPENED + timedelta(microseconds=1), "2.2"), (CLOSED, "2.2"),
    (CLOSED + timedelta(microseconds=1), "0"),
])
def test_only_opened_exclusive_closed_inclusive_settlements_are_counted(settled_at, expected):
    result = evaluate(baseline(books(), (event(settled_at=settled_at),)))
    assert result.funding_total == D(expected)


def test_long_window_without_settlements_never_extrapolates_observed_funding():
    result = evaluate(baseline(books(), closed_at=OPENED + timedelta(days=365)))
    assert result.estimate.funding_rate == D("0.01")
    assert result.funding_total == D("0") and result.net_profit == D("-12.9856")


def test_multiple_events_accumulate_signed_actual_notionals_and_ignore_other_symbol():
    events = (
        event("0.01"),
        replace(event("-0.005", CLOSED, "second"), settlement_mark_price=D("120")),
        event("1", CLOSED, "eth-only", "ETHUSDT"),
    )
    result = evaluate(baseline(books(), events))
    assert result.funding_total == D("1.0")  # 2*110*.01 - 2*120*.005.
    assert result.net_profit == D("-11.9856")


@pytest.mark.parametrize("duplicate", [
    event(settled_at=CLOSED), event(event_id="another-id"),
])
def test_duplicate_settlement_id_or_timestamp_is_rejected(duplicate):
    with pytest.raises(ValueError, match="duplicate funding"):
        evaluate(baseline(books(), (event(), duplicate)))


def test_fees_spread_and_slippage_are_not_charged_twice():
    definition = target(books(), events=(event(),))
    charged = evaluate(definition)
    free = evaluate(definition, fee_rates=FeeRates(*(D("0") for _ in range(4))))
    assert free.net_profit - charged.net_profit == charged.estimate.fees.total == D(".8728")
    assert free.net_profit == D("6.1")  # Execution PnL 4 + funding 2.2 - other .1.
    assert charged.net_profit == D("5.2272")  # No second 8+4 spread/slippage deduction.


def test_result_is_direct_projection_of_single_existing_estimate_call(monkeypatch):
    source_books = books()
    assembly = assembly_for((source_books[0], source_books[2]))
    settings, definition = config(), target(source_books, events=(event(),))
    expected = estimate_once(assembly, settings, definition.exit_scenario)
    delegate = Mock(return_value=expected)
    monkeypatch.setattr(engine, "estimate_once", delegate)
    result = engine.evaluate_scenario(assembly, settings, definition, evaluated_at=OPENED)
    delegate.assert_called_once_with(assembly, settings, definition.exit_scenario, evaluated_at=OPENED)
    assert result.estimate is expected and result.definition is definition
    assert result.net_profit is expected.net_profit
    assert result.net_return is expected.net_return
    assert result.spot_fees is expected.calculation.spot_fees
    assert result.perp_fees is expected.calculation.perp_fees
    assert result.entry_basis is expected.calculation.entry_basis
    assert result.assumed_exit_basis is expected.calculation.exit_basis


def test_recurring_vwap_preserves_decimal_reconciliation_and_caller_context():
    source_books = books(recurring=True)
    with localcontext() as context:
        context.prec = 4
        context.rounding = ROUND_DOWN
        context.traps[Inexact] = True
        previous_flags = context.flags.copy()
        definition = target(source_books, events=(event(),))
        result = evaluate(definition, source_books, target_quantity=D("3"))
        assert context.prec == 4 and context.rounding == ROUND_DOWN
        assert context.traps[Inexact] and context.flags == previous_flags
    assert result.estimate.spot_entry_vwap == D("101." + "6" * 46 + "7")
    assert result.estimate.perp_entry_vwap == D("108." + "3" * 47)
    assert result.price_pnl == D("9.5") and result.funding_total == D("3.3")
    assert result.net_profit == D("11.3925")
    assert result.net_profit == decomposed_net_profit(
        result.estimate.decomposition, result.funding_total, result.estimate.fees, D(".1"),
    )
    assert result.estimate.calculation == round_trip_from_fills(
        *result.estimate.fills, result.estimate.fees, result.funding_total, D(".1"),
    )
    assert all(isinstance(value, Decimal) for value in (
        result.entry_basis, result.assumed_exit_basis, result.entry_mid_basis,
        result.assumed_exit_mid_basis, result.matched_quantity, result.price_pnl,
        result.funding_total, result.spot_fees, result.perp_fees, result.other_costs,
        result.net_profit, result.capital_occupied, result.net_return,
    ))


@pytest.mark.parametrize("kind", list(ScenarioType))
def test_scenario_label_never_invents_target_prices_or_funding(kind):
    original = baseline(books())
    definition = ScenarioDefinition(kind, "Explicit unchanged prices", original.exit_scenario)
    result = evaluate(definition)
    assert result.scenario_type is kind
    assert result.entry_mid_basis == result.assumed_exit_mid_basis == D("10")
    assert result.funding_total == D("0") and result.net_profit == D("-12.9856")


def test_batch_preserves_explicit_order_and_independent_results():
    source_books = books()
    assembly = assembly_for((source_books[0], source_books[2]))
    definitions = (
        baseline(source_books), target(source_books, events=(event(),)),
        target(source_books, ScenarioType.ADVERSE, "0.20", (event(),)),
    )
    results = engine.evaluate_scenarios(assembly, config(), definitions)
    assert tuple(item.definition for item in results) == definitions
    assert tuple(item.net_profit for item in results) == (D("-12.9856"), D("5.2272"), D("-30.8016"))
    assert all(item.estimate.assembly is assembly for item in results)
    assert engine.evaluate_scenarios(assembly, config(), ()) == ()


def test_batch_duplicate_names_rejected_before_evaluation(monkeypatch):
    definition = baseline(books())
    forbidden = Mock(side_effect=AssertionError("duplicate comparison must not run"))
    monkeypatch.setattr(engine, "evaluate_scenario", forbidden)
    with pytest.raises(ValueError, match="unique"):
        engine.evaluate_scenarios(None, config(), (definition, definition))
    forbidden.assert_not_called()


@pytest.mark.parametrize("invalid", [False, True], ids=["incomplete", "stale-invalid"])
def test_incomplete_or_invalid_market_data_cannot_produce_scenario_profit(invalid):
    source_books = books()
    assembly = assembly_for((source_books[0], source_books[2]))
    when = OPENED + timedelta(minutes=1) if invalid else OPENED
    if not invalid:
        assembly = assemble_snapshot(
            assembly.symbol, replace(assembly.sources, spot_depth=None), assembled_at=OPENED,
        )
    result = engine.evaluate_scenario(assembly, config(), target(source_books), evaluated_at=when)
    assert not result.complete
    expected = EstimateStatus.INVALID_MARKET_DATA if invalid else EstimateStatus.INCOMPLETE_SNAPSHOT
    assert result.estimate.status is expected
    assert result.net_profit is result.net_return is result.price_pnl is None
    assert result.funding_total is result.spot_fees is result.perp_fees is None
    assert result.entry_basis is result.assumed_exit_basis is None
    assert result.entry_mid_basis is result.assumed_exit_mid_basis is None


@pytest.mark.parametrize("leg", ["spot_entry", "perp_entry", "spot_exit", "perp_exit"])
def test_insufficient_depth_is_never_extrapolated_or_reported_as_full_fill(leg):
    source_books = list(books())
    spot_template, perp_template = source_books[0], source_books[2]
    if leg == "spot_entry":
        source_books[0] = replace(source_books[0], asks=source_books[0].asks[:1])
    elif leg == "perp_entry":
        source_books[2] = replace(source_books[2], bids=source_books[2].bids[:1])
    elif leg == "spot_exit":
        spot_template = replace(spot_template, bids=(replace(spot_template.bids[0], quantity=D("1")),))
    else:
        perp_template = replace(perp_template, asks=(replace(perp_template.asks[0], quantity=D("1")),))
    definition = target(source_books, assumption=ExitBasisAssumption(
        D("100"), D(".02"), spot_template, perp_template,
    ))
    result = evaluate(definition, source_books)
    assert result.estimate.status is EstimateStatus.INSUFFICIENT_DEPTH
    assert not result.complete and result.net_profit is result.net_return is None
    incomplete = [fill for fill in result.estimate.fills if not fill.complete]
    assert len(incomplete) == 1 and incomplete[0].filled_quantity == D("1")
    assert incomplete[0].vwap is None


def test_translation_preserves_depth_quantities_spread_spacing_and_original_provenance():
    source_books = books()
    original = tuple(source_books)
    local_zone = timezone(timedelta(hours=8))
    definition = target(
        source_books, opened_at=OPENED.astimezone(local_zone), closed_at=CLOSED.astimezone(local_zone),
    )
    assert definition.exit_scenario.opened_at == OPENED
    assert definition.exit_scenario.closed_at == CLOSED
    assert definition.exit_scenario.closed_at.tzinfo is UTC
    assert definition.basis_assumption.spot_depth_template is source_books[0]
    assert definition.basis_assumption.perp_depth_template is source_books[2]
    for template, translated, offset in (
        (source_books[0], definition.exit_scenario.spot_exit_book, D("0")),
        (source_books[2], definition.exit_scenario.perp_exit_book, D("-8")),
    ):
        assert translated.symbol == template.symbol and translated.market is template.market
        assert translated.observed_at == CLOSED and translated.event_time is None
        assert translated.best_ask - translated.best_bid == template.best_ask - template.best_bid
        for side in ("bids", "asks"):
            old, new = getattr(template, side), getattr(translated, side)
            assert len(old) == len(new)
            assert tuple(level.quantity for level in old) == tuple(level.quantity for level in new)
            assert all(after.price == before.price + offset for before, after in zip(old, new, strict=True))
    assert tuple(source_books) == original
    assert source_books[2].event_time == OPENED
    reverse = baseline(source_books)
    assert reverse.exit_scenario.spot_exit_book is source_books[0]
    assert reverse.exit_scenario.perp_exit_book is source_books[2]
    assert reverse.exit_scenario.perp_exit_book.event_time == OPENED


@pytest.mark.parametrize("field,value,error", [
    ("spot_exit_mid", 100.0, TypeError), ("exit_basis_pct", 0.02, TypeError),
    ("spot_exit_mid", D("0"), ValueError), ("spot_exit_mid", D("-1"), ValueError),
    ("exit_basis_pct", D("-1"), ValueError), ("exit_basis_pct", D("NaN"), ValueError),
])
def test_invalid_or_float_targets_fail_before_evaluation(field, value, error):
    source_books = books()
    values = dict(spot_exit_mid=D("100"), exit_basis_pct=D(".02"),
                  spot_depth_template=source_books[0], perp_depth_template=source_books[2])
    values[field] = value
    with pytest.raises(error):
        ExitBasisAssumption(**values)


def test_nonpositive_translated_level_is_rejected_without_clamping():
    source_books = books()
    assumption = ExitBasisAssumption(D(".5"), D(".02"), source_books[0], source_books[2])
    with pytest.raises(ValueError, match="price"):
        target(source_books, assumption=assumption)


@pytest.mark.parametrize("problem", ["missing-side", "crossed", "unsorted", "symbol", "market"])
def test_invalid_depth_template_is_rejected(problem):
    spot, _, perp, _ = books()
    if problem == "missing-side":
        spot = replace(spot, bids=())
    elif problem == "crossed":
        spot = book(Market.SPOT, (("102", "3"),), (("101", "3"),))
    elif problem == "unsorted":
        perp = replace(perp, bids=perp.bids[::-1])
    elif problem == "symbol":
        perp = replace(perp, symbol="ETHUSDT")
    else:
        spot = replace(spot, market=Market.USDTM)
    with pytest.raises(ValueError):
        ExitBasisAssumption(D("100"), D(".02"), spot, perp)
