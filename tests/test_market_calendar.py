"""Task 1.10 — the IDX trading calendar and the three WIB signal windows.

Three signals a day, at three fixed IDX moments: ten minutes after the open,
ten minutes after the lunch break resumes, and ten minutes before the close.
Nothing else in this repository knows what day it is — no timezone, no holiday,
no notion that Friday's two sessions are not Monday's — so a scheduler built on
today's code would fire on Christmas and would fire ``post_lunch`` at the wrong
instant every Friday.

These tests are the specification for the calendar that closes that gap.  They
are **offline** (``holidays`` ships its Indonesian data as Python, no network at
import or at call time) and they are **date-independent**: every date below is
pinned literally, so the suite cannot start failing in January.

Two things are pinned that a thinner suite would miss.

*Friday is a different shape, not a shifted clock.*  Friday session 1 ends at
11:30 where Monday–Thursday ends at 12:00, and Friday session 2 opens at 14:00
where Monday–Thursday opens at 13:30.  ``post_lunch`` therefore moves by forty
minutes (13:40 against 14:10) while ``open`` and ``pre_close`` do not move at
all — the asymmetry is the behaviour, and both directions are asserted.

*``is_trading_day`` is not inverted.*  "Return ``False`` for any date not
positively known to be a trading day" taken literally would make every future
weekday ``False``, because no holiday list can enumerate the future — and the
scheduler would then never fire, silently, which is the failure this whole plan
exists to remove.  The predicate returns ``True`` for a weekday that is not in
the known-closure set, and that is asserted directly over a whole future year.
The holiday list's incompleteness is handled by *warning*, never by refusing to
trade: an unlisted closure is a false positive (a signal fires on a closed day)
rather than a false negative that stops all signalling.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

import market_calendar
from market_calendar import (
    FRIDAY_SESSIONS,
    IDX_EXCHANGE_HOLIDAYS,
    MON_THU_SESSIONS,
    WIB,
    SessionWindow,
    is_trading_day,
    signal_windows,
)

# Pinned dates.  ``2026-10-13`` is a Tuesday, ``2026-10-16`` the Friday of the
# same week, ``2026-10-17`` the Saturday after it, ``2026-10-12`` that Monday.
TUESDAY = date(2026, 10, 13)
WEDNESDAY = date(2026, 10, 14)
THURSDAY = date(2026, 10, 15)
FRIDAY = date(2026, 10, 16)
SATURDAY = date(2026, 10, 17)
MONDAY = date(2026, 10, 12)

MON_THU = (MONDAY, TUESDAY, WEDNESDAY, THURSDAY)

# 2026 closures taken from Indonesia's decree-backed holiday set as shipped by
# ``holidays`` 0.106, cross-checked against IDX's published 2026 monthly trading
# day counts (see ``market_calendar``'s COVERAGE note).
PUBLIC_HOLIDAYS_2026 = (
    date(2026, 1, 1),    # Tahun Baru Masehi
    date(2026, 1, 16),   # Isra Mikraj Nabi Muhammad
    date(2026, 2, 16),   # Cuti Bersama Tahun Baru Imlek
    date(2026, 3, 19),   # Hari Suci Nyepi
    date(2026, 5, 1),    # Hari Buruh Internasional
    date(2026, 12, 25),  # Hari Raya Natal
)

NORMAL_WEEKDAYS_2026 = (
    date(2026, 1, 2),    # the day after New Year — a trading day, not a closure
    date(2026, 4, 1),
    date(2026, 7, 15),
    date(2026, 11, 30),
)


def _weekdays(year: int) -> list[date]:
    """Every Monday–Friday of ``year``, computed, not listed."""
    day, out = date(year, 1, 1), []
    while day.year == year:
        if day.weekday() < 5:
            out.append(day)
        day += timedelta(days=1)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# THE BRIEF'S THREE TESTS — verbatim
# ═══════════════════════════════════════════════════════════════════════════════

def test_monday_thursday_lunch_window():
    w = dict(signal_windows(date(2026, 10, 13)))  # a Tuesday
    assert w["open"].strftime("%H:%M") == "09:10"
    assert w["post_lunch"].strftime("%H:%M") == "13:40"
    assert w["pre_close"].strftime("%H:%M") == "15:40"


def test_friday_lunch_window_differs():
    w = dict(signal_windows(date(2026, 10, 16)))  # a Friday
    assert w["post_lunch"].strftime("%H:%M") == "14:10"


def test_weekend_is_not_a_trading_day():
    assert not is_trading_day(date(2026, 10, 17))


# ═══════════════════════════════════════════════════════════════════════════════
# R4 — the three windows, and the two session shapes behind them
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("day", MON_THU + (FRIDAY,))
def test_open_window_is_0910_on_every_trading_shape(day):
    """``open`` does not move: both shapes open at 09:00, so both signal at 09:10."""
    assert dict(signal_windows(day))["open"].strftime("%H:%M") == "09:10"


@pytest.mark.parametrize("day", MON_THU + (FRIDAY,))
def test_pre_close_window_is_1540_on_every_trading_shape(day):
    """``pre_close`` does not move either: session 2 ends 15:49:59 either way."""
    assert dict(signal_windows(day))["pre_close"].strftime("%H:%M") == "15:40"


@pytest.mark.parametrize("day", MON_THU)
def test_monday_to_thursday_post_lunch_is_1340(day):
    """Session 2 resumes at 13:30, so the signal is 13:40."""
    assert dict(signal_windows(day))["post_lunch"].strftime("%H:%M") == "13:40"


def test_friday_post_lunch_is_1410():
    """Session 2 resumes at 14:00 on Friday, so the signal is 14:10 — not 13:40.

    Forty minutes later than the rest of the week.  A calendar that only knows
    "the market closes early on Friday" gets this wrong in the opposite
    direction from the naive one: it would signal into a closed session.
    """
    assert dict(signal_windows(FRIDAY))["post_lunch"].strftime("%H:%M") == "14:10"


def test_friday_session_one_ends_at_1130_where_monday_to_thursday_ends_at_1200():
    """The difference that makes Friday special, asserted as a session bound.

    Window times alone would pass for a calendar that hardcoded Friday's
    ``post_lunch`` offset; the session bound is the underlying fact and is
    pinned separately.
    """
    assert FRIDAY_SESSIONS[0].end == time(11, 30)
    assert FRIDAY_SESSIONS[1].start == time(14, 0)

    assert MON_THU_SESSIONS[0].end == time(12, 0)
    assert MON_THU_SESSIONS[1].start == time(13, 30)

    # ...and ``sessions_for`` picks the right pair rather than the tests reading
    # the module's own constants back at it.
    assert market_calendar.sessions_for(FRIDAY) == FRIDAY_SESSIONS
    assert market_calendar.sessions_for(THURSDAY) == MON_THU_SESSIONS


def test_both_shapes_open_at_0900_and_close_session_two_at_154959():
    """The invariants that let one clock cover both shapes."""
    for sessions in (MON_THU_SESSIONS, FRIDAY_SESSIONS):
        assert sessions[0].start == time(9, 0)
        assert sessions[1].end == time(15, 49, 59)
        assert sessions[0].end <= sessions[1].start, "the sessions overlap"


def test_sessions_are_session_window_instances():
    assert all(isinstance(s, SessionWindow) for s in MON_THU_SESSIONS)
    assert all(isinstance(s, SessionWindow) for s in FRIDAY_SESSIONS)


def test_signal_windows_returns_exactly_the_three_named_keys_in_order():
    """A dict-with-order contract: three keys, this order, nothing else.

    The keys are the three signals the plan names; ``list(...)`` rather than a
    set, because a scheduler iterating the result fires them in this order and
    an unordered return would make that incidental.
    """
    for day in (MONDAY, FRIDAY):
        windows = signal_windows(day)
        assert [name for name, _ in windows] == ["open", "post_lunch", "pre_close"]
        assert len(windows) == 3


def test_signal_windows_pairs_names_with_wib_datetimes():
    for name, stamp in signal_windows(TUESDAY):
        assert isinstance(name, str)
        assert isinstance(stamp, datetime), f"{name} is not a datetime"


def test_signal_windows_are_strictly_increasing_and_inside_the_session():
    """Ordered on the clock, and each one lands in a real trading session.

    ``open`` is in session 1, ``post_lunch`` is in session 2, ``pre_close`` is in
    session 2 before the close.  A window that fell in the lunch break would be
    a signal into no market at all.
    """
    for day in MON_THU + (FRIDAY,):
        windows = dict(signal_windows(day))
        stamps = [windows[k] for k in ("open", "post_lunch", "pre_close")]
        assert stamps == sorted(stamps)

        sessions = market_calendar.sessions_for(day)
        assert sessions[0].start <= windows["open"].time() <= sessions[0].end
        assert sessions[1].start <= windows["post_lunch"].time() <= sessions[1].end
        assert sessions[1].start <= windows["pre_close"].time() <= sessions[1].end


def test_the_date_is_the_wib_date():
    """The instant carries the day it was asked for, on the Jakarta calendar."""
    for day in MON_THU + (FRIDAY,):
        for _, stamp in signal_windows(day):
            assert stamp.date() == day


# ═══════════════════════════════════════════════════════════════════════════════
# R4 — WIB awareness: tz-aware, Asia/Jakarta, +07:00, and specifically not UTC
# ═══════════════════════════════════════════════════════════════════════════════

def test_wib_is_asia_jakarta():
    assert isinstance(WIB, ZoneInfo)
    assert str(WIB) == "Asia/Jakarta"


def test_windows_carry_asia_jakarta_and_a_plus_seven_offset():
    """Not naive, and not UTC — the two ways this can be wrong.

    A naive ``datetime`` would be read as whatever the host clock is set to; a
    UTC ``datetime`` would be seven hours early.  The offset is asserted both as
    a ``timedelta`` and through ``strftime("%z")``, so an ``Etc/GMT-7`` fixed
    offset that merely happens to equal +07:00 still fails the ``tzinfo`` check.
    """
    for day in MON_THU + (FRIDAY,):
        for name, stamp in signal_windows(day):
            assert stamp.tzinfo is not None, f"{name} is a naive datetime"
            assert stamp.tzinfo is WIB, f"{name} carries {stamp.tzinfo!r}, not Asia/Jakarta"
            assert stamp.utcoffset() == timedelta(hours=7), f"{name} is not UTC+7"
            assert stamp.strftime("%z") == "+0700"
            assert stamp.isoformat().endswith("+07:00")


def test_windows_are_not_utc():
    """09:10 WIB is 02:10 UTC: the same wall-clock string is a different instant.

    The absolute check, rather than only the shape of the offset: if the module
    built UTC instants and labelled them Asia/Jakarta, the ``astimezone`` below
    would still read 09:10.
    """
    open_window = dict(signal_windows(TUESDAY))["open"]
    assert open_window.astimezone(timezone.utc).strftime("%H:%M") == "02:10"
    assert open_window.hour == 9 and open_window.minute == 10
    assert open_window.utcoffset() != timezone.utc.utcoffset(open_window)


def test_the_windows_are_the_same_instants_when_read_from_another_zone():
    """``Asia/Jakarta`` for a reader in London is still 09:10 Jakarta."""
    london = ZoneInfo("Europe/London")
    open_window = dict(signal_windows(TUESDAY))["open"]
    assert open_window.astimezone(london) == datetime(
        2026, 10, 13, 9, 10, tzinfo=WIB
    )
    # 2026-10-13 is BST (UTC+1) in London, so the same instant reads 03:10 there.
    assert open_window.astimezone(london).strftime("%H:%M") == "03:10"


def test_jakarta_has_no_dst_so_the_offset_is_seven_all_year():
    """UTC+7 year round: the July and January offsets are identical.

    Named separately because a calendar that reached for a DST-observing zone
    would pass every October test in this module and fail nothing until July.
    """
    for day in (date(2026, 1, 15), date(2026, 7, 15)):
        for _, stamp in signal_windows(day):
            assert stamp.utcoffset() == timedelta(hours=7)
            assert stamp.dst() == timedelta(0)


# ═══════════════════════════════════════════════════════════════════════════════
# R1 — is_trading_day: weekends out, known closures out, ordinary weekdays IN
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("day", [
    date(2026, 10, 17),   # Saturday
    date(2026, 10, 18),   # Sunday
    date(2026, 10, 10),
    date(2026, 10, 11),
    date(2027, 1, 2),     # a Saturday in a future year
])
def test_weekends_are_not_trading_days(day):
    assert not is_trading_day(day)


@pytest.mark.parametrize("day", PUBLIC_HOLIDAYS_2026)
def test_known_holidays_are_not_trading_days(day):
    """Christmas and friends: the signal must not fire on a closed exchange."""
    assert not is_trading_day(day)


@pytest.mark.parametrize("day", NORMAL_WEEKDAYS_2026)
def test_normal_weekdays_are_trading_days(day):
    assert is_trading_day(day)


def test_an_ordinary_future_weekday_is_a_trading_day():
    """R1: the predicate is not inverted.

    No holiday list can enumerate the future, so "not positively known" cannot
    mean ``False`` — it would mean the scheduler never fires again after the
    last curated year, silently.  These are deliberately dates beyond every
    table this module ships, and they are weekdays.
    """
    for day in (date(2030, 6, 12), date(2031, 3, 4), date(2035, 9, 19),
                date(2040, 2, 7)):
        assert day.weekday() < 5, day
        assert is_trading_day(day) is True, day


def test_every_weekday_of_a_future_year_is_trading_unless_it_is_a_known_closure():
    """R1 as a property, over all ~261 weekdays of a year past the tables.

    Set equality, both directions: no weekday is dropped merely for being in the
    future, and every listed closure is dropped.  2030 is past every table this
    module ships, so the closures in it come from the ``holidays`` projection
    alone — and there are only nine of them, all of which are dropped.
    """
    year = 2030
    weekdays = set(_weekdays(year))
    listed = {d for d in weekdays
              if d in market_calendar.indonesia_holidays(year)
              or d in IDX_EXCHANGE_HOLIDAYS}

    assert 0 < len(listed) <= 12, sorted(listed)   # a plausible year, not an empty one
    assert len(weekdays - listed) > 250, "the fixture year is mostly closures"

    got = {d for d in weekdays if is_trading_day(d)}

    assert got == weekdays - listed, sorted(weekdays - listed - got)[:5]
    assert len(got) > 250, f"only {len(got)} trading days in {year} — the predicate is inverted"


def test_the_number_of_2026_trading_days_matches_idx_published_counts():
    """A whole-year cross-check against the exchange's own monthly figures.

    IDX publishes trading days per month.  Public holidays plus cuti bersama as
    ``holidays`` 0.106 records them, plus this module's explicit IDX-only
    closure of 31 December, reproduce all twelve of the published 2026 counts
    exactly; that is the evidence that the two halves of the holiday data are
    both right and that no day is closed twice.
    """
    published_2026 = {
        1: 20, 2: 18, 3: 17, 4: 21, 5: 16, 6: 20,
        7: 23, 8: 19, 9: 22, 10: 22, 11: 21, 12: 20,
    }
    counted: dict[int, int] = {}
    for day in _weekdays(2026):
        if is_trading_day(day):
            counted[day.month] = counted.get(day.month, 0) + 1

    assert counted == published_2026
    assert sum(counted.values()) == 239


def test_the_year_end_exchange_closure_is_not_a_public_holiday_but_is_a_closure():
    """The IDX-only half of the data, and why it needs its own structure.

    31 December is not in Indonesia's public-holiday set in any year this module
    covers, so a calendar built on ``holidays`` alone would trade on it.
    """
    for year, day in ((2024, date(2024, 12, 31)),
                      (2025, date(2025, 12, 31)),
                      (2026, date(2026, 12, 31))):
        assert day in IDX_EXCHANGE_HOLIDAYS, day
        assert day not in market_calendar.indonesia_holidays(year), (
            f"{day} is a public holiday, so it does not belong in the override list"
        )
        assert is_trading_day(day) is False, day

    # The neighbouring weekday is traded, so the assertion above is about the
    # closure and not about December being switched off.
    assert is_trading_day(date(2026, 12, 30)) is True


def test_the_idx_override_list_is_sourced_data_not_a_bare_set():
    """R2: named provenance travels with the dates.

    An unsourced date cannot be reviewed when it is wrong, and this list is the
    part an annual maintenance pass has to touch.
    """
    assert IDX_EXCHANGE_HOLIDAYS, "the IDX override list is empty"
    for day, source in IDX_EXCHANGE_HOLIDAYS.items():
        assert isinstance(day, date)
        assert day.weekday() < 5, f"{day} is a weekend and closes nothing"
        assert isinstance(source, str) and len(source) > 10, (day, source)


def test_a_date_typed_datetime_is_accepted_as_well_as_a_date():
    """``datetime`` is a ``date`` subclass; the predicate must not crash on one."""
    assert is_trading_day(datetime(2026, 10, 16, 14, 15)) is True
    assert is_trading_day(datetime(2026, 10, 17, 14, 15)) is False
    assert dict(signal_windows(datetime(2026, 10, 16, 14, 15)))["post_lunch"].hour == 14


# ═══════════════════════════════════════════════════════════════════════════════
# R1/R2 — data coverage: warn, never fail, and never silently
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def forget_warned_years(monkeypatch):
    """The "warn once per year" memo is process state; reset it per test."""
    monkeypatch.setattr(market_calendar, "_WARNED_COVERAGE_YEARS", set())


def test_a_date_beyond_the_holiday_data_warns_once_and_still_answers(caplog):
    """Beyond the tables, the answer is ``True`` — with a warning beside it.

    Failing would be the silent-stop failure this plan exists to remove; saying
    nothing would be the silent-wrong-answer failure.  The module does neither:
    it answers, and it says so, once.
    """
    far = date(2030, 6, 12)

    with caplog.at_level(logging.WARNING, logger="market_calendar"):
        assert is_trading_day(far) is True
        first = [r.getMessage() for r in caplog.records]
        assert first, "a date past the holiday data was answered without a warning"
        assert any("2030" in m for m in first), first

        caplog.clear()
        for _ in range(5):
            is_trading_day(far)

    assert caplog.records == [], [r.getMessage() for r in caplog.records]


def test_the_warning_names_the_last_curated_year():
    assert isinstance(market_calendar.HOLIDAY_DATA_LAST_YEAR, int)
    assert market_calendar.HOLIDAY_DATA_LAST_YEAR < 2030


def test_a_covered_date_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING, logger="market_calendar"):
        is_trading_day(date(2026, 10, 16))
        signal_windows(date(2026, 10, 16))
    assert caplog.records == [], [r.getMessage() for r in caplog.records]


def test_signal_windows_also_warns_beyond_coverage(caplog):
    """The window clock shares the coverage gate: one module, one boundary."""
    with caplog.at_level(logging.WARNING, logger="market_calendar"):
        signal_windows(date(2031, 3, 4))
    assert caplog.records, "signal_windows answered past coverage silently"
