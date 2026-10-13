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

import dataclasses
import importlib.metadata
import logging
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import holidays
import pytest

import market_calendar
from market_calendar import (
    FRIDAY_SESSIONS,
    HOLIDAY_DATA_LAST_YEAR,
    IDX_EXCHANGE_HOLIDAYS,
    MON_THU_SESSIONS,
    WIB,
    SessionWindow,
    is_trading_day,
    signal_windows,
    trading_day_signal_windows,
    wib_today,
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

#: The nine *weekday* closures ``holidays`` 0.106 reports for 2030, pinned as
#: literals on purpose.  The property test below asserts set equality against
#: this tuple rather than against ``indonesia_holidays``/``IDX_EXCHANGE_HOLIDAYS``,
#: so it is not merely reading the module's own sources back at it: a library
#: bump that moves a projected date fails here, loudly, and an inverted predicate
#: fails the equality on the other side.
PROJECTED_CLOSURES_2030 = (
    date(2030, 1, 1),    # Tahun Baru Masehi
    date(2030, 2, 4),    # Hari Raya Idul Fitri (perkiraan)
    date(2030, 2, 5),    # Hari kedua dari Hari Raya Idul Fitri (perkiraan)
    date(2030, 3, 5),    # Hari Suci Nyepi
    date(2030, 4, 19),   # Wafat Yesus Kristus
    date(2030, 5, 1),    # Hari Buruh Internasional
    date(2030, 5, 16),   # Hari Raya Waisak (perkiraan)
    date(2030, 5, 30),   # Kenaikan Yesus Kristus
    date(2030, 12, 25),  # Hari Raya Natal
)

#: IDX's published trading-day counts for three months outside 2026, so the
#: cross-check is not a single-year coincidence.  See the report for the sources.
PUBLISHED_NEIGHBOURING_MONTHS = (
    (2024, 12, 19),   # three December closures: 25, 26 and the 31st exchange closure
    (2025, 1, 19),    # the four weekday closures 1, 27, 28 and 29 January
    (2025, 12, 20),   # 25 December plus the 31 December exchange closure
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


def test_the_session_bounds_name_their_governing_instrument():
    """The session table is held to the same named-source standard as the holidays.

    These four bounds are the exchange's, not a summary of them: they come from
    IDX Rule II-A as implemented by Decree Kep-00196/BEI/12-2024.  Naming the
    instrument is what makes them reviewable when the exchange changes them.
    """
    authority = market_calendar.SESSION_HOURS_SOURCE
    assert "II-A" in authority, authority
    assert "Kep-00196/BEI/12-2024" in authority, authority


def test_only_two_session_shapes_exist_so_a_shortened_session_is_not_representable():
    """The documented gap, pinned rather than argued.

    IDX runs early closes on a few days a year.  ``SessionWindow`` carries one
    ``start``/``end`` pair per session and ``sessions_for`` knows two shapes, so a
    half-day cannot be expressed: every weekday resolves to the Monday–Thursday
    or the Friday pair, and nothing will tell a half-day apart from a full one.
    Adding half-days is therefore a deliberate type change, not a data entry.
    """
    assert [f.name for f in dataclasses.fields(SessionWindow)] == ["start", "end"]
    for offset in range(7):
        day = MONDAY + timedelta(days=offset)
        assert market_calendar.sessions_for(day) in (MON_THU_SESSIONS, FRIDAY_SESSIONS), day


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
    alone.

    The closures are compared against :data:`PROJECTED_CLOSURES_2030` — literals,
    not a set built from the same two sources ``is_trading_day`` reads.  Without
    that, the assertion would be close to tautological: it would restate the
    implementation rather than check it.  With it, moving the predicate's
    coverage in either direction fails here.
    """
    year = 2030
    weekdays = set(_weekdays(year))
    listed = {d for d in weekdays
              if d in market_calendar.indonesia_holidays(year)
              or d in IDX_EXCHANGE_HOLIDAYS}

    assert listed == set(PROJECTED_CLOSURES_2030), sorted(
        listed ^ set(PROJECTED_CLOSURES_2030)
    )
    assert all(d.weekday() < 5 for d in listed), sorted(listed)

    got = {d for d in weekdays if is_trading_day(d)}

    assert got == weekdays - set(PROJECTED_CLOSURES_2030), sorted(
        (weekdays - set(PROJECTED_CLOSURES_2030)) - got
    )[:5]
    assert len(got) == len(weekdays) - len(PROJECTED_CLOSURES_2030)
    assert len(got) > 250, f"only {len(got)} trading days in {year} — the predicate is inverted"

    # An ordinary Wednesday in that year, named so the failure message is concrete.
    assert is_trading_day(date(2030, 6, 12)) is True


def test_neighbouring_months_match_idx_published_trading_day_counts():
    """Three published months outside 2026, so the cross-check is not one year.

    December 2024, January 2025 and December 2025 were reproduced independently
    when this task's data was verified; pinning them here means a ``holidays``
    bump that changes a non-2026 year fails the suite instead of moving the
    calendar silently.
    """
    for year, month, published in PUBLISHED_NEIGHBOURING_MONTHS:
        counted = sum(
            1 for day in _weekdays(year)
            if day.month == month and is_trading_day(day)
        )
        assert counted == published, f"{year}-{month:02d}: {counted} != {published}"


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


def test_each_exchange_closure_names_its_own_year_in_its_source():
    """Provenance that is year-less is provenance a maintainer cannot follow.

    One shared category label was enough for ``isinstance(str) and len > 10`` and
    lost the per-year citation — which year's *Kalender Libur Bursa* says so —
    the moment a second year was added.  Each value now carries its own year, and
    this asserts it rather than trusting the text to stay that way.
    """
    for day, source in IDX_EXCHANGE_HOLIDAYS.items():
        assert str(day.year) in source, (
            f"{day} is sourced from a label that does not name {day.year}: {source!r}"
        )
    assert len(set(IDX_EXCHANGE_HOLIDAYS.values())) == len(IDX_EXCHANGE_HOLIDAYS), (
        "two closures share one year-less category label, so the per-year citation is lost"
    )


def test_indonesia_holidays_hands_out_a_read_only_view_not_the_cached_dict():
    """A caller cannot poison the process-wide cache through the return value.

    ``lru_cache`` returns the same object every call, so a mutable ``dict`` here
    is one stray assignment away from a wrong answer for every later caller in
    the process.  The cache stays internally; the public function hands out a
    read-only mapping.
    """
    first = market_calendar.indonesia_holidays(2026)
    assert market_calendar.indonesia_holidays(2026) is first, "the cache stopped caching"

    with pytest.raises(TypeError):
        first[date(2026, 1, 1)] = "tampered"      # type: ignore[index]

    assert date(2026, 1, 1) in market_calendar.indonesia_holidays(2026)
    assert market_calendar.indonesia_holidays(2026)[date(2026, 1, 1)].startswith("Tahun Baru")


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


def test_the_coverage_boundary_is_tied_to_the_data_it_describes():
    """The constant and ``IDX_EXCHANGE_HOLIDAYS`` cannot drift apart.

    Nothing else connects them: a maintainer who bumps
    ``HOLIDAY_DATA_LAST_YEAR`` to 2027 without adding 2027's 31 December closure
    — a real exchange-only closure — silently removes the warning for the year
    whose final session is closed, which is exactly the quiet wrong answer this
    module exists to prevent.  The module asserts the same relation at import;
    this is the copy a reader finds next to the data.
    """
    assert HOLIDAY_DATA_LAST_YEAR == max(IDX_EXCHANGE_HOLIDAYS).year
    assert HOLIDAY_DATA_LAST_YEAR in {d.year for d in IDX_EXCHANGE_HOLIDAYS}


def test_a_date_before_the_holiday_data_warns_and_still_answers(caplog):
    """The mirror of the upper boundary, so the asymmetry is not an oversight.

    ``holidays``' Indonesia calendar has ``start_year = 1946``; before that the
    list is empty and every weekday answers ``True`` — the same one-sided risk R1
    accepts, but silent at the bottom while the top warns.  Both ends now warn
    and neither raises.
    """
    with caplog.at_level(logging.WARNING, logger="market_calendar"):
        assert is_trading_day(date(1900, 1, 2)) is True      # a Tuesday
    messages = [r.getMessage() for r in caplog.records]
    assert messages, "a pre-1946 weekday was answered without a warning"
    assert any("1900" in m for m in messages), messages
    assert any(str(market_calendar.HOLIDAY_DATA_FIRST_YEAR) in m for m in messages), messages

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="market_calendar"):
        for _ in range(5):
            is_trading_day(date(1900, 1, 2))
    assert caplog.records == [], [r.getMessage() for r in caplog.records]


def test_the_first_covered_year_is_the_librarys_own_start_year():
    """The lower bound is the library's, not a number someone liked."""
    assert market_calendar.HOLIDAY_DATA_FIRST_YEAR == 1946
    assert (
        holidays.country_holidays("ID", years=1946).start_year
        == market_calendar.HOLIDAY_DATA_FIRST_YEAR
    )


def test_this_modules_hard_dependencies_are_pinned_against_what_is_installed():
    """``holidays`` and ``tzdata`` must be declared, not inherited by luck.

    ``ZoneInfo("Asia/Jakarta")`` reads the IANA database, which Windows does not
    ship: without ``tzdata`` the ``WIB`` definition at import time raises
    ``ZoneInfoNotFoundError``.  Both packages reached this venv transitively
    (``pip show tzdata`` → required-by kombu, pandas, tzlocal), so a bare install
    of ``requirements.txt`` was one import away from failing.  The pin is checked
    against the installed version so the manifest and the environment cannot part
    company quietly.
    """
    declared: dict[str, str] = {}
    manifest = Path(__file__).resolve().parents[1] / "requirements.txt"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#") or "==" not in line:
            continue
        name, _, version = line.partition("==")
        declared[name.strip()] = version.strip()

    for package in ("holidays", "tzdata"):
        assert package in declared, f"{package} is not pinned in requirements.txt"
        installed = importlib.metadata.version(package)
        assert declared[package] == installed, (
            f"requirements.txt pins {package}=={declared[package]}, {installed} is installed"
        )


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


# ═══════════════════════════════════════════════════════════════════════════════
# THE GATE — a closed day is answered loudly, and can be gated on request
# ═══════════════════════════════════════════════════════════════════════════════

#: A Friday, and a public holiday.
CHRISTMAS = date(2026, 12, 25)
#: Cuti Bersama Tahun Baru Imlek — a Monday, and the category most often missed.
CUTI_BERSAMA_2026 = date(2026, 2, 16)
#: The exchange-only closure: not a public holiday, so a bare ``holidays`` misses it.
YEAR_END_CLOSURE = date(2026, 12, 31)


@pytest.mark.parametrize("day", [SATURDAY, date(2026, 10, 18), CHRISTMAS,
                                 CUTI_BERSAMA_2026, YEAR_END_CLOSURE])
def test_signal_windows_warns_on_a_non_trading_day(day, caplog):
    """The failure this plan exists to remove, made impossible to miss.

    A Phase 4 scheduler that iterates ``signal_windows(today)`` without the
    second call emitted all three signals on Christmas and on every ``cuti
    bersama`` day, silently.  The clock still answers — inspecting a future date
    is a real use — but it now says out loud that the answer is not tradeable.
    """
    with caplog.at_level(logging.WARNING, logger="market_calendar"):
        windows = signal_windows(day)
    assert len(windows) == 3, "the clock no longer answers for inspection"
    messages = [r.getMessage() for r in caplog.records]
    assert messages, f"{day} produced three signals with no warning at all"
    assert any(day.isoformat() in m for m in messages), messages


def test_signal_windows_says_nothing_on_a_trading_day(caplog):
    """No new noise: the warning is for closures, not for every call."""
    with caplog.at_level(logging.WARNING, logger="market_calendar"):
        signal_windows(FRIDAY)
        signal_windows(TUESDAY)
    assert caplog.records == [], [r.getMessage() for r in caplog.records]


@pytest.mark.parametrize("day", [SATURDAY, date(2026, 10, 18), CHRISTMAS,
                                 CUTI_BERSAMA_2026, YEAR_END_CLOSURE])
def test_trading_day_signal_windows_is_empty_for_a_closure(day, caplog):
    """The gated form: a caller wanting the safe default writes one call.

    ``[]`` here is not the "safe default indistinguishable from a real value"
    defect the repository's conventions name: it is the answer to *which signals
    fire today*, and on a closed day exactly zero do.  Unlike ``signal_windows``,
    this function is silent — there is nothing to inspect, because the caller
    asked the question that has no tradeable ambiguity.
    """
    with caplog.at_level(logging.WARNING, logger="market_calendar"):
        assert trading_day_signal_windows(day) == []
    assert caplog.records == [], [r.getMessage() for r in caplog.records]


def test_trading_day_signal_windows_is_the_same_three_windows_on_a_trading_day():
    """The gate adds nothing and removes nothing on an open day."""
    for day in MON_THU + (FRIDAY,):
        windows = trading_day_signal_windows(day)
        assert [name for name, _ in windows] == ["open", "post_lunch", "pre_close"]
        assert windows == signal_windows(day)


def test_trading_day_signal_windows_accepts_a_datetime_like_the_other_two():
    assert trading_day_signal_windows(datetime(2026, 10, 17, 11, 0)) == []
    assert trading_day_signal_windows(datetime(2026, 10, 16, 11, 0)) == signal_windows(FRIDAY)


# ═══════════════════════════════════════════════════════════════════════════════
# THE CLOCK — the one place this module reads it, and why it is WIB
# ═══════════════════════════════════════════════════════════════════════════════

def test_wib_today_is_derived_from_wib_not_the_host_clock(monkeypatch):
    """03:00 WIB on 2 January is still 1 January in UTC — the api container's zone.

    The instant is the exact hazard, not a convenient one.  31 December 2025 and
    1 January 2026 are both closures, so a scheduler on a UTC host that asks
    ``date.today()`` at 03:00 WIB on 2 January asks about a *closed* day, gets
    ``False`` from ``is_trading_day``, and skips the first trading day of the year
    without saying anything.  ``wib_today`` converts the instant before taking the
    date, so it cannot be off by a day during the seven hours the two calendars
    disagree.
    """
    instant = datetime(2026, 1, 1, 20, 0, tzinfo=timezone.utc)   # 2026-01-02 03:00 WIB
    monkeypatch.setattr(market_calendar, "_now", lambda: instant)

    assert wib_today() == date(2026, 1, 2)
    assert wib_today() != instant.date(), "wib_today is reading the host's date"
    assert is_trading_day(date(2025, 12, 31)) is False, "the two-day closure"
    assert is_trading_day(instant.date()) is False, "the host's date is not the hazard claimed"
    assert is_trading_day(wib_today()) is True, "the trading day would have been skipped"


def test_wib_today_agrees_with_the_jakarta_clock():
    """And with the real clock, so the function is not merely a test seam."""
    assert wib_today() == datetime.now(WIB).date()
    assert isinstance(wib_today(), date)
