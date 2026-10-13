"""The IDX trading calendar and the three WIB instants a signal fires on.

This module is the repository's only statement of *when the Indonesian stock
exchange is open*.  Before it, nothing in the tree knew what day it was: a
scheduled signal job would have fired on Christmas, and nothing expressed that
Friday's two sessions are a different shape from Monday–Thursday's.  It is a
standalone, I/O-free module — no network, no clock reads, no configuration — so
it can be imported by a scheduler, a backtester or a test with no side effects.

Sessions
--------
The IDX regular market trades in two sessions with a lunch break between them,
and Friday is not a shifted Monday.  Published exchange hours, in WIB
(``Asia/Jakarta``, UTC+7 year round, no DST)::

    Monday–Thursday   session 1  09:00:00 – 12:00:00
                      session 2  13:30:00 – 15:49:59
    Friday            session 1  09:00:00 – 11:30:00
                      session 2  14:00:00 – 15:49:59

Friday closes session 1 thirty minutes early and resumes session 2 thirty
minutes late, so the lunch break is two hours rather than one and a half.  Both
shapes open at 09:00 and both end at 15:49:59 — which is why ``open`` and
``pre_close`` are the same wall-clock instant all week while ``post_lunch`` is
not.

Signal windows
--------------
Three signals a day, at fixed exchange moments::

    open        09:10   open + 10 min              same every trading day
    post_lunch  13:40   session 2 start + 10 min   Monday–Thursday
                14:10   session 2 start + 10 min   Friday
    pre_close   15:40   ten minutes before the close

``pre_close`` is pinned as 15:40 rather than derived as ``15:49:59 - 10 min``
(which is 15:39:59): 15:40 is the instant the signal plan names, and rounding it
to the minute keeps the three windows on a minute boundary a cron expression can
express.

``signal_windows`` is a **pure clock**.  It answers for any date, including a
weekend — ``is_trading_day`` is the gate, and it is a separate call.  A caller
that wants signals *for a trading day* must ask both; keeping the two apart
means the clock can be inspected and tested without a calendar, and the calendar
can be tested without a clock.

Holidays are data, and the data needs an annual maintenance pass
--------------------------------------------------------------
Two sources combine:

``holidays`` (currently 0.106, pinned in ``requirements.txt``) supplies
Indonesia's decree-backed set — public holidays *and* the ``cuti bersama``
(joint/collective leave) days the exchange also observes.  Both categories are
requested: ``cuti bersama`` is exactly the part that moves every year by
government decree and is therefore the part most likely to be got wrong by hand.

:data:`IDX_EXCHANGE_HOLIDAYS` supplies the closures that are **not** public
holidays — exchange-only closures, which no national calendar can know about.
Each entry names its source.  Currently that is the year-end exchange closure
(31 December), which IDX observes but the Indonesian public-holiday set does not
contain.

Coverage, measured rather than assumed (``holidays`` 0.106, probed empirically):

* ``cuti bersama`` (``special_government_holidays``) is tabulated by decree
  through **2027**.
* Election-day holidays (``special_public_holidays``) are tabulated through
  **2024**; later elections are announced ad hoc.
* From **2028** the Islamic dates are produced from the MABIMS projection table
  and carry the library's ``(perkiraan)``/"estimated" label rather than a
  decree; the public-holiday count drops from 25 in 2026 to 17 in 2028 and the
  joint-leave days disappear entirely.
* :data:`IDX_EXCHANGE_HOLIDAYS` is maintained through **2026**; IDX publishes its
  exchange calendar one year at a time.

:data:`HOLIDAY_DATA_LAST_YEAR` is the last year for which *both* halves are
complete, and a query beyond it logs **one warning per year** — it does not
raise and it does not refuse to answer.

Why the predicate is not inverted
---------------------------------
The obvious reading of "return ``False`` for any date not positively known to be
a trading day" is wrong, and expensively so.  No holiday list can enumerate the
future, so that reading makes *every* future weekday ``False``: the scheduler
would stop firing the day the curated data ran out, and it would stop silently —
the exact failure mode the rest of this plan exists to remove.

:func:`is_trading_day` therefore returns ``False`` for a weekend and for a date
in the known-closure set, and ``True`` for any other weekday.  The residual risk
is deliberately one-sided: an unlisted closure is a **false positive** (a signal
fires on a day the exchange is shut, and the ingest step sees no new bar) rather
than a false negative that stops all signalling.  The coverage warning above is
what makes that risk visible instead of silent.

Refresh procedure, once a year: bump ``holidays`` (its release notes name the
years it added), add the new year's exchange-only closures to
:data:`IDX_EXCHANGE_HOLIDAYS` from IDX's published *Kalender Libur Bursa*, then
extend ``tests/test_market_calendar.py``'s monthly cross-check to the new year.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import holidays
from holidays.constants import GOVERNMENT, PUBLIC

__all__ = [
    "FRIDAY_SESSIONS",
    "FRIDAY_SESSION_1",
    "FRIDAY_SESSION_2",
    "HOLIDAY_DATA_LAST_YEAR",
    "IDX_EXCHANGE_HOLIDAYS",
    "MON_THU_SESSIONS",
    "MON_THU_SESSION_1",
    "MON_THU_SESSION_2",
    "OPEN_SIGNAL_DELAY",
    "POST_LUNCH_SIGNAL_DELAY",
    "PRE_CLOSE_SIGNAL_AT",
    "SIGNAL_WINDOW_KEYS",
    "SessionWindow",
    "WIB",
    "indonesia_holidays",
    "is_trading_day",
    "sessions_for",
    "signal_windows",
]

LOGGER = logging.getLogger(__name__)

#: Jakarta is UTC+7 all year and observes no daylight saving, so this zone has a
#: constant offset — asserted in the tests rather than assumed.
WIB = ZoneInfo("Asia/Jakarta")


# ═══════════════════════════════════════════════════════════════════════════════
# Sessions
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SessionWindow:
    """One continuous trading session, as WIB wall-clock bounds.

    ``end`` is the last second of continuous trade: session 2 ends at
    ``15:49:59``, not at ``15:50:00``.  These are the exchange's published bounds
    and nothing here derives one session from the other, because Friday's differ
    in both directions.
    """

    start: time
    end: time


# IDX regular market hours.  Source: the exchange's published trading hours for
# the regular market (Senin–Kamis and Jumat sessions), as reproduced in IDX's
# annual *Kalender Libur Bursa* announcements and in the 2026 exchange-calendar
# coverage of the Jakarta market press.
MON_THU_SESSION_1 = SessionWindow(start=time(9, 0, 0), end=time(12, 0, 0))
MON_THU_SESSION_2 = SessionWindow(start=time(13, 30, 0), end=time(15, 49, 59))
FRIDAY_SESSION_1 = SessionWindow(start=time(9, 0, 0), end=time(11, 30, 0))
FRIDAY_SESSION_2 = SessionWindow(start=time(14, 0, 0), end=time(15, 49, 59))

MON_THU_SESSIONS: tuple[SessionWindow, ...] = (MON_THU_SESSION_1, MON_THU_SESSION_2)
FRIDAY_SESSIONS: tuple[SessionWindow, ...] = (FRIDAY_SESSION_1, FRIDAY_SESSION_2)


def sessions_for(d: date) -> tuple[SessionWindow, ...]:
    """The session bounds in force on ``d``: Friday's pair, or the week's.

    Only Friday is special; a Saturday and a Sunday get the Monday–Thursday pair
    because a non-trading day has no sessions at all and inventing a third pair
    for it would be a distinction with no observable consequence.  Pair this with
    :func:`is_trading_day` before treating anything here as tradeable.
    """
    return FRIDAY_SESSIONS if d.weekday() == 4 else MON_THU_SESSIONS


# ═══════════════════════════════════════════════════════════════════════════════
# The three signals
# ═══════════════════════════════════════════════════════════════════════════════

OPEN_SIGNAL_DELAY = timedelta(minutes=10)
POST_LUNCH_SIGNAL_DELAY = timedelta(minutes=10)

#: Ten minutes before the close, pinned to the minute the plan names.  Session 2
#: ends at 15:49:59, so the literal arithmetic would give 15:39:59.
PRE_CLOSE_SIGNAL_AT = time(15, 40, 0)

#: The keys :func:`signal_windows` returns, in the order it returns them.  The
#: order is part of the contract: a scheduler iterating the result fires the
#: three signals in this sequence.
SIGNAL_WINDOW_KEYS: tuple[str, ...] = ("open", "post_lunch", "pre_close")


def signal_windows(d: date) -> list[tuple[str, datetime]]:
    """The three WIB instants a signal fires on ``d``, in ``SIGNAL_WINDOW_KEYS`` order.

    Pure clock: every returned ``datetime`` carries
    :data:`WIB` (``Asia/Jakarta``, ``+07:00``) — never a naive value and never
    UTC — and ``d`` is the WIB date of each instant.  The result is a list of
    pairs rather than a mapping so that the order is explicit and a caller
    cannot accidentally iterate a set.

    No gate on the trading calendar: this answers for a Saturday too.
    :func:`is_trading_day` is the separate, explicit check, and a scheduler must
    call it.  Asking for the windows of a non-trading day yields the windows that
    would have applied had the exchange been open — useful for inspection, and
    never a reason to trade.
    """
    day = _normalise(d)
    _warn_if_beyond_coverage(day)

    sessions = sessions_for(day)
    return [
        ("open", _wib_instant(day, sessions[0].start) + OPEN_SIGNAL_DELAY),
        ("post_lunch", _wib_instant(day, sessions[1].start) + POST_LUNCH_SIGNAL_DELAY),
        ("pre_close", _wib_instant(day, PRE_CLOSE_SIGNAL_AT)),
    ]


def _wib_instant(day: date, at: time) -> datetime:
    """``day`` at WIB wall-clock time ``at``, as an aware ``datetime``.

    Built field by field rather than with ``datetime.combine`` because the
    ``tzinfo`` argument of ``combine`` is easy to leave out by accident, and a
    silently naive result here is read as the host's local time.
    """
    return datetime(day.year, day.month, day.day,
                    at.hour, at.minute, at.second,
                    tzinfo=WIB)


# ═══════════════════════════════════════════════════════════════════════════════
# Holidays
# ═══════════════════════════════════════════════════════════════════════════════

# IDX-only exchange closures: days the exchange does not trade that are *not* in
# Indonesia's public-holiday set, so ``holidays`` cannot supply them.  Data, not
# a rule: IDX publishes its exchange calendar one year at a time, so this cannot
# be derived from a formula and has to be extended by hand each year.
#
# Source, per entry, is the value.  The year-end closure ("penutupan bursa akhir
# tahun") is announced in IDX's *Kalender Libur Bursa* for each year:
#   * 2024 — "Jadwal Libur Bursa Akhir 2024", CNBC Indonesia, 24 Dec 2024: three
#     closure days in December (25, 26 and 31), giving the exchange's published
#     19 trading days that month.
#   * 2025 / 2026 — IDX's year-end notice for the 2025 Nataru period and the
#     2026 *Kalender Libur Bursa*, both reporting 31 December as an exchange
#     closure and 20 trading days in the respective Decembers.
_YEAR_END_CLOSURE = (
    "IDX Kalender Libur Bursa: year-end exchange closure ('penutupan bursa "
    "akhir tahun'); announced per year by the exchange and not a public holiday."
)

IDX_EXCHANGE_HOLIDAYS: dict[date, str] = {
    date(2024, 12, 31): _YEAR_END_CLOSURE,
    date(2025, 12, 31): _YEAR_END_CLOSURE,
    date(2026, 12, 31): _YEAR_END_CLOSURE,
}

#: The last calendar year for which both halves of the holiday data are complete
#: — see the module docstring's COVERAGE note.  ``holidays`` 0.106 carries
#: decree-backed ``cuti bersama`` through 2027 and switches to rule-projected
#: estimates from 2028; :data:`IDX_EXCHANGE_HOLIDAYS` is maintained through 2026,
#: the last year IDX had published a calendar for.  The conservative minimum of
#: the two is the boundary, so 2027 is already past it.
HOLIDAY_DATA_LAST_YEAR = 2026

# Years already warned about, so a scheduler looping over days does not repeat
# itself.  Module state by design; tests reset it.
_WARNED_COVERAGE_YEARS: set[int] = set()


@lru_cache(maxsize=None)
def indonesia_holidays(year: int) -> dict[date, str]:
    """Indonesia's public holidays and ``cuti bersama`` for ``year``, keyed by date.

    Both categories are requested deliberately.  ``cuti bersama`` — the joint
    leave days declared by decree around Idul Fitri, Natal and the rest — are
    days the exchange is shut, and they are the part of the calendar that moves
    every year, so leaving them to a hand-maintained list would be exactly the
    guesswork this module exists to remove.

    Cached per year: constructing the calendar is pure and the result is
    read-only for callers, and a scheduler asks about the same year all day.
    """
    calendar = holidays.country_holidays(
        "ID", years=year, categories=(PUBLIC, GOVERNMENT)
    )
    return {day: name for day, name in calendar.items()}


def _warn_if_beyond_coverage(day: date) -> None:
    """Log once per out-of-coverage year that the answer is unverified.

    Warn, never raise: refusing to answer past the curated data would stop all
    signalling the year the data ran out, and doing so silently is the failure
    this plan exists to remove.  See the module docstring.
    """
    year = day.year
    if year <= HOLIDAY_DATA_LAST_YEAR or year in _WARNED_COVERAGE_YEARS:
        return
    _WARNED_COVERAGE_YEARS.add(year)
    LOGGER.warning(
        "IDX holiday data is not maintained for %d (last complete year: %d). "
        "Answering from weekends and the listed holidays only, so an unlisted "
        "exchange closure in %d is treated as a trading day. Refresh "
        "holidays and IDX_EXCHANGE_HOLIDAYS.",
        year, HOLIDAY_DATA_LAST_YEAR, year,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# The trading-day predicate
# ═══════════════════════════════════════════════════════════════════════════════

def is_trading_day(d: date) -> bool:
    """``True`` when the IDX regular market trades on ``d``.

    ``False`` for a Saturday or Sunday, ``False`` for a date in the known-closure
    set (Indonesia's public holidays and ``cuti bersama`` from
    :func:`indonesia_holidays`, plus the exchange-only closures in
    :data:`IDX_EXCHANGE_HOLIDAYS`), and ``True`` for every other weekday —
    **including a weekday in the future that no list enumerates**.

    That last clause is the whole point of the predicate and the reason it is not
    inverted.  No holiday list can enumerate the future, so "``False`` unless
    positively known" would make every future weekday a non-trading day and stop
    every signal, silently, the year the data ran out.  The risk is instead
    taken on the other side: an unlisted closure produces a signal on a day the
    exchange is shut — visible and harmless — rather than silence.  A date past
    :data:`HOLIDAY_DATA_LAST_YEAR` is answered the same way and additionally
    logs one warning per year.

    The holiday data is therefore **data requiring annual maintenance**; the
    module docstring carries the refresh procedure.  A ``datetime`` is accepted
    and its date part is used.
    """
    day = _normalise(d)
    _warn_if_beyond_coverage(day)

    if day.weekday() >= 5:                      # Saturday or Sunday
        return False
    if day in IDX_EXCHANGE_HOLIDAYS:            # exchange-only closures
        return False
    return day not in indonesia_holidays(day.year)


def _normalise(d: date) -> date:
    """``d``'s date part — a ``datetime`` carries a time this module ignores.

    ``datetime`` is a ``date`` subclass but does **not** compare equal to, or hash
    like, its own date part, so a ``datetime`` reaching a ``dict[date, ...]``
    lookup would silently miss every holiday and every override.  Normalising at
    the boundary is what keeps the accepted-input widening honest.
    """
    return d.date() if isinstance(d, datetime) else d
