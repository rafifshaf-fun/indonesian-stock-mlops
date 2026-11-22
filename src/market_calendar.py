"""The IDX trading calendar and the three WIB instants a signal fires on.

This module is the repository's only statement of *when the Indonesian stock
exchange is open*.  Before it, nothing in the tree knew what day it was: a
scheduled signal job would have fired on Christmas, and nothing expressed that
Friday's two sessions are a different shape from Monday–Thursday's.  It is a
standalone module — no network, no configuration, and exactly one clock read,
:func:`wib_today` — so it can be imported by a scheduler, a backtester or a test
with no side effects.

**Never use ``date.today()`` to ask this module a question.**  The ``api``
container runs on UTC.  Between 00:00 and 07:00 WIB it is still the previous
calendar date in UTC, so ``date.today()`` names *yesterday*: a 03:00 WIB run on
the first session after a two-day closure asks about the closed day, gets
``False``, and skips the whole trading day without saying anything.
:func:`wib_today` is the date this domain means.

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

``signal_windows`` is a **pure clock** — it answers for any date, including a
Saturday — but it is no longer a *silent* one.  There are two calls, and each one
is safe in a different way:

* ``signal_windows(d)`` returns the three windows that would apply *if* the
  exchange were open, and **logs a warning when ``d`` is not a trading day**.  The
  brief's signature and return type are unchanged, so a caller already pairing it
  with ``is_trading_day`` is unaffected, and the clock can still be inspected and
  tested without a calendar — it just cannot be wrong quietly any more.
* ``trading_day_signal_windows(d)`` is the **gated** form: ``[]`` for a closure,
  the three windows for a trading day.  A caller that wants the safe default
  should not have to remember to write ``if is_trading_day(d)``.

The reasoning is the same as the coverage boundary below: a signal emitted on a
day the exchange is shut is a false positive, which is survivable but must be
visible, and "no signals, silently" is indistinguishable from a correct quiet
day.

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

Coverage, measured rather than assumed (``holidays`` 0.106, probed empirically).
The two halves of the data run out in different years, and the year matters:

* Indonesia's **decree-backed** tables are complete through **2026** — the last
  year in which no entry carries the library's ``(perkiraan)``/"estimated" label.
  **2027** is already partly projected: both Idul Fitri days and Idul Adha come
  from the MABIMS projection table, three entries out of twenty-six.
* ``cuti bersama`` (``special_government_holidays``) is tabulated by decree one
  year further, through **2027**.  From **2028** it disappears entirely and the
  annual holiday count drops from 26 (2027) to 17 — eight of them projected — so
  2028 is materially incomplete rather than merely less precise.
* Election-day holidays (``special_public_holidays``) are tabulated through
  **2024**; later elections are announced ad hoc.
* :data:`IDX_EXCHANGE_HOLIDAYS` is maintained through **2026**; IDX publishes its
  exchange calendar one year at a time.
* At the bottom, ``holidays``' Indonesia calendar has ``start_year = 1946``.
  Before that its list is empty and every weekday answers ``True``.

:data:`HOLIDAY_DATA_LAST_YEAR` is the last year for which *both* halves are
complete, and a query outside the covered range — beyond it, or before
:data:`HOLIDAY_DATA_FIRST_YEAR` — logs **one warning per year**.  It does not
raise and it does not refuse to answer; the same one-sided risk is taken at both
ends rather than silently at one.

Manifests
---------
``requirements.txt`` declares ``holidays`` and ``tzdata``.  ``tzdata`` is not
optional: :data:`WIB` reads the IANA database, which Windows does not ship, so
without the package the import above fails outright.  ``requirements-docker.txt``
is an independent list; it carries both at the same pins, so the api container
can import this module once ``serve.py`` does.  Nothing compares the two files,
so a pin bump must be made in both.

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
:data:`IDX_EXCHANGE_HOLIDAYS` from IDX's published *Kalender Libur Bursa*, bump
:data:`HOLIDAY_DATA_LAST_YEAR` with them — the module asserts the two agree, so
neither can be forgotten on its own — then extend
``tests/test_market_calendar.py``'s monthly cross-check to the new year.
"""
from __future__ import annotations

import calendar
import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from types import MappingProxyType
from typing import Mapping
from zoneinfo import ZoneInfo

import holidays
from holidays.constants import GOVERNMENT, PUBLIC

__all__ = [
    "FRIDAY_SESSIONS",
    "FRIDAY_SESSION_1",
    "FRIDAY_SESSION_2",
    "HOLIDAY_DATA_FIRST_YEAR",
    "HOLIDAY_DATA_LAST_YEAR",
    "IDX_EXCHANGE_HOLIDAYS",
    "MON_THU_SESSIONS",
    "MON_THU_SESSION_1",
    "MON_THU_SESSION_2",
    "OPEN_SIGNAL_DELAY",
    "POST_LUNCH_SIGNAL_DELAY",
    "PRE_CLOSE_SIGNAL_AT",
    "SESSION_HOURS_SOURCE",
    "SIGNAL_WINDOW_KEYS",
    "SessionWindow",
    "WIB",
    "indonesia_holidays",
    "is_trading_day",
    "sessions_for",
    "signal_windows",
    "trading_day_signal_windows",
    "wib_today",
]

LOGGER = logging.getLogger(__name__)

#: Jakarta is UTC+7 all year and observes no daylight saving, so this zone has a
#: constant offset — asserted in the tests rather than assumed.  Resolving this
#: name reads the IANA database, which is why ``tzdata`` is a hard dependency and
#: not a transitive one (see the module docstring's *Manifests* note).
WIB = ZoneInfo("Asia/Jakarta")


# ═══════════════════════════════════════════════════════════════════════════════
# The clock — this module's one read of it
# ═══════════════════════════════════════════════════════════════════════════════

def _now() -> datetime:
    """The current instant, in :data:`WIB`.

    Split out so the tests can pin "now" without patching ``datetime`` globally,
    and so that this module has exactly one clock read to audit.
    """
    return datetime.now(WIB)


def wib_today() -> date:
    """Today's date **in Jakarta**, which is not the host's date for seven hours a day.

    On a UTC host — the ``api`` container's default — ``date.today()`` returns the
    previous date between 00:00 and 07:00 WIB.  A 03:00 WIB run on the first
    session after a two-day closure would then ask about the closed day, get
    ``False`` from :func:`is_trading_day`, and skip the whole trading day without
    a word.  This function converts the instant to WIB before taking its date, so
    the answer follows the exchange's calendar rather than the container's.

    ``_now`` is converted rather than read as a WIB wall clock because a seam
    that returns a UTC instant must still land on the WIB date; the conversion is
    a no-op for the real clock.
    """
    return _now().astimezone(WIB).date()


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

    **A shortened (half-day) session is not representable.**  IDX runs early
    closes on a few days a year — historically the last trading day before Idul
    Fitri.  This type carries one ``start``/``end`` pair per session and
    :func:`sessions_for` knows exactly two shapes, so a half-day is either a
    deliberate type change (a fifth constant and a fifth branch) or nothing at
    all.  Today it is nothing at all: nothing in this module will tell a half-day
    apart from a full one, and ``pre_close`` at 15:40 would be past the close on
    such a day.  Recorded here, at the type, so the gap is discoverable rather
    than argued in review.
    """

    start: time
    end: time


#: The instrument the four bounds below come from: the exchange's own rule, not a
#: summary of it.  Held to the same named-source standard as the holiday data —
#: when IDX amends the rule, this is the string that says what was in force.
SESSION_HOURS_SOURCE = (
    "IDX Rule II-A (Peraturan Bursa Nomor II-A tentang Perdagangan Efek Bersifat "
    "Ekuitas) as implemented by Decree Kep-00196/BEI/12-2024"
)

# IDX regular-market trading hours, per SESSION_HOURS_SOURCE above: the Senin–Kamis
# and Jumat sessions of the regular market.  The annual *Kalender Libur Bursa*
# reproduces the same bounds; the decree is the authority, the calendar is the
# cross-check.  There is no half-day table here on purpose — see SessionWindow.
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

    A half-day is not representable — see :class:`SessionWindow`.
    """
    return FRIDAY_SESSIONS if d.weekday() == calendar.FRIDAY else MON_THU_SESSIONS


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

    It answers for any date, including a Saturday — but **not quietly**.  When
    ``d`` is not a trading day this logs a ``warning`` naming the date and the
    reason, then returns the windows that would have applied had the exchange
    been open.  It does not raise and it does not return ``[]``: inspecting a
    future or hypothetical date is a real use, the brief's signature and return
    type are unchanged, and a caller already paired with :func:`is_trading_day`
    is unaffected.

    A caller that wants the safe default rather than the clock should ask
    :func:`trading_day_signal_windows`, which returns ``[]`` for a closure.  The
    warning is emitted on every call rather than memoised: a scheduler iterating
    days wants one line per closed day, and the alternative — a memo keyed by
    date — grows without bound.
    """
    day = _normalise(d)
    _warn_if_outside_coverage(day)

    if not is_trading_day(day):
        LOGGER.warning(
            "signal_windows(%s) asked for a non-trading day (%s). Returning the "
            "three windows that would have applied had the exchange been open; "
            "do not trade them. Use trading_day_signal_windows() for the gated "
            "form, which returns [] here.",
            day.isoformat(), _non_trading_reason(day),
        )

    sessions = sessions_for(day)
    return [
        ("open", _wib_instant(day, sessions[0].start) + OPEN_SIGNAL_DELAY),
        ("post_lunch", _wib_instant(day, sessions[1].start) + POST_LUNCH_SIGNAL_DELAY),
        ("pre_close", _wib_instant(day, PRE_CLOSE_SIGNAL_AT)),
    ]


def trading_day_signal_windows(d: date) -> list[tuple[str, datetime]]:
    """The three WIB instants for ``d``, or ``[]`` when the exchange is shut.

    The gated companion to :func:`signal_windows`: the same three windows on a
    trading day, an empty list on a weekend or a known closure.  A scheduler that
    iterates the result of this function and nothing else cannot fire on
    Christmas, because the day is filtered out here rather than by a second call
    the caller has to remember to write.  :func:`is_trading_day` remains the
    single definition of "open", so there is no second calendar to drift.

    The empty result is **not** the "safe default indistinguishable from a real
    value" this repository treats as a defect: it is the answer to *which signals
    fire on that day*, and on a closed day exactly zero do.  No warning is logged
    for a closure — the caller asked the question that has an unambiguous
    answer — while :func:`signal_windows` still warns, because there the caller
    asked for instants that look tradeable.
    """
    day = _normalise(d)
    if not is_trading_day(day):
        return []
    return signal_windows(day)


def _non_trading_reason(day: date) -> str:
    """Why :func:`is_trading_day` says ``False``, in words, for the log line.

    Deliberately a second, narrower look at the same data: it exists so the
    warning names *which* kind of closure, which is what a reader needs to tell a
    missing ``cuti bersama`` entry from a weekend.  It never decides anything.
    """
    if day.weekday() >= calendar.SATURDAY:
        return "a weekend"
    if day in IDX_EXCHANGE_HOLIDAYS:
        return "an exchange-only closure"
    return "a listed public holiday or cuti bersama"


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
# The value is the source, and it names its own year, so the citation survives
# the addition of the next year's entry instead of collapsing into one year-less
# category label.  The year-end closure ("penutupan bursa akhir tahun") is
# announced in IDX's *Kalender Libur Bursa* for each year:
#   * 2024 — "Jadwal Libur Bursa Akhir 2024", CNBC Indonesia, 24 Dec 2024: three
#     December closure days in 2024 (25, 26 and 31), giving the exchange's
#     published 19 trading days that month.
#   * 2025 — IDX's year-end notice for the 2025 Nataru period: 31 December 2025
#     is a closure and December 2025 has 20 trading days.
#   * 2026 — IDX's 2026 *Kalender Libur Bursa*: 31 December 2026 is a closure and
#     December 2026 has 20 trading days.
def _year_end_closure_source(year: int) -> str:
    """The per-year citation for that year's 31 December exchange closure."""
    return (
        f"IDX Kalender Libur Bursa {year}: year-end exchange closure "
        f"('penutupan bursa akhir tahun') on {year}-12-31; announced for {year} by "
        f"the exchange and not an Indonesian public holiday."
    )


IDX_EXCHANGE_HOLIDAYS: dict[date, str] = {
    date(2024, 12, 31): _year_end_closure_source(2024),
    date(2025, 12, 31): _year_end_closure_source(2025),
    date(2026, 12, 31): _year_end_closure_source(2026),
}

#: The earliest year ``holidays``' Indonesia calendar describes.  Before it the
#: library's list is empty, so a pre-1946 weekday answers ``True`` silently —
#: exactly the one-sided false positive R1 accepts, and the mirror of
#: :data:`HOLIDAY_DATA_LAST_YEAR`.  A test ties this to the library's own
#: ``start_year``, which is where the number comes from.
HOLIDAY_DATA_FIRST_YEAR = 1946

#: The last calendar year for which the holiday data is **complete and decreed**
#: — see the module docstring's *Coverage* note.  2026 is the last year with no
#: ``(perkiraan)`` projection entry in ``holidays`` 0.106, and it is the last
#: year :data:`IDX_EXCHANGE_HOLIDAYS` covers.  2027 is already partly projected
#: (both Idul Fitri days and Idul Adha) even though its ``cuti bersama`` is
#: decreed, so it is past the boundary.
HOLIDAY_DATA_LAST_YEAR = 2026

# The constant above is only meaningful if it describes the data it names, and
# nothing else joins the two.  A maintainer who bumps the year but forgets to add
# that year's 31 December closure would *silently* remove the warning for the year
# whose final session is closed — the quiet wrong answer this module exists to
# prevent.  Fail at import instead; the test suite asserts the same relation.
assert HOLIDAY_DATA_LAST_YEAR == max(IDX_EXCHANGE_HOLIDAYS).year, (
    "HOLIDAY_DATA_LAST_YEAR must equal the last year in IDX_EXCHANGE_HOLIDAYS: "
    f"{HOLIDAY_DATA_LAST_YEAR} != {max(IDX_EXCHANGE_HOLIDAYS).year}. Add the "
    "missing year-end closure, or correct the constant."
)

# Years already warned about, so a scheduler looping over days does not repeat
# itself.  Module state by design; tests reset it.
_WARNED_COVERAGE_YEARS: set[int] = set()


@lru_cache(maxsize=None)
def _indonesia_holidays_cached(year: int) -> Mapping[date, str]:
    """Build ``year``'s calendar once, and hand back a read-only mapping.

    ``lru_cache`` returns the same object to every caller in the process, so
    returning a plain ``dict`` would let one caller's stray assignment poison the
    answer for everyone after it.  The proxy makes the documented read-only
    contract enforced rather than advisory.
    """
    id_calendar = holidays.country_holidays(
        "ID", years=year, categories=(PUBLIC, GOVERNMENT)
    )
    return MappingProxyType({day: name for day, name in id_calendar.items()})


def indonesia_holidays(year: int) -> Mapping[date, str]:
    """Indonesia's public holidays and ``cuti bersama`` for ``year``, keyed by date.

    Both categories are requested deliberately.  ``cuti bersama`` — the joint
    leave days declared by decree around Idul Fitri, Natal and the rest — are
    days the exchange is shut, and they are the part of the calendar that moves
    every year, so leaving them to a hand-maintained list would be exactly the
    guesswork this module exists to remove.

    Cached per year: constructing the calendar is pure and a scheduler asks about
    the same year all day.  The return value is a **read-only ``Mapping``** — a
    ``MappingProxyType`` around the cached dict, so a caller cannot mutate the
    process-wide cache through it.  Use membership tests and ``.items()``; a
    caller needing a mutable copy should take ``dict(...)`` of it.
    """
    return _indonesia_holidays_cached(year)


def _warn_if_outside_coverage(day: date) -> None:
    """Log once per out-of-coverage year that the answer is unverified.

    Warn, never raise: refusing to answer past the curated data would stop all
    signalling the year the data ran out, and doing so silently is the failure
    this plan exists to remove.  Both ends are checked, because the bottom end
    has the same one-sided risk as the top and only the top used to say so.  See
    the module docstring.
    """
    year = day.year
    if HOLIDAY_DATA_FIRST_YEAR <= year <= HOLIDAY_DATA_LAST_YEAR:
        return
    if year in _WARNED_COVERAGE_YEARS:
        return
    _WARNED_COVERAGE_YEARS.add(year)

    if year > HOLIDAY_DATA_LAST_YEAR:
        LOGGER.warning(
            "IDX holiday data is not maintained for %d (last complete year: %d). "
            "Answering from weekends and the listed holidays only, so an unlisted "
            "exchange closure in %d is treated as a trading day. Refresh "
            "holidays and IDX_EXCHANGE_HOLIDAYS.",
            year, HOLIDAY_DATA_LAST_YEAR, year,
        )
        return

    LOGGER.warning(
        "IDX holiday data does not reach back to %d (earliest year in Indonesia's "
        "decree-backed tables: %d). Answering from weekends and the listed "
        "holidays only, so a closure in %d is treated as a trading day.",
        year, HOLIDAY_DATA_FIRST_YEAR, year,
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
    exchange is shut — visible and harmless — rather than silence.  A date
    outside ``[HOLIDAY_DATA_FIRST_YEAR, HOLIDAY_DATA_LAST_YEAR]`` is answered the
    same way and additionally logs one warning per year, at both ends.

    The holiday data is therefore **data requiring annual maintenance**; the
    module docstring carries the refresh procedure.  A ``datetime`` is accepted
    and its date part is used.

    This is the single definition of "open".  :func:`signal_windows` warns when
    it says ``False``, and :func:`trading_day_signal_windows` returns ``[]``
    because of it — neither keeps a second copy of the calendar.
    """
    day = _normalise(d)
    _warn_if_outside_coverage(day)

    if day.weekday() >= calendar.SATURDAY:      # Saturday or Sunday
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
