"""Thin OpenF1 client with rate limiting, retries and an on-disk cache (P1.5).

API facts this relies on (checked against live responses on 2026-11-18, recorded in
``docs/data.md``):

- Base URL ``https://api.openf1.org/v1``; endpoints used: ``meetings``, ``sessions``,
  ``drivers``, ``laps``, ``stints``, ``location``, ``car_data``. Each returns a JSON list of flat objects.
- Equality filters are query parameters (``session_key=9492``); comparison filters put the
  operator in the parameter (``date>=2024-04-06T06:58:14``), URL-encoded or not.
- An empty result is **HTTP 404** with ``{"detail": "No results found."}``; it is returned (and
  cached) as an empty list.
- ``location`` x/y/z are integers in decimetres (``utils.units.OPENF1_POS_TO_M``); samples are
  ~3.7-4 Hz.

Every response is cached under ``<APEX_CACHE_DIR>/openf1/`` keyed by the sha256 of the canonical
request URL, so repeated queries never touch the network. Requests are throttled by two token
buckets matching OpenF1's published limits (3 req/s and 30 req/min, the latter observed as HTTP
429 "Max 30 requests/minute" on 2026-11-18) and retried with exponential backoff (honouring
``Retry-After``) on 429, 5xx and transport errors.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Literal, Self, cast
from urllib.parse import quote

import httpx

from apex_zero.data.cache import SessionKey, normalize_session, slugify
from apex_zero.errors import DataUnavailableError
from apex_zero.utils.logging import get_logger

__all__ = [
    "BASE_URL",
    "ENDPOINTS",
    "Filter",
    "JSONValue",
    "OpenF1Client",
    "OpenF1Session",
    "Record",
    "TokenBucket",
    "build_url",
]

log = get_logger(__name__)

BASE_URL: str = "https://api.openf1.org/v1"
ENDPOINTS: frozenset[str] = frozenset(
    {"meetings", "sessions", "drivers", "laps", "stints", "location", "car_data"}
)

JSONValue = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
Record = dict[str, JSONValue]
Scalar = str | int | float | bool
Op = Literal["=", ">", ">=", "<", "<="]
#: ``(field, operator, value)``, e.g. ``("date", ">=", "2024-04-06T06:58:14")``.
Filter = tuple[str, Op, Scalar]

_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
_MAX_WAIT_S = 60.0


class TokenBucket:
    """Token-bucket rate limiter.

    Args:
        rate_per_s: Sustained requests per second.
        capacity: Burst size (tokens available when idle).
        clock: Monotonic clock in seconds (injectable for tests).
        sleep: Sleep function (injectable for tests).
    """

    def __init__(
        self,
        rate_per_s: float,
        capacity: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rate_per_s <= 0 or capacity < 1:
            raise ValueError("rate_per_s must be > 0 and capacity >= 1")
        self.rate_per_s = rate_per_s
        self.capacity = capacity
        self._clock = clock
        self._sleep = sleep
        self._tokens = capacity
        self._last = clock()

    def acquire(self) -> float:
        """Take one token, sleeping until one is available. Returns the time slept [s]."""
        now = self._clock()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate_per_s)
        self._last = now
        waited = 0.0
        if self._tokens < 1.0:
            waited = (1.0 - self._tokens) / self.rate_per_s
            self._sleep(waited)
            self._tokens = 1.0
            self._last = self._clock()
        self._tokens -= 1.0
        return waited


def _fmt(value: Scalar) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def build_url(
    base_url: str,
    endpoint: str,
    params: Mapping[str, Scalar] | None = None,
    filters: Sequence[Filter] = (),
) -> str:
    """Canonical request URL: parameters sorted, values percent-encoded.

    The same query always gives the same URL, hence the same cache key.
    """
    terms = [(k, "=", v) for k, v in (params or {}).items()] + list(filters)
    # Equality: ``key=value``. Comparison: the whole ``key>=value`` term percent-encoded (OpenF1
    # accepts both the raw and the encoded form; encoded keeps the URL unambiguous).
    parts = [
        f"{quote(k, safe='_')}={quote(_fmt(v), safe='')}"
        if op == "="
        else quote(f"{k}{op}{_fmt(v)}", safe="_")
        for k, op, v in sorted(terms, key=lambda t: (t[0], t[1], _fmt(t[2])))
    ]
    query = "&".join(parts)
    return f"{base_url.rstrip('/')}/{endpoint}" + (f"?{query}" if query else "")


@dataclass(frozen=True, slots=True)
class OpenF1Session:
    """An OpenF1 session and the FastF1-compatible identity it maps to."""

    session_key: int
    meeting_key: int
    year: int
    meeting_name: str
    session_name: str
    session_code: str
    circuit_short_name: str
    country_name: str
    location: str
    date_start: str

    @property
    def cache_key(self) -> SessionKey:
        """Same processed-cache key FastF1 gives this session (event slug from the GP name)."""
        return SessionKey(self.year, slugify(self.meeting_name), self.session_code)


class OpenF1Client:
    """Cached, rate-limited OpenF1 client.

    Args:
        cache_dir: Response cache directory (``data.cache.openf1_cache_dir(paths)``).
        base_url: API root.
        rate_per_s: Per-second request rate (OpenF1 limit: 3 req/s).
        burst: Per-second bucket capacity (1: evenly spaced requests).
        per_minute: Requests per minute (OpenF1 limit: 30; default 29 for margin), enforced by
            evenly spaced requests (a bucket of capacity 1), so no 60 s window can exceed it.
            A bucket with a larger burst would let ``burst + per_minute`` through in a minute.
        max_retries: Retries after the first attempt for retryable failures.
        backoff_s: Base of the exponential backoff (``backoff_s * 2**attempt``).
        timeout_s: Per-request timeout.
        transport: httpx transport (tests pass ``httpx.MockTransport``).
        clock: Monotonic clock (tests).
        sleep: Sleep function (tests).
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        base_url: str = BASE_URL,
        rate_per_s: float = 3.0,
        burst: float = 1.0,
        per_minute: float = 29.0,
        max_retries: int = 5,
        backoff_s: float = 1.0,
        timeout_s: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cache_dir = cache_dir
        self.base_url = base_url
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self._sleep = sleep
        self._buckets = (
            TokenBucket(rate_per_s, burst, clock=clock, sleep=sleep),
            TokenBucket(per_minute / 60.0, 1.0, clock=clock, sleep=sleep),
        )
        self._http = httpx.Client(timeout=timeout_s, transport=transport)
        #: Number of HTTP requests actually sent (retries included).
        self.network_calls = 0
        #: Number of queries answered from the disk cache.
        self.cache_hits = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the HTTP connection pool."""
        self._http.close()

    # -- raw access ---------------------------------------------------------------------------

    def cache_path(self, url: str) -> Path:
        """Cache file for a canonical URL."""
        h = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / h[:2] / f"{h}.json"

    def get(
        self,
        endpoint: str,
        params: Mapping[str, Scalar] | None = None,
        filters: Sequence[Filter] = (),
    ) -> list[Record]:
        """Query an endpoint, from the cache if possible.

        Raises:
            ValueError: Unknown endpoint.
            DataUnavailableError: Non-retryable HTTP error, retries exhausted, or a response
                that is not a JSON list of objects.
        """
        if endpoint not in ENDPOINTS:
            known = ", ".join(sorted(ENDPOINTS))
            raise ValueError(f"unknown OpenF1 endpoint {endpoint!r}; use one of {known}")
        url = build_url(self.base_url, endpoint, params, filters)
        path = self.cache_path(url)
        if path.is_file():
            self.cache_hits += 1
            cached = json.loads(path.read_text(encoding="utf-8"))
            return _records(cached["data"], url)
        data = self._fetch(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        payload = {"url": url, "fetched_at_utc": datetime.now(UTC).isoformat(), "data": data}
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)
        return data

    def _fetch(self, url: str) -> list[Record]:
        last_error = ""
        for attempt in range(self.max_retries + 1):
            for bucket in self._buckets:
                bucket.acquire()
            self.network_calls += 1
            retry_after = 0.0
            try:
                resp = self._http.get(url)
            except httpx.TransportError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code == 200:
                    return _records(resp.json(), url)
                if resp.status_code == 404 and _is_no_results(resp):
                    return []
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if resp.status_code not in _RETRY_STATUS:
                    raise DataUnavailableError(f"OpenF1 {url}: {last_error}")
                retry_after = _retry_after_s(resp)
            if attempt < self.max_retries:
                wait = min(max(retry_after, self.backoff_s * 2**attempt), _MAX_WAIT_S)
                log.warning("openf1_retry", url=url, attempt=attempt + 1, wait_s=wait,
                            error=last_error)
                self._sleep(wait)
        raise DataUnavailableError(
            f"OpenF1 {url}: giving up after {self.max_retries + 1} attempts ({last_error})"
        )

    # -- typed helpers ------------------------------------------------------------------------

    def find_session(self, year: int, event: str, session: str) -> OpenF1Session:
        """Resolve (year, event query, session) to an OpenF1 session.

        ``event`` matches (case-insensitively, as a substring) the meeting name, official name,
        country, location or circuit short name. An exact match on any of them wins over
        substring matches.

        Raises:
            DataUnavailableError: No match, or an ambiguous event query.
        """
        code = normalize_session(session)
        q = event.casefold().strip()
        fields = ("meeting_name", "meeting_official_name", "country_name", "location",
                  "circuit_short_name")
        meetings = self.get("meetings", {"year": year})

        def values(m: Record) -> list[str]:
            return [str(m.get(f, "")).casefold() for f in fields]

        exact = [m for m in meetings if q in values(m)]
        matches = exact or [m for m in meetings if any(q in v for v in values(m))]
        if not matches:
            raise DataUnavailableError(f"OpenF1: no {year} meeting matching {event!r}")
        if len(matches) > 1:
            names = ", ".join(sorted(str(m.get("meeting_name")) for m in matches))
            raise DataUnavailableError(f"OpenF1: {event!r} is ambiguous in {year}: {names}")
        meeting = matches[0]
        sessions = self.get("sessions", {"meeting_key": _int(meeting, "meeting_key")})
        for s in sessions:
            try:
                s_code = normalize_session(str(s.get("session_name", "")))
            except ValueError:
                continue
            if s_code == code:
                return _session(meeting, s, s_code)
        raise DataUnavailableError(f"OpenF1: {meeting.get('meeting_name')} {year} has no {code}")

    def session_by_key(self, session_key: int) -> OpenF1Session:
        """Look up an OpenF1 ``session_key`` and map it to its FastF1 identity.

        Raises:
            DataUnavailableError: Unknown key, or a session type FastF1 does not have.
        """
        sessions = self.get("sessions", {"session_key": session_key})
        if len(sessions) != 1:
            raise DataUnavailableError(f"OpenF1: session_key {session_key} not found")
        s = sessions[0]
        meetings = self.get("meetings", {"meeting_key": _int(s, "meeting_key")})
        if len(meetings) != 1:
            raise DataUnavailableError(f"OpenF1: meeting of session_key {session_key} not found")
        try:
            code = normalize_session(str(s.get("session_name", "")))
        except ValueError as exc:
            raise DataUnavailableError(f"OpenF1: session_key {session_key}: {exc}") from exc
        return _session(meetings[0], s, code)

    def laps(self, session_key: int, driver_number: int | None = None) -> list[Record]:
        """Lap records of a session (optionally one driver)."""
        params: dict[str, Scalar] = {"session_key": session_key}
        if driver_number is not None:
            params["driver_number"] = driver_number
        return self.get("laps", params)

    def drivers(self, session_key: int) -> list[Record]:
        """Driver records of a session (``driver_number``, ``name_acronym``, ``team_name``...)."""
        return self.get("drivers", {"session_key": session_key})

    def stints(self, session_key: int) -> list[Record]:
        """Tyre stints of a session (``lap_start``, ``lap_end``, ``compound``...)."""
        return self.get("stints", {"session_key": session_key})

    def location(
        self, session_key: int, driver_number: int, start: str | None = None,
        end: str | None = None,
    ) -> list[Record]:
        """Position samples of one driver, optionally in ``[start, end)`` (ISO timestamps)."""
        return self.get("location", *_window(session_key, driver_number, start, end))

    def car_data(
        self, session_key: int, driver_number: int, start: str | None = None,
        end: str | None = None,
    ) -> list[Record]:
        """Car telemetry samples of one driver, optionally in ``[start, end)``."""
        return self.get("car_data", *_window(session_key, driver_number, start, end))


def _window(
    session_key: int, driver_number: int, start: str | None, end: str | None
) -> tuple[dict[str, Scalar], list[Filter]]:
    filters: list[Filter] = []
    if start is not None:
        filters.append(("date", ">=", start))
    if end is not None:
        filters.append(("date", "<", end))
    return {"session_key": session_key, "driver_number": driver_number}, filters


def _records(data: object, url: str) -> list[Record]:
    if not isinstance(data, list) or not all(isinstance(r, dict) for r in data):
        raise DataUnavailableError(f"OpenF1 {url}: expected a JSON list of objects")
    return cast(list[Record], data)


def _is_no_results(resp: httpx.Response) -> bool:
    try:
        body = resp.json()
    except ValueError:
        return False
    return isinstance(body, dict) and body.get("detail") == "No results found."


def _retry_after_s(resp: httpx.Response) -> float:
    try:
        return max(float(resp.headers.get("Retry-After", "0")), 0.0)
    except ValueError:
        return 0.0


def _int(r: Record, key: str) -> int:
    v = r.get(key)
    if isinstance(v, bool) or not isinstance(v, int):
        raise DataUnavailableError(f"OpenF1: field {key!r} is not an int: {v!r}")
    return v


def _session(meeting: Record, s: Record, code: str) -> OpenF1Session:
    return OpenF1Session(
        session_key=_int(s, "session_key"),
        meeting_key=_int(meeting, "meeting_key"),
        year=_int(s, "year"),
        meeting_name=str(meeting.get("meeting_name", "")),
        session_name=str(s.get("session_name", "")),
        session_code=code,
        circuit_short_name=str(s.get("circuit_short_name", "")),
        country_name=str(s.get("country_name", "")),
        location=str(s.get("location", "")),
        date_start=str(s.get("date_start", "")),
    )
