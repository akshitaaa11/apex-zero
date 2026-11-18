from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from apex_zero.data.cache import SessionKey
from apex_zero.data.openf1_client import BASE_URL, OpenF1Client, TokenBucket, build_url
from apex_zero.errors import DataUnavailableError
from apex_zero.utils.units import OPENF1_POS_TO_M

SUZUKA_OFFICIAL_M = 5807.0  # FIA/F1 official lap length of Suzuka (5.807 km)

MEETINGS = [
    {"meeting_key": 1232, "meeting_name": "Japanese Grand Prix",
     "meeting_official_name": "FORMULA 1 JAPANESE GRAND PRIX 2024", "country_name": "Japan",
     "location": "Suzuka", "circuit_short_name": "Suzuka", "year": 2024},
    {"meeting_key": 1229, "meeting_name": "Australian Grand Prix",
     "meeting_official_name": "FORMULA 1 AUSTRALIAN GRAND PRIX 2024",
     "country_name": "Australia", "location": "Melbourne", "circuit_short_name": "Melbourne",
     "year": 2024},
    {"meeting_key": 1240, "meeting_name": "Miami Grand Prix",
     "meeting_official_name": "FORMULA 1 MIAMI GRAND PRIX 2024", "country_name": "United States",
     "location": "Miami", "circuit_short_name": "Miami", "year": 2024},
    {"meeting_key": 1250, "meeting_name": "United States Grand Prix",
     "meeting_official_name": "FORMULA 1 UNITED STATES GRAND PRIX 2024",
     "country_name": "United States", "location": "Austin", "circuit_short_name": "Austin",
     "year": 2024},
]
SESSIONS = [
    {"session_key": 9492, "session_name": "Qualifying", "meeting_key": 1232, "year": 2024,
     "circuit_short_name": "Suzuka", "country_name": "Japan", "location": "Suzuka",
     "date_start": "2024-04-06T06:00:00+00:00"},
    {"session_key": 9496, "session_name": "Race", "meeting_key": 1232, "year": 2024,
     "circuit_short_name": "Suzuka", "country_name": "Japan", "location": "Suzuka",
     "date_start": "2024-04-07T05:00:00+00:00"},
    {"session_key": 9999, "session_name": "Day 1", "meeting_key": 1232, "year": 2024,
     "circuit_short_name": "Suzuka", "country_name": "Japan", "location": "Suzuka",
     "date_start": "2024-04-01T05:00:00+00:00"},
]


class FakeAPI:
    """Minimal OpenF1 stand-in: filters by equality params; scripted failures per URL."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.failures: list[httpx.Response | Exception] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.failures:
            f = self.failures.pop(0)
            if isinstance(f, Exception):
                raise f
            return f
        endpoint = request.url.path.rsplit("/", 1)[-1]
        rows: list[dict[str, Any]] = {"meetings": MEETINGS, "sessions": SESSIONS}.get(
            endpoint, [{"lap_number": 1, "session_key": 9492, "driver_number": 1}]
        )
        for k, v in request.url.params.items():
            rows = [r for r in rows if str(r.get(k)) == v]
        if not rows:
            return httpx.Response(404, json={"detail": "No results found."})
        return httpx.Response(200, json=rows)


@pytest.fixture
def api() -> FakeAPI:
    return FakeAPI()


@pytest.fixture
def sleeps() -> list[float]:
    return []


@pytest.fixture
def make_client(
    tmp_path: Path, api: FakeAPI, sleeps: list[float]
) -> Callable[..., OpenF1Client]:
    def make(**kwargs: Any) -> OpenF1Client:
        clock = iter(float(i) * 100 for i in range(10**6))  # far apart: bucket never empty
        return OpenF1Client(
            tmp_path / "openf1",
            transport=httpx.MockTransport(api),
            clock=lambda: next(clock),
            sleep=sleeps.append,
            **kwargs,
        )

    return make


def test_build_url_is_canonical() -> None:
    a = build_url(BASE_URL, "location", {"session_key": 9492, "driver_number": 1},
                  [("date", ">=", "2024-04-06T06:58:14+00:00")])
    b = build_url(BASE_URL + "/", "location", {"driver_number": 1, "session_key": 9492},
                  [("date", ">=", "2024-04-06T06:58:14+00:00")])
    assert a == b
    assert a == (
        "https://api.openf1.org/v1/location?date%3E%3D2024-04-06T06%3A58%3A14%2B00%3A00"
        "&driver_number=1&session_key=9492"
    )
    assert build_url(BASE_URL, "laps", {"is_pit_out_lap": False}).endswith("=false")
    assert build_url(BASE_URL, "meetings") == "https://api.openf1.org/v1/meetings"


def test_token_bucket_limits_rate() -> None:
    now = [0.0]
    slept: list[float] = []

    def sleep(dt: float) -> None:
        slept.append(dt)
        now[0] += dt

    bucket = TokenBucket(3.0, 3.0, clock=lambda: now[0], sleep=sleep)
    for _ in range(9):
        bucket.acquire()
    # 3 burst tokens free, then 6 more at 3/s -> 2 s of waiting in total
    assert sum(slept) == pytest.approx(2.0)
    assert now[0] / 9 <= 1 / 3 + 1e-9  # never faster than the burst allows
    now[0] += 10.0
    assert bucket.acquire() == 0.0  # refilled (capped at capacity)
    with pytest.raises(ValueError):
        TokenBucket(0.0, 1.0)


def test_cache_hit_makes_no_request(make_client: Callable[..., OpenF1Client], api: FakeAPI) -> None:
    with make_client() as c:
        first = c.laps(9492, 1)
        assert (c.network_calls, c.cache_hits) == (1, 0)
    with make_client() as c2:  # new client, same disk cache
        assert c2.laps(9492, 1) == first
        assert (c2.network_calls, c2.cache_hits) == (0, 1)
    assert len(api.requests) == 1
    cached = json.loads(c2.cache_path(api.requests[0].url.__str__()).read_text())
    assert cached["url"] == str(api.requests[0].url)


def test_no_results_is_empty_and_cached(
    make_client: Callable[..., OpenF1Client], api: FakeAPI
) -> None:
    with make_client() as c:
        assert c.get("sessions", {"session_key": 1}) == []
        assert c.get("sessions", {"session_key": 1}) == []
    assert len(api.requests) == 1


def test_retries_with_backoff_and_retry_after(
    make_client: Callable[..., OpenF1Client], api: FakeAPI, sleeps: list[float]
) -> None:
    api.failures = [
        httpx.Response(429, headers={"Retry-After": "7"}),
        httpx.Response(503),
        httpx.ConnectError("reset"),
    ]
    with make_client(backoff_s=0.5) as c:
        assert c.laps(9492) != []
        assert c.network_calls == 4
    assert sleeps == [7.0, 1.0, 2.0]  # max(Retry-After, 0.5*2**k)


def test_gives_up_after_max_retries(
    make_client: Callable[..., OpenF1Client], api: FakeAPI
) -> None:
    api.failures = [httpx.Response(500, headers={"Retry-After": "x"})] * 3
    with make_client(max_retries=2) as c, pytest.raises(DataUnavailableError, match="3 attempts"):
        c.laps(9492)
    assert len(api.requests) == 3
    assert not any((c.cache_dir).rglob("*.json"))  # failures are not cached


@pytest.mark.parametrize(
    "response",
    [httpx.Response(400, text="bad"), httpx.Response(404, text="not json"),
     httpx.Response(404, json={"detail": "Not Found"}), httpx.Response(200, json={"a": 1}),
     httpx.Response(200, json=[1, 2])],
)
def test_non_retryable_and_malformed(
    make_client: Callable[..., OpenF1Client], api: FakeAPI, response: httpx.Response
) -> None:
    api.failures = [response]
    with make_client() as c, pytest.raises(DataUnavailableError):
        c.laps(9492)
    assert len(api.requests) == 1


def test_unknown_endpoint(make_client: Callable[..., OpenF1Client]) -> None:
    with make_client() as c, pytest.raises(ValueError, match="unknown OpenF1 endpoint"):
        c.get("weather")


@pytest.mark.parametrize("query", ["Japan", "suzuka", "Japanese Grand Prix", "JAPANESE"])
def test_find_session_maps_to_fastf1_identity(
    make_client: Callable[..., OpenF1Client], query: str
) -> None:
    with make_client() as c:
        s = c.find_session(2024, query, "qualifying")
    assert (s.session_key, s.session_code, s.meeting_name) == (9492, "Q", "Japanese Grand Prix")
    assert s.cache_key == SessionKey(2024, "japanese_grand_prix", "Q")


def test_find_session_exact_match_beats_substring(
    make_client: Callable[..., OpenF1Client],
) -> None:
    with make_client() as c:
        # "United States" is the exact country of both Miami and Austin -> ambiguous
        with pytest.raises(DataUnavailableError, match="ambiguous"):
            c.find_session(2024, "United States", "R")
        with pytest.raises(DataUnavailableError, match="no 2024 meeting"):
            c.find_session(2024, "Monaco", "R")
        with pytest.raises(DataUnavailableError, match="has no FP1"):
            c.find_session(2024, "Japan", "FP1")


def test_session_by_key(make_client: Callable[..., OpenF1Client]) -> None:
    with make_client() as c:
        s = c.session_by_key(9496)
        assert (s.session_code, s.cache_key.relpath.as_posix()) == ("R", "2024/japanese_grand_prix/R")
        with pytest.raises(DataUnavailableError, match="not found"):
            c.session_by_key(1)
        with pytest.raises(DataUnavailableError, match="unknown session"):
            c.session_by_key(9999)


def test_window_helpers_send_filters(
    make_client: Callable[..., OpenF1Client], api: FakeAPI
) -> None:
    with make_client() as c:
        c.location(9492, 1, "2024-04-06T06:58:14", "2024-04-06T06:59:43")
        c.car_data(9492, 1)
    q0 = str(api.requests[0].url)
    assert "/location?" in q0 and "date%3E%3D2024" in q0 and "date%3C2024" in q0
    assert str(api.requests[1].url).endswith("/car_data?driver_number=1&session_key=9492")


def lap_polyline_length_m(real_lap: dict[str, Any]) -> float:
    """Closed polyline length of the samples inside [lap start, lap start + duration]."""
    start = datetime.fromisoformat(real_lap["lap"]["date_start"])
    end = start + timedelta(seconds=real_lap["lap"]["lap_duration"])
    pts = [
        (p["x"], p["y"])
        for p in real_lap["location"]
        if start <= datetime.fromisoformat(p["date"]) <= end
    ]
    closed = [*pts, pts[0]]
    return OPENF1_POS_TO_M * sum(math.dist(a, b) for a, b in zip(closed, closed[1:]))


def test_real_lap_fixture_pins_openf1_units(real_lap: dict[str, Any]) -> None:
    """Offline twin of tests/network/test_units.py on the committed real lap."""
    assert abs(lap_polyline_length_m(real_lap) / SUZUKA_OFFICIAL_M - 1) < 0.02
    assert real_lap["lap"]["lap_duration"] == pytest.approx(88.197)
