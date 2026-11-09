"""The history is stamped in the past: the API reads only the events before the
request, by its own clock, so every event, of every session, must be older."""

import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from behaviorgpt import Order, Search, UnboxAIClient, View

PAGE = {"products": {"items": [], "offset": 0, "limit": 10}}


def sent(history) -> list[dict]:
    client = UnboxAIClient("US", api_key="test", base_url="http://sdk.test/v1")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=PAGE)

    client._http_client._transport = httpx.MockTransport(handler)
    client.complete(history)
    return json.loads(requests[0].content)["history"]


def times(events: list[dict]) -> list[datetime]:
    return [
        datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00")) for e in events
    ]


@pytest.mark.parametrize(
    "history",
    [
        [View(f"i{k}") for k in range(1500)],
        [[View("a"), View("b")], [View("c"), Order(["c"])], [View("d"), View("e")]],
        [[View(f"i{k}") for k in range(1200)], [View("x"), View("y")]],
    ],
    ids=["one long session", "three sessions", "a session longer than a second"],
)
def test_every_event_is_before_now_and_in_order(history):
    before = datetime.now(tz=UTC)
    events = sent(history)
    stamped = times(events)
    assert max(stamped) <= before - timedelta(seconds=1) + timedelta(milliseconds=50)
    assert stamped == sorted(stamped) and len(set(stamped)) == len(stamped)


def test_sessions_keep_their_own_ids_and_order():
    events = sent([[View("a"), View("b")], [View("c")], [Search("sofa"), View("d")]])
    ids = [e["session_id"] for e in events]
    assert len(set(ids)) == 3
    assert [e["event"].get("product") for e in events] == ["a", "b", "c", None, "d"]


# Events with their own timestamps keep them, and `timestamp=` is the request's.

T = datetime(2022, 3, 1, 8, tzinfo=UTC)


def request(history, **kwargs) -> dict:
    client = UnboxAIClient("US", api_key="test", base_url="http://sdk.test/v1")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=PAGE)

    client._http_client._transport = httpx.MockTransport(handler)
    client.complete(history, **kwargs)
    return json.loads(requests[0].content)


def test_events_keep_their_timestamps_and_the_request_its_time():
    history = [
        [View("a", timestamp=T), View("b", timestamp=T + timedelta(minutes=2))],
        [Order(["b"], timestamp=T + timedelta(days=3))],
    ]
    body = request(history, timestamp=T + timedelta(days=4))
    assert times(body["history"]) == [
        T,
        T + timedelta(minutes=2),
        T + timedelta(days=3),
    ]
    assert times([body]) == [T + timedelta(days=4)]
    assert "timestamp" not in body["history"][0]["event"]
    assert len({e["session_id"] for e in body["history"]}) == 2


def test_a_search_with_a_timestamp_goes_to_search_at_that_time():
    body = request(
        [View("a", timestamp=T), Search("sofa", timestamp=T + timedelta(seconds=5))],
        timestamp=T + timedelta(seconds=6),
    )
    assert body["query"] == "sofa"
    assert times(body["history"]) == [T]


def test_a_history_without_timestamps_ends_before_the_request_time():
    body = request([[View("a"), View("b")], [View("c")]], timestamp=T)
    stamped = times(body["history"])
    assert max(stamped) <= T - timedelta(seconds=1)
    assert stamped == sorted(stamped)


def test_without_timestamp_the_request_leaves_the_time_to_the_server():
    assert request([View("a", timestamp=T)]).get("timestamp") is None


@pytest.mark.parametrize(
    "history, kwargs, message",
    [
        ([View("a", timestamp=T), View("b")], {}, "every event"),
        (
            [View("a", timestamp=T), View("b", timestamp=T - timedelta(1))],
            {},
            "in order",
        ),
        ([View("a", timestamp=T)], {"timestamp": T}, "before the request"),
        ([View("a")], {"timestamp": datetime(2022, 3, 1)}, "timezone-aware"),
    ],
    ids=["some timed", "out of order", "not before the request", "naive request time"],
)
def test_timestamps_that_cannot_be_read_raise(history, kwargs, message):
    with pytest.raises(ValueError, match=message):
        request(history, **kwargs)


def test_an_event_timestamp_must_be_timezone_aware():
    with pytest.raises(ValueError, match="timezone-aware"):
        View("a", timestamp=datetime(2022, 3, 1))
