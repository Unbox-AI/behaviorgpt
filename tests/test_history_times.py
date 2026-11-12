"""The history is stamped in the past: the API reads only the events before the
request's timestamp, so every event, of every session, must be older. The
request's timestamp comes from the same clock reading as the history's, so a
local clock that is off from the API's changes nothing."""

import json
from datetime import UTC, datetime, timedelta
from typing import Any, Optional

import httpx
import pytest

from behaviorgpt import Order, Search, UnboxAIClient, View

PAGE = {"products": {"items": [], "offset": 0, "limit": 10}}
# How far the local clock ran ahead of the API's in the bug report
AHEAD = timedelta(seconds=5)


def body(history, user_id: Optional[str] = None) -> dict[str, Any]:
    client = UnboxAIClient("US", api_key="test", base_url="http://sdk.test/v1")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=PAGE)

    client._http_client._transport = httpx.MockTransport(handler)
    client.complete(history, user_id=user_id)
    return json.loads(requests[0].content)


def sent(history) -> list[dict]:
    return body(history)["history"]


def parse(timestamp: str) -> datetime:
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


def times(events: list[dict]) -> list[datetime]:
    return [parse(e["timestamp"]) for e in events]


class ClockAhead(datetime):
    """This computer's clock, running AHEAD of the API's."""

    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        return datetime.now(tz) + AHEAD


@pytest.mark.parametrize(
    "history",
    [
        [View(f"i{k}") for k in range(1500)],
        [[View("a"), View("b")], [View("c"), Order(["c"])], [View("d"), View("e")]],
        [[View(f"i{k}") for k in range(1200)], [View("x"), View("y")]],
    ],
    ids=["one long session", "three sessions", "a session longer than a second"],
)
def test_every_event_is_before_the_request_and_in_order(history):
    sent_body = body(history)
    stamped = times(sent_body["history"])
    assert max(stamped) <= parse(sent_body["timestamp"]) - timedelta(seconds=1)
    assert stamped == sorted(stamped) and len(set(stamped)) == len(stamped)


def test_sessions_keep_their_own_ids_and_order():
    events = sent([[View("a"), View("b")], [View("c")], [Search("sofa"), View("d")]])
    ids = [e["session_id"] for e in events]
    assert len(set(ids)) == 3
    assert [e["event"].get("product") for e in events] == ["a", "b", "c", None, "d"]


@pytest.mark.parametrize(
    "history",
    [[View("a"), View("b")], [View("a"), Search("sofa")]],
    ids=["recommendations", "search"],
)
def test_a_history_goes_with_a_timestamp_from_the_same_clock(history, monkeypatch):
    monkeypatch.setattr("behaviorgpt._client.datetime", ClockAhead)
    before = datetime.now(tz=UTC) + AHEAD

    sent_body = body(history)

    timestamp = parse(sent_body["timestamp"])
    assert before <= timestamp <= datetime.now(tz=UTC) + AHEAD
    assert max(times(sent_body["history"])) < timestamp


@pytest.mark.parametrize(
    ("history", "user_id"),
    [([], None), ([Search("sofa")], None), ([], "u1"), ([Search("sofa")], "u1")],
    ids=["nothing", "only a search", "a user", "a user's search"],
)
def test_without_history_events_no_timestamp_is_sent(history, user_id):
    assert "timestamp" not in body(history, user_id)
