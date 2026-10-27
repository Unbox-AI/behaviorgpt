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
