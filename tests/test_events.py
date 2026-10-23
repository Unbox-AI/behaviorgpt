"""Tracked events: `events.track` and `events.list`, and `user_id=` on
`complete` and `sections`, which read the history from them."""

import asyncio
import json
from datetime import UTC, datetime

import httpx
import pytest
from pydantic import ValidationError

from behaviorgpt import AsyncUnboxAIClient, Order, Search, UnboxAIClient, View
from behaviorgpt.types import UnboxAIRequest
from behaviorgpt.types.domains import Domains
from behaviorgpt.types.events import SessionEvent

BASE = "http://sdk.test/v1"
MOMENT = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
PAGE = {"products": {"items": [], "offset": 0, "limit": 10}}
LISTED = {
    "events": [
        {
            "user_id": "u1",
            "timestamp": "2026-10-23T12:03:00Z",
            "store_id": "cat",
            "event": {"type": "viewItem", "product": "p3"},
            "domains": {"market": "US"},
            "timezone": "UTC",
        }
    ]
}


def handler_for(requests: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/events":
            return httpx.Response(201, json={"id": "ev-1"})
        if request.url.path == "/v1/events/own":
            return httpx.Response(200, json=LISTED)
        return httpx.Response(200, json=PAGE)

    return handler


def mock_client() -> tuple[UnboxAIClient, list[httpx.Request]]:
    requests: list[httpx.Request] = []
    client = UnboxAIClient("US", api_key="test", base_url=BASE)
    client._http_client._transport = httpx.MockTransport(handler_for(requests))
    return client, requests


def body(request: httpx.Request) -> dict:
    return json.loads(request.content)


def test_track_posts_the_event_and_returns_its_id():
    client, requests = mock_client()
    event_id = client.events.track("u1", View("sku-1"), timestamp=MOMENT)

    assert event_id == "ev-1"
    assert requests[0].method == "POST" and requests[0].url.path == "/v1/events"
    assert body(requests[0]) == {
        "user_id": "u1",
        "store_id": "retail_catalog",
        "event": {"type": "viewItem", "product": "sku-1"},
        "domains": {
            "market": "US",
            "postal_code": "unk",
            "gender": "unk",
            "age_group": "unk",
        },
        "timezone": "UTC",
        "timestamp": "2026-10-23T12:00:00+00:00",
    }


def test_track_sends_searches_and_the_catalog():
    client, requests = mock_client()
    client.events.track("u1", Search("sofa"), catalog_id="cat")
    sent = body(requests[0])
    assert sent["event"] == {"type": "search", "query": "sofa"}
    assert sent["store_id"] == "cat"
    assert datetime.fromisoformat(sent["timestamp"]).tzinfo is not None


def test_track_sends_orders_as_the_api_stores_them():
    client, requests = mock_client()
    client.events.track("u1", Order(["a", ("b", 2)], cart_id="cart-7"))
    assert body(requests[0])["event"] == {
        "type": "order",
        "cart_id": "cart-7",
        "items": [
            {"product": "a", "quantity": 1, "unit_selling_price": 0.0},
            {"product": "b", "quantity": 2, "unit_selling_price": 0.0},
        ],
    }


def test_track_gives_an_order_a_cart_id():
    client, requests = mock_client()
    client.events.track("u1", Order(["a"]))
    assert body(requests[0])["event"]["cart_id"]


def test_track_refuses_an_order_without_items():
    client, requests = mock_client()
    with pytest.raises(ValueError, match="items bought"):
        client.events.track("u1", Order([]))
    assert requests == []


def test_track_refuses_a_naive_timestamp():
    client, _ = mock_client()
    with pytest.raises(ValueError, match="timezone-aware"):
        client.events.track("u1", View("a"), timestamp=datetime(2026, 10, 23))


def test_list_reads_the_accounts_events():
    client, requests = mock_client()
    events = client.events.list(user_id="u1", limit=10, types=["viewItem", "search"])

    assert requests[0].method == "GET" and requests[0].url.path == "/v1/events/own"
    assert requests[0].url.params.multi_items() == [
        ("limit", "10"),
        ("user_id", "u1"),
        ("type", "viewItem"),
        ("type", "search"),
    ]
    assert [(ev.user_id, ev.event) for ev in events] == [
        ("u1", {"type": "viewItem", "product": "p3"})
    ]
    assert events[0].timestamp == datetime(2026, 10, 23, 12, 3, tzinfo=UTC)


@pytest.mark.parametrize("limit", [0, 201])
def test_list_refuses_a_limit_out_of_range(limit):
    client, _ = mock_client()
    with pytest.raises(ValueError, match="limit"):
        client.events.list(limit=limit)


def test_complete_for_a_user_sends_the_user_and_no_history():
    client, requests = mock_client()
    client.complete(user_id="u1", catalog_id="cat")
    sent = body(requests[0])
    assert requests[0].url.path == "/v1/vector/recommendations"
    assert sent["user_id"] == "u1"
    assert "history" not in sent


def test_search_for_a_user_is_recorded_by_default():
    client, requests = mock_client()
    client.complete([Search("sofa")], user_id="u1")
    sent = body(requests[0])
    assert requests[0].url.path == "/v1/vector/search"
    assert sent["query"] == "sofa" and sent["user_id"] == "u1"
    assert "history" not in sent
    assert sent["register_event"] is True


def test_search_for_a_user_can_be_left_unrecorded():
    client, requests = mock_client()
    client.complete([Search("sofa")], user_id="u1", register_event=False)
    assert body(requests[0])["register_event"] is False


def test_without_a_user_nothing_is_recorded_by_default():
    client, requests = mock_client()
    client.complete([View("a"), Search("sofa")])
    sent = body(requests[0])
    assert sent["register_event"] is False
    assert "user_id" not in sent


@pytest.mark.parametrize(
    "history",
    [[View("a")], [Search("a"), Search("b")], [[View("a")], [Search("b")]]],
)
def test_a_user_with_more_history_than_the_current_search_is_refused(history):
    client, requests = mock_client()
    with pytest.raises(ValueError, match="events.track"):
        client.complete(history, user_id="u1")
    assert requests == []


def test_sections_for_a_user_send_the_user():
    client, requests = mock_client()

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"sections": []})

    client._http_client._transport = httpx.MockTransport(handler)
    client.sections(user_id="u1")
    sent = body(requests[0])
    assert requests[0].url.path == "/v1/vector/sections"
    assert sent["user_id"] == "u1" and "history" not in sent


def test_request_refuses_a_user_and_a_history():
    with pytest.raises(ValidationError, match="not both"):
        UnboxAIRequest(
            user_id="u1",
            store_id="cat",
            domains={"market": "US"},
            history=[
                SessionEvent(
                    store_id="cat",
                    domains=Domains(market="US"),
                    timezone="UTC",
                    event=Search("a"),
                    session_id="s",
                )
            ],
        )


def test_async_client_tracks_lists_and_completes_for_a_user():
    requests: list[httpx.Request] = []

    async def run() -> None:
        async with AsyncUnboxAIClient("US", api_key="test", base_url=BASE) as client:
            client._http_client._transport = httpx.MockTransport(handler_for(requests))
            assert await client.events.track("u1", View("a")) == "ev-1"
            assert len(await client.events.list()) == 1
            await client.complete([Search("sofa")], user_id="u1")

    asyncio.run(run())
    assert [r.url.path for r in requests] == [
        "/v1/events",
        "/v1/events/own",
        "/v1/vector/search",
    ]
    assert body(requests[2])["user_id"] == "u1"
