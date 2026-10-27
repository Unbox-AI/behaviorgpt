"""Tracked events: `events.track` records them, `events.list` reads them back."""

import asyncio
import json
from datetime import UTC, datetime

import httpx
import pytest

from behaviorgpt import AsyncUnboxAIClient, Order, Search, UnboxAIClient, User, View
from behaviorgpt.types import Item

BASE = "http://sdk.test/v1"
MOMENT = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
LISTED = {
    "events": [
        {
            "user_id": "u1",
            "timestamp": "2026-10-23T12:03:00Z",
            "store_id": "cat",
            "stream": "api",
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
        return httpx.Response(200, json=LISTED)

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


def test_track_takes_the_catalog_an_item_names():
    client, requests = mock_client()
    item = Item(id="r1", data={}, score=1.0, catalog_id="other")
    client.events.track("u1", View(item))
    client.events.track("u1", Order([item]))
    client.events.track("u1", View(item), catalog_id="cat")
    assert [body(r)["store_id"] for r in requests] == ["other", "other", "cat"]


def test_track_refuses_an_order_over_several_catalogs():
    client, requests = mock_client()
    a = Item(id="a", data={}, score=1.0, catalog_id="one")
    b = Item(id="b", data={}, score=1.0, catalog_id="two")
    with pytest.raises(ValueError, match="one Order per catalog"):
        client.events.track("u1", Order([a, b]))
    assert requests == []


def test_track_sends_who_the_user_is():
    client, requests = mock_client()
    shopper = User(gender="female", age=34, market="SE", timezone="Europe/Stockholm")
    client.events.track("u1", View("a"), user=shopper)
    sent = body(requests[0])
    assert sent["domains"] == {
        "market": "SE",
        "postal_code": "unk",
        "gender": "female",
        "age_group": "26-35",
    }
    assert sent["timezone"] == "Europe/Stockholm"


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
    events = client.events.list(
        user_id="u1", limit=10, types=["viewItem", "search"], stream="api"
    )

    assert requests[0].method == "GET" and requests[0].url.path == "/v1/events/own"
    assert requests[0].url.params.multi_items() == [
        ("limit", "10"),
        ("user_id", "u1"),
        ("type", "viewItem"),
        ("type", "search"),
        ("stream", "api"),
    ]
    assert [(ev.user_id, ev.stream, ev.event) for ev in events] == [
        ("u1", "api", {"type": "viewItem", "product": "p3"})
    ]
    assert events[0].timestamp == datetime(2026, 10, 23, 12, 3, tzinfo=UTC)


def test_list_sends_only_the_limit_by_default():
    client, requests = mock_client()
    client.events.list()
    assert requests[0].url.params.multi_items() == [("limit", "50")]


@pytest.mark.parametrize("limit", [0, 201])
def test_list_refuses_a_limit_out_of_range(limit):
    client, _ = mock_client()
    with pytest.raises(ValueError, match="limit"):
        client.events.list(limit=limit)


def test_async_client_tracks_and_lists():
    requests: list[httpx.Request] = []

    async def run() -> None:
        async with AsyncUnboxAIClient("US", api_key="test", base_url=BASE) as client:
            client._http_client._transport = httpx.MockTransport(handler_for(requests))
            assert await client.events.track("u1", View("a")) == "ev-1"
            assert len(await client.events.list(stream="api")) == 1

    asyncio.run(run())
    assert [r.url.path for r in requests] == ["/v1/events", "/v1/events/own"]
    assert requests[1].url.params["stream"] == "api"
