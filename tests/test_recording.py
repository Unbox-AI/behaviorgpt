"""What a call made for a `user_id` asks the API to record for them: its
search as their event, and the call itself (`events.calls`), unless
`register_event=False`. Without a user_id nothing is recorded."""

import asyncio
import json
from datetime import UTC, datetime

import httpx
import pytest
from pydantic import ValidationError

from behaviorgpt import AsyncUnboxAIClient, Search, UnboxAIClient, View
from behaviorgpt.types import Domains, SessionEvent, UnboxAIRequest

BASE = "http://sdk.test/v1"
PAGE = {"products": {"items": [], "offset": 0, "limit": 10}}
SECTIONS = {"sections": []}
CALLS = {
    "calls": [
        {
            "user_id": "u1",
            "timestamp": "2026-10-28T12:03:00Z",
            "call": "search",
            "store_id": "cat",
            "query": "sofa",
            "offset": 0,
            "items": [{"id": "p1", "store_id": "cat"}],
            "stream": "api",
        }
    ]
}


def mock(client) -> list[httpx.Request]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/calls"):
            return httpx.Response(200, json=CALLS)
        body = SECTIONS if request.url.path.endswith("/sections") else PAGE
        return httpx.Response(200, json=body)

    client._http_client._transport = httpx.MockTransport(handler)
    return requests


def client() -> tuple[UnboxAIClient, list[httpx.Request]]:
    c = UnboxAIClient("US", api_key="test", base_url=BASE)
    return c, mock(c)


def sent(request: httpx.Request) -> dict:
    return json.loads(request.content)


def recording(body: dict) -> tuple:
    """What a request body says about recording."""
    return (
        body.get("user_id"),
        "history" in body,
        body["register_event"],
        body.get("record_call"),
        body.get("call"),
    )


def test_a_search_for_a_user_is_recorded_by_default():
    c, requests = client()
    c.complete([Search("sofa")], user_id="u1")

    assert requests[0].url.path == "/v1/vector/search"
    # record_call is left to the API, which records by default
    assert recording(sent(requests[0])) == ("u1", False, True, None, None)


def test_recommendations_for_a_user_are_recorded_by_default():
    c, requests = client()
    c.complete(user_id="u1")
    c.sections(user_id="u1")

    assert [recording(sent(r)) for r in requests] == [
        ("u1", False, True, None, None),
        ("u1", False, True, None, None),
    ]


def test_register_event_false_records_nothing():
    c, requests = client()
    c.complete([Search("sofa")], user_id="u1", register_event=False)
    c.sections([Search("sofa")], user_id="u1", register_event=False)

    assert [recording(sent(r)) for r in requests] == [
        ("u1", False, False, False, None),
        ("u1", False, False, False, None),
    ]


def test_without_a_user_id_nothing_is_recorded():
    c, requests = client()
    c.complete([View("a"), Search("sofa")])
    c.sections([View("a")])

    for request in requests:
        assert recording(sent(request)) == (None, True, False, None, None)


def test_register_event_needs_a_user_id():
    c, requests = client()
    with pytest.raises(ValueError, match="needs a user_id"):
        c.complete([Search("sofa")], register_event=True)
    with pytest.raises(ValueError, match="needs a user_id"):
        c.sections([Search("sofa")], register_event=True)
    assert requests == []


def test_a_user_id_with_a_history_is_refused_before_sending():
    c, requests = client()
    with pytest.raises(ValueError, match="pass no history"):
        c.complete([View("a")], user_id="u1")
    with pytest.raises(ValueError, match="pass no history"):
        c.complete([View("a"), Search("sofa")], user_id="u1")
    with pytest.raises(ValueError, match="pass no history"):
        c.sections([[View("a")], [Search("sofa")]], user_id="u1")
    with pytest.raises(ValueError, match="pass no history"):
        c.autocomplete("so", [View("a")], user_id="u1")
    assert requests == []


def test_a_request_with_a_user_id_and_a_history_is_invalid():
    with pytest.raises(ValidationError, match="not both"):
        UnboxAIRequest(
            user_id="u1",
            history=[
                SessionEvent(
                    event=View("a"),
                    domains=Domains(market="US"),
                    store_id="cat",
                    timezone="UTC",
                    session_id="s1",
                )
            ],
            store_id="cat",
            domains={"market": "US"},
        )


def test_autocomplete_is_recorded_as_a_call_never_as_a_search():
    c, requests = client()
    c.autocomplete("so", user_id="u1")

    # the typed text is the query; with no suggestions found, the text alone is
    # ranked again, and that second ranking is not recorded
    assert [r.url.path for r in requests] == ["/v1/vector/search"] * 2
    assert [recording(sent(r)) for r in requests] == [
        ("u1", False, False, None, "autocomplete"),
        ("u1", False, False, False, "autocomplete"),
    ]
    assert sent(requests[0])["query"] == "so"


def test_calls_lists_what_was_recorded():
    c, requests = client()
    calls = c.events.calls(user_id="u1", limit=20, calls=["search", "autocomplete"])

    assert requests[0].url.path == "/v1/events/own/calls"
    assert requests[0].url.params.multi_items() == [
        ("limit", "20"),
        ("user_id", "u1"),
        ("call", "search"),
        ("call", "autocomplete"),
    ]
    (call,) = calls
    assert (call.user_id, call.call, call.query, call.catalog_id) == (
        "u1",
        "search",
        "sofa",
        "cat",
    )
    assert [(i.id, i.catalog_id) for i in call.items] == [("p1", "cat")]
    assert call.timestamp == datetime(2026, 10, 28, 12, 3, tzinfo=UTC)


@pytest.mark.parametrize("limit", [0, 201])
def test_calls_refuses_a_limit_out_of_range(limit):
    c, _ = client()
    with pytest.raises(ValueError, match="limit"):
        c.events.calls(limit=limit)


def test_async_client_records_the_same():
    async def run() -> list[httpx.Request]:
        async with AsyncUnboxAIClient("US", api_key="test", base_url=BASE) as c:
            requests = mock(c)
            await c.complete([Search("sofa")], user_id="u1")
            await c.complete(user_id="u1", register_event=False)
            await c.sections(user_id="u1")
            assert len(await c.events.calls(user_id="u1")) == 1
            with pytest.raises(ValueError, match="needs a user_id"):
                await c.complete([Search("sofa")], register_event=True)
            return requests

    requests = asyncio.run(run())
    assert [recording(sent(r)) for r in requests[:3]] == [
        ("u1", False, True, None, None),
        ("u1", False, False, False, None),
        ("u1", False, True, None, None),
    ]
