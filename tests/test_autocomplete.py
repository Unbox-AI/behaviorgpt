"""autocomplete() against a mock transport: the ranking each call asks for,
and the suggestions read from it."""

import asyncio
import json

import httpx
import pytest

from behaviorgpt import AsyncUnboxAIClient, Search, UnboxAIClient, View
from behaviorgpt._autocomplete import match_score, normalize, suggest
from behaviorgpt.types import Item

from .test_embed import BASE


def item(item_id: str, *terms: str, name: str | None = None) -> dict:
    data: dict = {"name": name or f"Item {item_id}"}
    if terms:
        data["top_terms"] = [{"term": t, "score": 0.5} for t in terms]
    return {"id": item_id, "data": data, "score": 1.0}


def ranking(*items: dict) -> dict:
    return {"products": {"items": list(items), "offset": 0, "limit": 50}}


COFFEE = ranking(
    item("a", "coffee maker", "coffee", "mr coffee"),
    item("b", "coffee maker", "ninja coffee maker"),
    item("c", "espresso machine", "nespresso"),
    item("d", "keurig coffee machine"),
)


def mock(client_class, *responses: dict):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200, json=responses[min(len(requests), len(responses)) - 1]
        )

    client = client_class("US", api_key="test", base_url=BASE)
    client._http_client._transport = httpx.MockTransport(handler)
    return client, requests


# ---- the request ------------------------------------------------------------


def test_the_typed_text_is_a_search_that_is_not_recorded():
    client, requests = mock(UnboxAIClient, COFFEE)

    client.autocomplete("coffee m", catalog_id="cat")

    assert requests[0].url.path == "/v1/vector/search"
    assert requests[0].headers["x-catalog-id"] == "cat"
    body = json.loads(requests[0].content)
    assert body["query"] == "coffee m"
    assert body["register_event"] is False
    assert body["limit"] == 50
    assert body["fields_to_include"] == ["name", "top_terms"]


def test_the_history_comes_before_the_typed_text():
    client, requests = mock(UnboxAIClient, COFFEE)

    client.autocomplete("m", [View("sku-1"), Search("coffee")])

    body = json.loads(requests[0].content)
    assert body["query"] == "m"
    assert [e["event"]["type"] for e in body["history"]] == ["viewItem", "search"]


def test_the_typed_text_ends_the_last_session():
    client, requests = mock(UnboxAIClient, COFFEE)

    client.autocomplete("m", [[View("sku-1")], [View("sku-2")]])

    body = json.loads(requests[0].content)
    sessions = [e["session_id"] for e in body["history"]]
    assert len(set(sessions)) == 2
    assert body["history"][-1]["event"]["product"] == "sku-2"


def test_empty_text_suggests_what_comes_after_the_history():
    client, requests = mock(UnboxAIClient, COFFEE)

    response = client.autocomplete("  ", [View("sku-1")])

    assert requests[0].url.path == "/v1/vector/recommendations"
    body = json.loads(requests[0].content)
    assert "query" not in body
    assert body["fields_to_include"] == ["name", "top_terms"]
    assert response.texts == [
        "coffee maker",
        "ninja coffee maker",
        "espresso machine",
        "keurig coffee machine",
    ]


def test_limits_are_checked_before_sending():
    client, requests = mock(UnboxAIClient, COFFEE)

    with pytest.raises(ValueError, match="limit"):
        client.autocomplete("a", limit=21)
    with pytest.raises(ValueError, match="candidates"):
        client.autocomplete("a", limit=10, candidates=5)
    with pytest.raises(ValueError, match="candidates"):
        client.autocomplete("a", candidates=101)
    assert requests == []


# ---- the suggestions --------------------------------------------------------


def test_one_suggestion_per_item_in_the_items_order():
    client, _ = mock(UnboxAIClient, COFFEE)

    response = client.autocomplete("coffee m")

    # "a" offers coffee maker; "b" skips it, already suggested; "c" has no
    # term that continues the text; "d" matches past its first word
    assert response.texts == [
        "coffee maker",
        "ninja coffee maker",
        "keurig coffee machine",
    ]
    assert [s.item.id for s in response.suggestions] == ["a", "b", "d"]
    assert {s.source for s in response.suggestions} == {"keyword"}
    assert response.suggestions[0].item.catalog_id == "retail_catalog"


def test_a_typo_still_matches():
    client, _ = mock(UnboxAIClient, COFFEE)

    assert client.autocomplete("cofee mak", limit=1).texts == ["coffee maker"]


def test_items_without_keywords_offer_their_names():
    client, _ = mock(
        UnboxAIClient,
        ranking(
            item("a", name="Pocket Knife, Steel"),
            item("b", name="Garden hose"),
            item("c", "knife set"),
        ),
    )

    response = client.autocomplete("kni")

    assert [(s.text, s.source) for s in response.suggestions] == [
        ("Pocket Knife, Steel", "name"),
        ("knife set", "keyword"),
    ]


def test_too_few_suggestions_after_a_history_are_filled_from_the_text_alone():
    golf = ranking(item("a", "golf clubs", "golf set"), item("b", "golf gloves"))
    alone = ranking(item("c", "glasses"), item("d", "golf gloves"), item("e", "glue"))
    client, requests = mock(UnboxAIClient, golf, alone)

    response = client.autocomplete("gl", [Search("golf clubs")], limit=3)

    assert len(requests) == 2
    first, second = (json.loads(r.content) for r in requests)
    assert first.get("max_trailing_searches") == 3
    assert second["max_trailing_searches"] == 1
    assert second["query"] == "gl"
    assert second["register_event"] is False
    assert response.texts == ["golf gloves", "glasses", "glue"]


def test_no_second_ranking_without_a_history_or_once_there_are_enough():
    client, requests = mock(UnboxAIClient, ranking(item("a", "glasses")))
    client.autocomplete("gl")
    assert len(requests) == 1

    client, requests = mock(UnboxAIClient, COFFEE)
    client.autocomplete("coffee m", [Search("coffee")], limit=2)
    assert len(requests) == 1


def test_several_catalogs():
    client, requests = mock(
        UnboxAIClient,
        ranking({**item("a", "coffee maker"), "store_id": "mine"}),
    )

    response = client.autocomplete("cof", catalog_ids=["mine", "retail_catalog"])

    assert requests[0].headers["x-catalog-ids"] == "mine,retail_catalog"
    assert response.suggestions[0].item.catalog_id == "mine"


def test_async_client():
    async def run():
        client, requests = mock(AsyncUnboxAIClient, COFFEE)
        async with client:
            return await client.autocomplete("coffee m", limit=2), requests

    response, requests = asyncio.run(run())

    assert response.texts == ["coffee maker", "ninja coffee maker"]
    assert json.loads(requests[0].content)["query"] == "coffee m"


# ---- matching ---------------------------------------------------------------


@pytest.mark.parametrize(
    "typed, candidate, expected",
    [
        ("kni", "pocket knife", 1.0),
        ("coffee m", "ninja coffee maker", 1.0),
        ("KNÍ", "Pocket knife", 1.0),
        ("", "anything", 1.0),
        ("cofee", "coffee maker", 0.8),
        ("nife", "pocket knife", 0.75),
        ("ife", "pocket knife", 1 / 3),
        ("xyz", "pocket knife", 0.0),
    ],
)
def test_match_score(typed, candidate, expected):
    assert match_score(typed, candidate) == pytest.approx(expected)


def test_normalize():
    assert normalize("  Café   Crème ") == "cafe creme"


def test_alike_suggestions_are_left_out():
    items = [
        Item(**item("a", "coffee maker")),
        Item(**item("b", "coffee makers", "coffee grinder")),
    ]

    assert [s.text for s in suggest(items, "coffee", 5)] == [
        "coffee maker",
        "coffee grinder",
    ]


def test_terms_as_pairs():
    items = [Item(id="a", data={"top_terms": [["knife set", 0.4]]}, score=1.0)]

    assert [s.text for s in suggest(items, "kn", 5)] == ["knife set"]
