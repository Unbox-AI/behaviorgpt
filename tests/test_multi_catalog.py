"""Ranking several catalogs together: `catalog_ids=` sends x-catalog-ids, each
history event names the catalog of its items, and every item of a response
knows its catalog."""

import asyncio
import json

import httpx
import pytest

from behaviorgpt import AsyncUnboxAIClient, UnboxAIClient
from behaviorgpt.types import CartItem, Item, Order, Search, View


class Api:
    """Records the requests and answers each like the API: items carry a
    store_id only when the request reads several catalogs."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        multi = "x-catalog-ids" in request.headers
        item = {"id": "p1", "data": {"name": "P1"}, "score": 0.5}
        if multi:
            item["store_id"] = "retail_catalog"
        if request.url.path.endswith("/sections"):
            return httpx.Response(
                200, json={"sections": [{"title": "coats", "products": [item]}]}
            )
        return httpx.Response(
            200, json={"products": {"items": [item], "offset": 0, "limit": 10}}
        )

    @property
    def last(self) -> httpx.Request:
        return self.requests[-1]

    @property
    def body(self) -> dict:
        return json.loads(self.last.content)


@pytest.fixture
def api() -> Api:
    return Api()


@pytest.fixture
def client(api: Api) -> UnboxAIClient:
    client = UnboxAIClient("SE", api_key="test", base_url="http://sdk.test/v1")
    client._http_client._transport = httpx.MockTransport(api.handler)
    return client


MINE = "cat_mine"
BOTH = [MINE, "retail_catalog"]


def test_a_single_catalog_request_is_unchanged(client: UnboxAIClient, api: Api):
    resp = client.complete([View("p1")], catalog_id=MINE)

    assert api.last.headers["x-catalog-id"] == MINE
    assert "x-catalog-ids" not in api.last.headers
    assert [e["store_id"] for e in api.body["history"]] == [MINE]
    # the API does not name the catalog of a single-catalog request's items
    assert resp.items[0].catalog_id == MINE


def test_catalog_ids_ranks_several_together(client: UnboxAIClient, api: Api):
    resp = client.complete(
        [View("p1"), View("r9", catalog_id="retail_catalog")],
        catalog_ids=BOTH,
        filters={"catalog_id": "retail_catalog"},
    )

    assert api.last.headers["x-catalog-ids"] == "cat_mine,retail_catalog"
    assert "x-catalog-id" not in api.last.headers
    assert api.body["store_id"] == MINE
    assert api.body["filters"] == {"catalog_id": "retail_catalog"}
    history = api.body["history"]
    assert [(e["store_id"], e["event"]["product"]) for e in history] == [
        (MINE, "p1"),
        ("retail_catalog", "r9"),
    ]
    # the event's catalog is its store_id, not a field of the event
    assert "catalog_id" not in history[1]["event"]
    assert resp.items[0].catalog_id == "retail_catalog"


def test_items_of_a_response_carry_their_catalog_into_the_next(
    client: UnboxAIClient, api: Api
):
    first = client.complete([], catalog_ids=BOTH)
    client.complete([View(first.items[0])], catalog_ids=BOTH)

    (event,) = api.body["history"]
    assert (event["store_id"], event["event"]["product"]) == ("retail_catalog", "p1")


def test_an_order_of_several_catalogs_is_one_order_per_catalog(
    client: UnboxAIClient, api: Api
):
    retail_item = Item(id="r1", data={}, score=1.0, catalog_id="retail_catalog")
    client.complete(
        [Order(["p1", (retail_item, 2), CartItem("p2", 1)], cart_id="c1")],
        catalog_ids=BOTH,
    )

    orders = api.body["history"]
    assert [
        (o["store_id"], [i["product"] for i in o["event"]["items"]]) for o in orders
    ] == [(MINE, ["p1", "p2"]), ("retail_catalog", ["r1"])]
    assert {o["event"]["cart_id"] for o in orders} == {"c1"}
    first, second = (o["timestamp"] for o in orders)
    assert first < second


def test_a_search_is_in_the_first_catalog(client: UnboxAIClient, api: Api):
    client.complete(
        [View("r1", catalog_id="retail_catalog"), Search("coats")], catalog_ids=BOTH
    )
    assert api.last.url.path.endswith("/vector/search")
    assert api.body["query"] == "coats"
    assert [e["store_id"] for e in api.body["history"]] == ["retail_catalog"]


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"catalog_ids": BOTH, "catalog_id": MINE}, "not both"),
        ({"catalog_ids": []}, "names no catalog"),
        ({"catalog_ids": "cat_mine"}, "list of catalog ids"),
    ],
)
def test_catalog_ids_are_checked(client: UnboxAIClient, kwargs: dict, error: str):
    with pytest.raises((ValueError, TypeError), match=error):
        client.complete([], **kwargs)


def test_an_item_of_a_catalog_not_read_is_refused(client: UnboxAIClient, api: Api):
    with pytest.raises(ValueError, match="'cat_other'.*catalog_ids="):
        client.complete([View("x", catalog_id="cat_other")], catalog_id=MINE)
    with pytest.raises(ValueError, match="'cat_other'"):
        client.complete([View("x", catalog_id="cat_other")], catalog_ids=BOTH)
    assert api.requests == []


def test_sections_over_several_catalogs(client: UnboxAIClient, api: Api):
    resp = client.sections([View("p1")], catalog_ids=BOTH)
    assert api.last.headers["x-catalog-ids"] == "cat_mine,retail_catalog"
    assert resp.sections[0].items[0].catalog_id == "retail_catalog"


def test_similar_items_of_an_item_in_one_catalog_over_several(
    client: UnboxAIClient, api: Api
):
    item = Item(id="r1", data={}, score=1.0, catalog_id="retail_catalog")
    resp = client.similar_items(item, catalog_ids=BOTH, filters={"brand": "x"})

    assert api.last.headers["x-catalog-ids"] == "cat_mine,retail_catalog"
    assert api.body["product_id"] == "r1"
    assert api.body["store_id"] == "retail_catalog"
    assert api.body["filters"] == {"brand": "x"}
    assert resp.items[0].catalog_id == "retail_catalog"

    client.similar_items("p1", catalog_ids=BOTH)
    assert api.body["store_id"] == MINE
    client.similar_items("p1", catalog_id="retail_catalog", catalog_ids=BOTH)
    assert api.body["store_id"] == "retail_catalog"
    with pytest.raises(ValueError, match="'cat_other'"):
        client.similar_items("p1", catalog_id="cat_other", catalog_ids=BOTH)


def test_similar_items_of_an_item_reads_its_catalog(client: UnboxAIClient, api: Api):
    item = Item(id="r1", data={}, score=1.0, catalog_id="retail_catalog")
    client.similar_items(item)
    assert api.last.headers["x-catalog-id"] == "retail_catalog"
    assert api.body["store_id"] == "retail_catalog"


def test_to_pandas_names_the_catalog(client: UnboxAIClient):
    pytest.importorskip("pandas")
    frame = client.complete([], catalog_ids=BOTH).to_pandas()
    assert frame.loc[0, "catalog_id"] == "retail_catalog"


def test_the_async_client_reads_several_catalogs(api: Api):
    async def run():
        async with AsyncUnboxAIClient(
            "SE", api_key="test", base_url="http://sdk.test/v1"
        ) as client:
            client._http_client._transport = httpx.MockTransport(api.handler)
            return (
                await client.complete([View("p1")], catalog_ids=BOTH),
                await client.sections([], catalog_ids=BOTH),
                await client.similar_items("p1", catalog_ids=BOTH),
            )

    resp, sections, similar = asyncio.run(run())

    assert all("x-catalog-ids" in r.headers for r in api.requests)
    assert resp.items[0].catalog_id == "retail_catalog"
    assert sections.sections[0].items[0].catalog_id == "retail_catalog"
    assert similar.items[0].catalog_id == "retail_catalog"


def test_the_default_catalog_is_the_argument_then_the_environment(monkeypatch):
    monkeypatch.setenv("UNBOXAI_API_KEY", "k")
    monkeypatch.delenv("UNBOXAI_CATALOG_ID", raising=False)
    assert UnboxAIClient("se").default_catalog_id == "retail_catalog"
    monkeypatch.setenv("UNBOXAI_CATALOG_ID", "cat_env")
    assert UnboxAIClient("se").default_catalog_id == "cat_env"
    assert AsyncUnboxAIClient("se").default_catalog_id == "cat_env"
    client = UnboxAIClient("se", default_catalog_id="cat_arg")
    assert client.default_catalog_id == "cat_arg"
