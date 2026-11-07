"""`ExternalItem`: an item in none of the catalogs, by its name and picture, in a
history where an item id goes, and as the item `similar_items` starts from. It
travels as `item`, where a catalog item travels as `product`."""

import base64
import json

import httpx
import pytest

from behaviorgpt import ExternalItem, Order, Search, UnboxAIClient, View
from behaviorgpt.types import CartItem

PNG = b"\x89PNG\r\n\x1a\n" + bytes(16)
PAGE = {"products": {"items": [], "offset": 0, "limit": 10}}


def mock_client(**options):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=PAGE)

    client = UnboxAIClient(
        "US", api_key="test", base_url="http://sdk.test/v1", **options
    )
    client._http_client._transport = httpx.MockTransport(handler)
    return client, requests


def sent_events(request: httpx.Request) -> list[dict]:
    return [doc["event"] for doc in json.loads(request.content)["history"]]


def test_an_item_needs_a_name_or_a_picture():
    assert ExternalItem("Linen shirt").name == "Linen shirt"
    with pytest.raises(ValueError, match="a name or an image"):
        ExternalItem(brand="Acme")
    with pytest.raises(ValueError, match="a name or an image"):
        ExternalItem.model_validate({"categories": "Shirts"})


def test_a_picture_is_a_link_or_the_picture_itself(tmp_path):
    link = "https://cdn.example/shirt.jpg"
    assert ExternalItem(image=link).image == link
    data_url = "data:image/png;base64," + base64.b64encode(PNG).decode()
    assert ExternalItem(image=PNG).image == data_url
    assert ExternalItem(image=data_url).image == data_url
    path = tmp_path / "shirt.png"
    path.write_bytes(PNG)
    assert ExternalItem(image=path).image == data_url
    assert ExternalItem(image=str(path)).image == data_url
    with pytest.raises(ValueError, match="https"):
        ExternalItem(image="http://cdn.example/shirt.jpg")


def test_a_history_sends_an_item_in_no_catalog_as_item():
    client, requests = mock_client()
    shirt = ExternalItem("Linen shirt", brand="Acme", categories="Clothing, Shirts")
    photo = ExternalItem(image="https://cdn.example/look.jpg")

    client.complete([View("sku-1"), View(shirt), Order(["sku-2", (photo, 2)])])

    (request,) = requests
    view, view_external, order = sent_events(request)
    assert view == {"product": "sku-1", "type": "viewItem"}
    assert view_external == {
        "item": {
            "name": "Linen shirt",
            "brand": "Acme",
            "categories": "Clothing, Shirts",
        },
        "type": "viewItem",
    }
    assert order["items"] == [
        {"product": "sku-2", "quantity": 1},
        {"item": {"image": "https://cdn.example/look.jpg"}, "quantity": 2},
    ]


def test_an_item_in_no_catalog_is_in_the_requests_first_catalog():
    client, requests = mock_client(default_catalog_id="a")
    client.complete([View(ExternalItem("x")), Search("shirts")], catalog_ids=["a", "b"])
    (request,) = requests
    (doc,) = json.loads(request.content)["history"]
    assert doc["store_id"] == "a"


def test_cart_items_can_be_items_in_no_catalog():
    photo = ExternalItem(image=PNG)
    assert CartItem(photo, 3).item == photo
    assert Order([photo]).items[0].item == photo


def test_tracked_events_take_catalog_items_only():
    client, requests = mock_client()
    with pytest.raises(ValueError, match="catalog items only"):
        client.events.track("user-1", View(ExternalItem("x")))
    with pytest.raises(ValueError, match="catalog items only"):
        client.events.track("user-1", Order([ExternalItem("x")]))
    assert requests == []
