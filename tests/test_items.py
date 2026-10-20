"""The client speaks of items; the API still names them products on the wire.
The old product names keep working for one release, with a DeprecationWarning."""

import json
import warnings

import httpx
import pytest

from behaviorgpt import AddToCart, Order, RemoveFromCart, UnboxAIClient, View
from behaviorgpt.types import CartItem, ItemsPage, SimilarItemsRequest, UnboxAIResponse

PAGE = {
    "products": {
        "items": [
            {"id": "a", "data": {"name": "A", "price": "10"}, "score": 0.9},
            {"id": "b", "data": {"name": "B", "price": "20"}, "score": 0.5},
        ],
        "offset": 0,
        "limit": 2,
    }
}


def mock_client():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=PAGE)

    client = UnboxAIClient("US", api_key="test", base_url="http://sdk.test/v1")
    client._http_client._transport = httpx.MockTransport(handler)
    return client, requests


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("event_type", [View, AddToCart, RemoveFromCart])
def test_item_events_send_product_on_the_wire(event_type):
    event = event_type("sku-1")
    assert event.item == "sku-1"
    assert event.model_dump()["product"] == "sku-1"
    assert "item" not in event.model_dump()


def test_item_events_accept_product_with_a_warning():
    with pytest.warns(DeprecationWarning, match="item="):
        event = View(product="sku-1")
    assert event.item == "sku-1"
    with pytest.warns(DeprecationWarning, match=r"\.item"):
        assert event.product == "sku-1"


def test_order_accepts_item_quantity_tuples():
    order = Order(items=[("sku-1", 2), CartItem("sku-2", 1)])
    assert [(i.item, i.quantity) for i in order.items] == [("sku-1", 2), ("sku-2", 1)]
    assert order.model_dump()["items"][0] == {"product": "sku-1", "quantity": 2}


def test_order_takes_the_items_bought_first():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        order = Order(["sku-1", ("sku-2", 2)], cart_id="cart-1")
    assert [(i.item, i.quantity) for i in order.items] == [("sku-1", 1), ("sku-2", 2)]
    assert order.cart_id == "cart-1"


def test_a_string_first_is_still_the_cart_id_with_a_warning():
    with pytest.warns(DeprecationWarning, match="cart_id="):
        order = Order("cart-1", ["sku-1"])
    assert order.cart_id == "cart-1"
    assert [i.item for i in order.items] == ["sku-1"]


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: AddToCart("sku-1"), "same as `View`"),
        (lambda: RemoveFromCart("sku-1"), "does not change results"),
        (lambda: Order(), r"Order\(\[item"),
    ],
)
def test_cart_events_and_an_order_without_items_are_deprecated(make, match):
    with pytest.warns(DeprecationWarning, match=match):
        make()


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_an_order_without_items_buys_the_cart_less_what_was_removed():
    client, requests = mock_client()
    client.complete(
        [
            AddToCart("sku-1"),
            AddToCart("sku-2"),
            AddToCart("sku-2"),
            RemoveFromCart("sku-1"),
            RemoveFromCart("sku-2"),
            Order(),
        ]
    )
    history = json.loads(requests[-1].content)["history"]
    assert history[-1]["event"]["items"] == [{"product": "sku-2", "quantity": 1}]


def test_response_flattens_the_products_page():
    response = UnboxAIResponse(**PAGE)
    assert [i.id for i in response.items] == ["a", "b"]
    assert (response.offset, response.limit) == (0, 2)
    assert response.names == ["A", "B"]
    with pytest.warns(DeprecationWarning, match="response.items"):
        page = response.products
    assert isinstance(page, ItemsPage)
    assert page.items == response.items


def test_similar_items_request_sends_product_id():
    req = SimilarItemsRequest(item_id="sku-1", store_id="cat")
    assert req.model_dump()["product_id"] == "sku-1"


def test_similar_items_hits_the_similar_products_endpoint():
    client, requests = mock_client()
    response = client.similar_items("sku-1", catalog_id="cat")
    assert requests[-1].url.path == "/v1/vector/similar_products"
    assert json.loads(requests[-1].content)["product_id"] == "sku-1"
    assert [i.id for i in response.items] == ["a", "b"]


def test_history_sends_product_in_events():
    client, requests = mock_client()
    client.complete([View("sku-1"), Order(["sku-1"])])
    history = json.loads(requests[-1].content)["history"]
    assert history[0]["event"]["product"] == "sku-1"
    assert history[1]["event"]["items"] == [{"product": "sku-1", "quantity": 1}]


def test_deprecated_client_methods_still_work():
    client, requests = mock_client()
    with pytest.warns(DeprecationWarning, match="similar_items"):
        client.similar_products(product_id="sku-1", catalog_id="cat")
    assert json.loads(requests[-1].content)["product_id"] == "sku-1"
    with pytest.warns(DeprecationWarning, match="random_item"):
        assert client.random_product("cat").id in {"a", "b"}


def test_deprecated_type_names_still_import():
    with pytest.warns(DeprecationWarning, match="ItemsPage"):
        from behaviorgpt.types import ProductResponse
    assert ProductResponse is ItemsPage
    with pytest.warns(DeprecationWarning, match="SimilarItemsRequest"):
        from behaviorgpt.types import SimilarProductsRequest
    assert SimilarProductsRequest is SimilarItemsRequest
