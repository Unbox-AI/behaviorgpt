"""client.shop against a mock transport: the body each call sends the portal,
how its answer comes back, and the picture search's waiting."""

import asyncio
import base64
import json

import httpx
import pytest

from behaviorgpt import (
    AsyncUnboxAIClient,
    AuthenticationError,
    Only,
    Order,
    Search,
    ShopItem,
    UnboxAIClient,
    UnboxAIError,
    User,
    View,
)
from behaviorgpt.resources import shop as shop_module

CARD = {
    "id": "sofa-1",
    "name": "Linen sofa",
    "brand": "Nord",
    "image": "https://img.example.com/s1.jpg",
    "price": 799.0,
    "currency": "EUR",
    "url": None,
}


def mock(client_class, answers: dict | None = None, **options):
    """A client whose shop calls go to `answers` by path: a dict answers 200, a
    (status, body) pair that status, a list one entry per call."""
    requests: list[httpx.Request] = []
    answers = answers or {}

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path.removeprefix("/api/v1/shop/")
        answer = answers.get(path, {})
        if isinstance(answer, list):
            answer = answer.pop(0)
        status, body = answer if isinstance(answer, tuple) else (200, answer)
        return httpx.Response(status, json=body)

    client = client_class(
        "us",
        api_key="ubx_test",
        default_catalog_id="cat_1",
        timezone="America/New_York",
        **options,
    )
    client._shop_client._transport = httpx.MockTransport(handler)
    return client, requests


def sent(request: httpx.Request) -> dict:
    return json.loads(request.content)


def test_a_search_goes_to_the_portal_with_the_key_and_the_shopper():
    page = {"items": [CARD], "offset": 0, "more": True}
    client, requests = mock(UnboxAIClient, {"search": page})
    found = client.shop.search(
        " linen sofa ",
        user_id="shopper-42",
        history=[Search("linen"), View("lamp-1")],
        user=User(gender="female", age=34),
        only=Only(categories=["Sofas"], max_price=900),
        limit=24,
    )
    assert found.items == [ShopItem(**CARD)] and found.more is True
    (request,) = requests
    assert str(request.url) == "https://portal.unboxai.com/api/v1/shop/search"
    assert request.headers["authorization"] == "Bearer ubx_test"
    assert sent(request) == {
        "catalog": "cat_1",
        "user": "shopper-42",
        "history": [
            {"type": "search", "query": "linen"},
            {"type": "view", "item": "lamp-1"},
        ],
        # the client's market and time zone, with who the shopper is
        "who": {
            "gender": "female",
            "age_group": "26-35",
            "market": "US",
            "timezone": "America/New_York",
        },
        "query": "linen sofa",
        "limit": 24,
        "offset": 0,
        "only": {"categories": ["Sofas"], "max_price": 900.0},
    }


def test_each_call_asks_its_own_path_and_reads_its_answer():
    client, requests = mock(
        UnboxAIClient,
        {
            "suggest": {
                "suggestions": [
                    {"text": "linen sofa", "source": "keyword", "item": "sofa-1"}
                ]
            },
            "browse": {"items": [CARD], "offset": 12, "more": False},
            "recommend": {"items": [{**CARD, "tag": "Sofas"}]},
            "sections": {"sections": [{"title": "Linen sofas", "items": [CARD]}]},
            "item": {"item": {**CARD, "keywords": ["linen sofa"]}},
        },
    )
    assert client.shop.suggest("lin")[0].text == "linen sofa"
    assert client.shop.browse(Only(brands=["Nord"]), offset=12).offset == 12
    shelf = client.shop.recommend("bought_together", item="sofa-1", limit=4)
    assert shelf[0].tag == "Sofas"
    assert client.shop.sections(sections=3)[0].title == "Linen sofas"
    assert client.shop.item("sofa-1", catalog_id="cat_2").keywords == ["linen sofa"]
    bodies = {r.url.path.rsplit("/", 1)[-1]: sent(r) for r in requests}
    assert bodies["suggest"]["text"] == "lin" and "user" not in bodies["suggest"]
    assert bodies["browse"]["only"] == {"brands": ["Nord"]}
    assert bodies["recommend"] == {
        "catalog": "cat_1",
        "who": {"market": "US", "timezone": "America/New_York"},
        "kind": "bought_together",
        "item": "sofa-1",
        "limit": 4,
    }
    assert "query" not in bodies["sections"] and bodies["sections"]["sections"] == 3
    assert bodies["item"]["catalog"] == "cat_2"


def test_the_portal_is_another_when_asked(monkeypatch):
    client, requests = mock(
        UnboxAIClient, {"item": {"item": CARD}}, portal_url="http://localhost:8000/"
    )
    client.shop.item("sofa-1")
    assert str(requests[0].url) == "http://localhost:8000/api/v1/shop/item"
    monkeypatch.setenv("UNBOXAI_PORTAL_URL", "https://portal.test")
    assert UnboxAIClient("us", api_key="k").portal_url == "https://portal.test"


def test_what_is_wrong_comes_back_as_the_sdk_s_errors():
    client, _ = mock(
        UnboxAIClient,
        {
            "search": (404, {"detail": "unknown catalog: cat_x"}),
            "item": (401, {"detail": "That API key isn't valid."}),
        },
    )
    with pytest.raises(UnboxAIError, match="unknown catalog: cat_x") as error:
        client.shop.search("sofa", catalog_id="cat_x")
    assert error.value.status_code == 404
    with pytest.raises(AuthenticationError):
        client.shop.item("sofa-1")


def test_history_takes_searches_and_views():
    client, _ = mock(UnboxAIClient)
    with pytest.raises(TypeError, match="after_purchase"):
        client.shop.search("sofa", history=[Order(["sofa-1"])])
    with pytest.raises(ValueError, match="category"):
        client.shop.browse(Only())


def test_a_picture_goes_as_itself_or_as_its_link(tmp_path):
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 20
    path = tmp_path / "chair.png"
    path.write_bytes(png)
    ready = {"status": "ready", "picture": "photo.1", "items": [CARD]}
    client, requests = mock(UnboxAIClient, {"visual": ready})
    assert client.shop.visual(path).items == [ShopItem(**CARD)]
    assert client.shop.visual(png).status == "ready"
    client.shop.visual("https://shop.example.com/p/chair", kind="for_you")
    encoded = base64.b64encode(png).decode()
    assert sent(requests[0])["image"] == f"data:image/png;base64,{encoded}"
    assert sent(requests[1])["image"] == f"data:image/png;base64,{encoded}"
    assert sent(requests[2])["url"] == "https://shop.example.com/p/chair"
    assert sent(requests[2])["kind"] == "for_you"
    with pytest.raises(ValueError, match="not a file"):
        client.shop.visual(str(tmp_path / "missing.jpg"))


def test_a_new_picture_is_waited_for(monkeypatch):
    monkeypatch.setattr(shop_module, "PICTURE_POLL", 0)
    client, requests = mock(
        UnboxAIClient,
        {
            "visual": {"status": "pending", "picture": "photo.1", "phase": "Reading"},
            "visual/status": [
                {"status": "embedding", "picture": "photo.1", "phase": "Embedding"},
                {"status": "ready", "picture": "photo.1", "items": [CARD]},
            ],
        },
    )
    heard: list[str] = []
    found = client.shop.visual(
        "https://img.example.com/c.jpg", on_progress=heard.append
    )
    assert found.status == "ready" and found.picture == "photo.1"
    assert heard == ["Reading", "Embedding"]
    assert [r.url.path for r in requests][-1] == "/api/v1/shop/visual/status"
    assert sent(requests[-1])["picture"] == "photo.1"


def test_a_picture_that_fails_says_why():
    client, _ = mock(
        UnboxAIClient,
        {
            "visual": {
                "status": "failed",
                "picture": "photo.1",
                "error": "Not a picture",
            }
        },
    )
    found = client.shop.visual("https://img.example.com/c.jpg")
    assert found.status == "failed" and found.error == "Not a picture"


def test_the_async_client_makes_the_same_calls():
    async def run():
        client, requests = mock(
            AsyncUnboxAIClient,
            {
                "recommend": {"items": [CARD]},
                "search": {"items": [], "offset": 0, "more": False},
            },
        )
        async with client:
            shelf = await client.shop.recommend("similar", item="sofa-2")
            page = await client.shop.search("sofa", user_id="u")
        assert shelf == [ShopItem(**CARD)] and page.more is False
        assert sent(requests[0])["kind"] == "similar"
        assert sent(requests[1])["user"] == "u"

    asyncio.run(run())
