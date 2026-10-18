"""list_catalogs and delete_catalog, for the sync and the async client,
against a mock transport."""

import asyncio

import httpx
import pytest

from behaviorgpt import AsyncUnboxAIClient, UnboxAIClient, UnboxAIError
from behaviorgpt.types import CatalogSummary

from .test_embed import BASE

CATALOG = "cat_0123456789abcdef01234567"
LISTED = {
    "catalogs": [
        {
            "catalog_id": CATALOG,
            "catalog_name": "shop",
            "status": "pending",
            "owned": True,
        },
        {
            "catalog_id": "retail_catalog",
            "catalog_name": "retail_catalog",
            "status": "ready",
            "owned": False,
        },
    ]
}


def handler_for(calls: list[httpx.Request], response: httpx.Response):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return response

    return handler


def sync_client(response: httpx.Response):
    calls: list[httpx.Request] = []
    client = UnboxAIClient(market="us", api_key="ubx_test", base_url=BASE)
    client._http_client._transport = httpx.MockTransport(handler_for(calls, response))
    return client, calls


def async_client(response: httpx.Response):
    calls: list[httpx.Request] = []
    client = AsyncUnboxAIClient("us", api_key="ubx_test", base_url=BASE)
    client._http_client._transport = httpx.MockTransport(handler_for(calls, response))
    return client, calls


def test_list_catalogs():
    client, calls = sync_client(httpx.Response(200, json=LISTED))
    catalogs = client.list_catalogs()

    assert catalogs == [CatalogSummary(**c) for c in LISTED["catalogs"]]
    assert catalogs[0].owned and catalogs[0].status == "pending"
    (get,) = calls
    assert get.method == "GET" and get.url.path == "/v1/catalogs"


def test_list_catalogs_async():
    async def run():
        client, calls = async_client(httpx.Response(200, json=LISTED))
        async with client:
            catalogs = await client.list_catalogs()
        return catalogs, calls

    catalogs, calls = asyncio.run(run())
    assert [c.catalog_id for c in catalogs] == [CATALOG, "retail_catalog"]
    assert calls[0].url.path == "/v1/catalogs"


def test_delete_catalog():
    client, calls = sync_client(httpx.Response(204))
    assert client.delete_catalog(CATALOG) is None

    (delete,) = calls
    assert delete.method == "DELETE"
    assert delete.url.path == f"/v1/catalogs/{CATALOG}"


def test_delete_catalog_async():
    async def run():
        client, calls = async_client(httpx.Response(204))
        async with client:
            result = await client.delete_catalog(CATALOG)
        return result, calls

    result, calls = asyncio.run(run())
    assert result is None
    assert calls[0].method == "DELETE"
    assert calls[0].url.path == f"/v1/catalogs/{CATALOG}"


@pytest.mark.parametrize(
    "status,detail",
    [
        (404, f"unknown catalog: {CATALOG}"),
        (
            409,
            "an embed, ingest or items job is still running for this catalog; "
            "delete it once that job is ready or failed",
        ),
    ],
)
def test_a_refused_delete_raises_with_the_server_reason(status, detail):
    client, _ = sync_client(httpx.Response(status, json={"detail": detail}))
    with pytest.raises(UnboxAIError, match="still running|unknown catalog") as exc:
        client.delete_catalog(CATALOG)
    assert exc.value.status_code == status


def test_delete_needs_a_catalog_id():
    client, calls = sync_client(httpx.Response(204))
    with pytest.raises(ValueError, match="catalog_id"):
        client.delete_catalog("")
    assert calls == []


def test_the_catalog_id_cannot_reach_another_route():
    client, calls = sync_client(httpx.Response(404, json={"detail": "unknown"}))
    with pytest.raises(UnboxAIError):
        client.delete_catalog("../embed-catalog/x?y=1")
    (delete,) = calls
    assert delete.url.raw_path == b"/v1/catalogs/..%2Fembed-catalog%2Fx%3Fy%3D1"
