"""add_items: items added to, or replaced in, an embedded catalog, for the
sync and the async client, against a mock transport."""

import asyncio

import httpx
import pytest

from behaviorgpt import AsyncUnboxAIClient, UnboxAIClient, UnboxAIError
from behaviorgpt.types import AddItemsJobDetails

from .test_embed import BASE, status

CATALOG = "cat_0123456789abcdef01234567"
ACCEPTED = {
    "job_id": f"{CATALOG}-items-0badcafe",
    "catalog_id": CATALOG,
    "catalog_name": "shop",
    "uri": f"https://blob.example/raw-catalogs/{CATALOG}/items/b.parquet",
    "products": 3,
}


def handler_for(calls: list[httpx.Request], post: httpx.Response | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.method == "POST":
            return post or httpx.Response(201, json=ACCEPTED)
        return httpx.Response(200, json=status("ready", job_id=ACCEPTED["job_id"]))

    return handler


def sync_client(post: httpx.Response | None = None):
    calls: list[httpx.Request] = []
    client = UnboxAIClient(market="us", api_key="ubx_test", base_url=BASE)
    client._http_client._transport = httpx.MockTransport(handler_for(calls, post))
    return client, calls


def async_client(post: httpx.Response | None = None):
    calls: list[httpx.Request] = []
    client = AsyncUnboxAIClient("us", api_key="ubx_test", base_url=BASE)
    client._http_client._transport = httpx.MockTransport(handler_for(calls, post))
    return client, calls


def test_items_are_posted_to_the_catalog(catalog_path):
    client, calls = sync_client()
    job = client.add_items(str(catalog_path), catalog_id=CATALOG)

    assert isinstance(job, AddItemsJobDetails)
    assert job.job_id == ACCEPTED["job_id"] and job.products == 3
    (post,) = calls
    assert post.method == "POST"
    assert post.url.path == f"/v1/catalogs/{CATALOG}/items"
    body = post.read()
    assert b'name="file"; filename="catalog.parquet"' in body
    # the catalog's own reference time is used, so the SDK sends none
    assert b'name="reference_time"' not in body


def test_wait_polls_the_items_job_like_an_embed(catalog_path):
    client, calls = sync_client()
    client.add_items(
        catalog_path,
        catalog_id=CATALOG,
        wait=True,
        interval=0.001,
        on_progress=lambda _: None,
    )
    assert calls[-1].url.path == f"/v1/embed-catalog/{ACCEPTED['job_id']}"


def test_the_catalog_is_never_the_client_default(catalog_path):
    """Adding to `default_catalog_id` by accident would be refused anyway, but
    an explicit id says which catalog the items change."""
    client, _ = sync_client()
    with pytest.raises(TypeError):
        client.add_items(catalog_path)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="catalog_id"):
        client.add_items(catalog_path, catalog_id="")


def test_a_bad_schema_is_refused_before_any_request(write_catalog):
    from .conftest import catalog_table

    path = write_catalog(catalog_table().drop_columns(["frequency"]))
    client, calls = sync_client()
    with pytest.raises(UnboxAIError, match="frequency: missing"):
        client.add_items(path, catalog_id=CATALOG)
    assert calls == []


@pytest.mark.parametrize(
    "code, detail",
    [
        (409, "items are already being added to this catalog"),
        (409, "upload the whole catalog once more, then add items"),
        (413, "with these items the catalog would hold 20001 products"),
        (404, f"unknown catalog: {CATALOG}"),
    ],
)
def test_a_refusal_is_an_sdk_error_with_the_servers_reason(catalog_path, code, detail):
    client, _ = sync_client(httpx.Response(code, json={"detail": detail}))
    with pytest.raises(UnboxAIError) as exc:
        client.add_items(catalog_path, catalog_id=CATALOG)
    assert exc.value.status_code == code
    assert detail in str(exc.value)


def test_async_items_are_posted_and_waited_for(catalog_path):
    async def run():
        client, calls = async_client()
        async with client:
            job = await client.add_items(
                catalog_path,
                catalog_id=CATALOG,
                wait=True,
                interval=0.001,
                on_progress=lambda _: None,
            )
        return job, calls

    job, calls = asyncio.run(run())
    assert job.products == 3
    assert calls[0].url.path == f"/v1/catalogs/{CATALOG}/items"
    assert calls[-1].url.path == f"/v1/embed-catalog/{ACCEPTED['job_id']}"


def test_async_refusal_is_an_sdk_error(catalog_path):
    async def run():
        client, _ = async_client(
            httpx.Response(409, json={"detail": "catalog x is pending"})
        )
        async with client:
            await client.add_items(catalog_path, catalog_id=CATALOG)

    with pytest.raises(UnboxAIError) as exc:
        asyncio.run(run())
    assert exc.value.status_code == 409
