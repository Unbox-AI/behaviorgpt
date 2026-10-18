"""AsyncUnboxAIClient against a mock transport: the same requests as the sync
client, awaited, and polling that leaves the event loop free."""

import asyncio
import json

import httpx
import pytest

from behaviorgpt import AsyncUnboxAIClient, Search, UnboxAIError, View
from behaviorgpt.resources.catalogs import AsyncCatalogs

from .test_embed import BASE, UPLOADED, status

PAGE = {
    "products": {
        "items": [
            {"id": "a", "data": {"name": "A"}, "score": 0.9},
            {"id": "b", "data": {"name": "B"}, "score": 0.5},
        ],
        "offset": 0,
        "limit": 2,
    }
}


def mock_client(response: httpx.Response | None = None):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response or httpx.Response(200, json=PAGE)

    client = AsyncUnboxAIClient("US", api_key="test", base_url=BASE)
    client._http_client._transport = httpx.MockTransport(handler)
    return client, requests


def scripted(steps: list):
    """As in test_embed: each status GET consumes one step, the last repeats."""
    calls: list[httpx.Request] = []
    steps = list(steps)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.method == "POST":
            return httpx.Response(201, json=UPLOADED)
        step = steps.pop(0) if len(steps) > 1 else steps[0]
        if isinstance(step, BaseException):
            raise step
        if isinstance(step, int):
            return httpx.Response(step, json={"detail": f"http {step}"})
        return httpx.Response(200, json=step)

    client = httpx.AsyncClient(base_url=BASE, transport=httpx.MockTransport(handler))
    return AsyncCatalogs(client), calls


def wait(steps, **kw):
    catalogs, calls = scripted(steps)
    seen = []
    kw = {"interval": 0.001, "timeout": 5, "on_progress": seen.append} | kw
    return asyncio.run(catalogs.wait_for_job("job-1", **kw)), seen, calls


# ---- requests ---------------------------------------------------------------


def test_a_view_history_asks_for_recommendations():
    client, requests = mock_client()
    response = asyncio.run(client.complete([View("sku-1")], catalog_id="cat"))
    assert requests[0].url.path == "/v1/vector/recommendations"
    assert requests[0].headers["x-catalog-id"] == "cat"
    assert requests[0].headers["x-api-key"] == "test"
    body = json.loads(requests[0].content)
    assert body["history"][0]["event"]["product"] == "sku-1"
    assert response.names == ["A", "B"]


def test_a_history_ending_in_a_search_asks_for_search():
    client, requests = mock_client()
    asyncio.run(client.complete([View("sku-1"), Search("shoes")]))
    assert requests[0].url.path == "/v1/vector/search"
    body = json.loads(requests[0].content)
    assert body["query"] == "shoes" and len(body["history"]) == 1


def test_similar_items_hits_the_similar_products_endpoint():
    client, requests = mock_client()
    response = asyncio.run(client.similar_items("sku-1", catalog_id="cat"))
    assert requests[0].url.path == "/v1/vector/similar_products"
    assert json.loads(requests[0].content)["product_id"] == "sku-1"
    assert [i.id for i in response.items] == ["a", "b"]


def test_random_item_picks_from_the_cold_start_ranking():
    client, _ = mock_client()
    assert asyncio.run(client.random_item("cat")).id in {"a", "b"}


def test_umap_returns_the_page_and_raises_on_error():
    client, _ = mock_client(httpx.Response(200, text="<html>map</html>"))
    assert asyncio.run(client.umap(catalog_id="cat")) == "<html>map</html>"
    client, _ = mock_client(httpx.Response(404, json={"detail": "unknown catalog"}))
    with pytest.raises(UnboxAIError, match="unknown catalog") as exc:
        asyncio.run(client.umap(catalog_id="cat"))
    assert exc.value.status_code == 404


def test_calls_run_concurrently_on_one_client():
    client, requests = mock_client()

    async def many():
        return await asyncio.gather(
            *(client.complete([View(f"sku-{i}")]) for i in range(10))
        )

    assert len(asyncio.run(many())) == 10 and len(requests) == 10


def test_async_with_closes_the_connections():
    async def use():
        async with AsyncUnboxAIClient("US", api_key="test") as client:
            pass
        return client

    assert asyncio.run(use())._http_client.is_closed


# ---- embed ------------------------------------------------------------------


def test_upload_is_a_multipart_file_field(catalog_path):
    catalogs, calls = scripted([status("ready")])
    job = asyncio.run(catalogs.embed(catalog_path))
    assert job.catalog_id == UPLOADED["catalog_id"]
    assert calls[0].url.path == "/v1/embed-catalog"
    assert b'name="file"; filename="catalog.parquet"' in calls[0].content


def test_client_embed_waits_for_the_job(catalog_path):
    client = AsyncUnboxAIClient("US", api_key="test", base_url=BASE)
    client.catalogs, calls = scripted([404, status("embedding"), status("ready")])
    job = asyncio.run(
        client.embed(
            str(catalog_path), wait=True, interval=0.001, on_progress=lambda _: None
        )
    )
    assert job.job_id == "job-1" and len(calls) == 4


def test_polling_reads_queued_then_survives_transient_errors():
    steps = [404, status("pending"), 503, httpx.ConnectError("reset"), status("ready")]
    result, seen, _ = wait(steps)
    assert result.status == "ready"
    assert [s.status for s in seen] == ["queued", "pending", "ready"]


def test_failure_raises_with_the_reason():
    with pytest.raises(UnboxAIError, match="bad file"):
        wait([status("pending"), status("failed", error="bad file")])


def test_client_errors_stop_polling():
    with pytest.raises(UnboxAIError) as exc:
        wait([status("pending"), 401, status("ready")])
    assert exc.value.status_code == 401


def test_timeout_names_the_last_state():
    with pytest.raises(UnboxAIError, match="still 'pending'"):
        wait([status("pending")], timeout=0.05, interval=0.01)


def test_waiting_leaves_the_event_loop_free():
    catalogs, _ = scripted([status("pending")] * 5 + [status("ready")])
    ticks = 0

    async def ticker(done: asyncio.Event):
        nonlocal ticks
        while not done.is_set():
            ticks += 1
            await asyncio.sleep(0.005)

    async def both():
        done = asyncio.Event()
        task = asyncio.create_task(ticker(done))
        await catalogs.wait_for_job("job-1", interval=0.02, on_progress=lambda _: None)
        done.set()
        await task

    asyncio.run(both())
    assert ticks >= 10
