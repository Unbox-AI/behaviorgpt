"""Joint maps of several catalogs: `umap(catalog_ids=...)` asks for the map,
polls until it is drawn and returns its page; the catalogs resource reads its
status and points."""

import asyncio
import gzip
import json

import httpx
import pytest

from behaviorgpt import AsyncUnboxAIClient, UnboxAIClient, UnboxAIError

BASE = "http://sdk.test/v1"
MAP_ID = "map_0123456789abcdef01234567"
IDS = ["cat_a", "retail_catalog"]


def joint(status: str, error: str | None = None, code: int = 200) -> httpx.Response:
    return httpx.Response(
        code,
        json={
            "map_id": MAP_ID,
            "status": status,
            "catalog_ids": IDS,
            "requested_at": "2026-10-24T21:16:28Z",
            "error": error,
        },
    )


def serve(statuses: list[httpx.Response], client_class=UnboxAIClient):
    """POST /maps answers the first status, each GET /maps/{id} the next (the
    last repeats), and the page and points answer as stored."""
    calls: list[httpx.Request] = []
    pending = list(statuses)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        path = request.url.path
        if path.endswith("/page"):
            return httpx.Response(200, text="<html>joint</html>")
        if path.endswith("/points"):
            body = gzip.compress(json.dumps({"catalog": [0, 1]}).encode())
            gzipped = {"content-encoding": "gzip"}
            return httpx.Response(200, content=body, headers=gzipped)
        return pending.pop(0) if len(pending) > 1 else pending[0]

    client = client_class("US", api_key="test", base_url=BASE)
    client._http_client._transport = httpx.MockTransport(handler)
    return client, calls


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("behaviorgpt._client.time.monotonic", lambda: now[0])
    monkeypatch.setattr(
        "behaviorgpt._client.time.sleep", lambda s: now.__setitem__(0, now[0] + s)
    )
    return now


def test_a_ready_map_is_read_at_once():
    client, calls = serve([joint("ready")])
    assert client.umap(catalog_ids=IDS) == "<html>joint</html>"
    post, page = calls
    assert (post.method, post.url.path) == ("POST", "/v1/maps")
    assert json.loads(post.content) == {"catalog_ids": IDS}
    assert page.url.path == f"/v1/maps/{MAP_ID}/page"


def test_waits_while_the_map_is_drawn(clock):
    client, calls = serve(
        [joint("building", code=202), joint("building"), joint("ready")]
    )
    assert client.umap(catalog_ids=IDS, wait=60) == "<html>joint</html>"
    assert [c.url.path for c in calls] == [
        "/v1/maps",
        f"/v1/maps/{MAP_ID}",
        f"/v1/maps/{MAP_ID}",
        f"/v1/maps/{MAP_ID}/page",
    ]


def test_a_map_still_drawn_at_the_deadline_raises_202(clock):
    client, calls = serve([joint("building", code=202)])
    with pytest.raises(UnboxAIError, match="still being drawn") as exc:
        client.umap(catalog_ids=IDS, wait=12)
    assert exc.value.status_code == 202
    assert exc.value.response["map_id"] == MAP_ID
    assert len(calls) == 3


def test_without_wait_a_map_being_drawn_raises_at_once():
    client, calls = serve([joint("building", code=202)])
    with pytest.raises(UnboxAIError, match="wait="):
        client.umap(catalog_ids=IDS)
    assert len(calls) == 1


def test_a_failed_map_raises_its_reason(clock):
    client, _ = serve([joint("building", code=202), joint("failed", "too few")])
    with pytest.raises(UnboxAIError, match="failed: too few"):
        client.umap(catalog_ids=IDS, wait=60)


def test_catalog_errors_raise_as_they_stand():
    client, _ = serve([httpx.Response(404, json={"detail": "unknown catalog: x"})])
    with pytest.raises(UnboxAIError, match="unknown catalog") as exc:
        client.umap(catalog_ids=["x"])
    assert exc.value.status_code == 404


def test_catalog_id_and_catalog_ids_are_exclusive():
    client, calls = serve([joint("ready")])
    with pytest.raises(ValueError, match="not both"):
        client.umap(catalog_id="cat_a", catalog_ids=IDS)
    with pytest.raises(TypeError):
        client.umap(catalog_ids="cat_a")
    assert calls == []


def test_the_resource_reads_status_and_points():
    client, _ = serve([joint("ready")])
    status = client.catalogs.get_map(MAP_ID)
    assert (status.status, status.catalog_ids) == ("ready", IDS)
    assert status.requested_at is not None
    assert client.catalogs.get_map_points(MAP_ID) == {"catalog": [0, 1]}


def test_async_waits_while_the_map_is_drawn(monkeypatch):
    async def no_sleep(_):
        pass

    monkeypatch.setattr("behaviorgpt._client.asyncio.sleep", no_sleep)
    client, calls = serve(
        [joint("building", code=202), joint("ready")], AsyncUnboxAIClient
    )
    page = asyncio.run(client.umap(catalog_ids=IDS, wait=60))
    assert page == "<html>joint</html>"
    assert len(calls) == 3

    points = asyncio.run(client.catalogs.get_map_points(MAP_ID))
    assert points == {"catalog": [0, 1]}
