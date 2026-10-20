"""sections() and keywords on embed, against a mock transport: the request
each history makes, the response, and the keywords stage in job statuses."""

import asyncio
import json

import httpx
import pytest

from behaviorgpt import AsyncUnboxAIClient, Search, UnboxAIClient, UnboxAIError, View
from behaviorgpt.resources.catalogs import ProgressPrinter
from behaviorgpt.types import JobStatus

from .test_embed import BASE, scripted, status, wait

SECTIONS = {
    "sections": [
        {
            "title": "black coats",
            "products": [
                {"id": "a", "data": {"name": "A"}, "score": 0.9},
                {"id": "b", "data": {"name": "B"}, "score": 0.5},
            ],
        },
        {
            "title": "wool scarves",
            "products": [{"id": "c", "data": {"name": "C"}, "score": 0.4}],
        },
    ]
}


def mock(client_class, response: httpx.Response | None = None):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response or httpx.Response(200, json=SECTIONS)

    client = client_class("US", api_key="test", base_url=BASE)
    client._http_client._transport = httpx.MockTransport(handler)
    return client, requests


# ---- sections ---------------------------------------------------------------


def test_sections_group_recommendations_after_a_view():
    client, requests = mock(UnboxAIClient)

    response = client.sections([View("sku-1")], catalog_id="cat")

    assert requests[0].url.path == "/v1/vector/sections"
    assert requests[0].headers["x-catalog-id"] == "cat"
    body = json.loads(requests[0].content)
    assert "query" not in body
    assert body["store_id"] == "cat"
    assert body["history"][0]["event"]["product"] == "sku-1"
    assert (
        body["max_sections"],
        body["max_products_per_section"],
        body["min_products_per_section"],
    ) == (10, 25, 5)
    assert response.titles == ["black coats", "wool scarves"]


def test_sections_group_a_search_s_results():
    client, requests = mock(UnboxAIClient)

    client.sections([View("sku-1"), Search("coats")])

    body = json.loads(requests[0].content)
    assert body["query"] == "coats"
    assert len(body["history"]) == 1
    assert body["store_id"] == "retail_catalog"


def test_sections_send_their_limits_and_filters():
    client, requests = mock(UnboxAIClient)

    client.sections(
        [Search("coats")],
        max_sections=3,
        max_items_per_section=8,
        min_items_per_section=2,
        filters={"brand": "Acme"},
    )

    body = json.loads(requests[0].content)
    assert (
        body["max_sections"],
        body["max_products_per_section"],
        body["min_products_per_section"],
    ) == (3, 8, 2)
    assert body["filters"] == {"brand": "Acme"}
    assert body["register_event"] is False


def test_a_section_holds_its_items():
    client, _ = mock(UnboxAIClient)

    first = client.sections([Search("coats")]).sections[0]

    assert first.title == "black coats"
    assert [i.id for i in first.items] == ["a", "b"]
    assert first.items[0].data == {"name": "A"}


def test_a_catalog_without_keywords_has_no_sections():
    client, _ = mock(UnboxAIClient, httpx.Response(200, json={"sections": []}))

    response = client.sections([Search("coats")], catalog_id="cat_mine")

    assert response.sections == [] and response.titles == []


def test_a_sections_error_raises():
    client, _ = mock(
        UnboxAIClient, httpx.Response(404, json={"detail": "unknown catalog: x"})
    )

    with pytest.raises(UnboxAIError, match="unknown catalog") as exc:
        client.sections([Search("coats")], catalog_id="x")
    assert exc.value.status_code == 404


def test_the_async_client_sends_the_same_request():
    sync_client, sync_requests = mock(UnboxAIClient)
    async_client, async_requests = mock(AsyncUnboxAIClient)
    history = [View("sku-1"), Search("coats")]

    sync_client.sections(history, catalog_id="cat", max_sections=4)
    response = asyncio.run(
        async_client.sections(history, catalog_id="cat", max_sections=4)
    )

    sent = [json.loads(r.content) for r in (sync_requests[0], async_requests[0])]
    for body in sent:
        for event in body["history"]:
            event.pop("timestamp"), event.pop("session_id")
    assert sent[0] == sent[1]
    assert async_requests[0].url.path == "/v1/vector/sections"
    assert response.titles == ["black coats", "wool scarves"]


# ---- keywords on embed ------------------------------------------------------


def test_an_upload_asks_for_no_keywords_by_default(catalog_path):
    catalogs, calls = scripted([status("ready")])
    catalogs.embed(catalog_path)
    assert b'name="keywords"' not in calls[0].read()


def test_keywords_are_sent_as_a_form_field(catalog_path):
    catalogs, calls = scripted([status("ready")])
    catalogs.embed(catalog_path, keywords=True)
    body = calls[0].read()
    assert b'name="keywords"' in body
    assert b"true" in body


def test_the_clients_pass_keywords_on(catalog_path):
    client = UnboxAIClient(market="us", api_key="ubx_test", base_url=BASE)
    client.catalogs, calls = scripted([status("ready")])
    client.embed(catalog_path, keywords=True)
    assert b'name="keywords"' in calls[0].read()

    async_client = AsyncUnboxAIClient(market="us", api_key="ubx_test", base_url=BASE)
    from .test_async import scripted as async_scripted

    async_client.catalogs, async_calls = async_scripted([status("ready")])
    asyncio.run(async_client.embed(catalog_path, keywords=True))
    assert b'name="keywords"' in async_calls[0].read()


def test_a_deployment_without_keywords_refuses_the_upload(catalog_path):
    def refuse(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"detail": "keyword generation is not available on this deployment"},
        )

    client = UnboxAIClient(market="us", api_key="ubx_test", base_url=BASE)
    client._http_client._transport = httpx.MockTransport(refuse)
    client.catalogs._client = client._http_client

    with pytest.raises(UnboxAIError, match="not available") as exc:
        client.embed(catalog_path, keywords=True)
    assert exc.value.status_code == 400


# ---- the keywords stage in job statuses -------------------------------------


def keywords_status(state: str, **kw) -> dict:
    phase = state.split()[0]
    return status("pending", stage="top_terms", stage_state=state, phase=phase) | kw


def test_waiting_runs_through_the_keywords_stage():
    result, seen, _ = wait(
        [
            status("pending", stage="embed", phase="embedding"),
            keywords_status(
                "generating 10/100", done=10, total=100, keywords="generating"
            ),
            status("ready", stage="ingest", phase="done", keywords="done"),
        ]
    )
    assert result.status == "ready"
    assert result.keywords == "done" and not result.keywords_skipped
    assert seen[1].describe() == "keywords generating 10/100"


def test_skipped_keywords_still_end_in_a_ready_catalog():
    reason = "the language model did not start"
    result, seen, _ = wait(
        [
            keywords_status("skipped", keywords="skipped", keywords_error=reason),
            status(
                "ready",
                stage="ingest",
                stage_state="done",
                phase="done",
                keywords="skipped",
                keywords_error=reason,
            ),
        ]
    )
    assert result.status == "ready"
    assert result.keywords_skipped
    assert result.keywords_error == reason
    assert seen[0].describe() == f"keywords skipped: {reason}"
    assert result.describe() == f"ingest done (keywords skipped: {reason})"


def test_the_progress_line_names_skipped_keywords_and_rejected_items(capsys):
    ready = JobStatus(
        **status(
            "ready",
            stage="ingest",
            stage_state="done",
            phase="done",
            keywords="skipped",
            products_rejected={
                "count": 1,
                "total": 10,
                "products": [{"id": "x", "reason": "r"}],
            },
        )
    )
    ProgressPrinter()(ready)
    assert capsys.readouterr().out.strip() == (
        "ingest done (keywords skipped; 1 of 10 items rejected: 'x')"
    )


def test_a_server_without_the_fields_reads_as_no_keywords():
    result, _, _ = wait([status("ready")])
    assert result.keywords is None and result.keywords_error is None
    assert not result.keywords_skipped
