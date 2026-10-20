"""add_keywords, for the sync and the async client, and the progress bar
`wait=True` draws by default."""

import asyncio
import io
import sys

import httpx
import pytest

from behaviorgpt import AsyncUnboxAIClient, ProgressBar, UnboxAIClient, UnboxAIError
from behaviorgpt.resources import catalogs as catalogs_module
from behaviorgpt.types import AddKeywordsJobDetails, JobStatus

from .test_embed import BASE

CATALOG = "cat_0123456789abcdef01234567"
JOB = f"{CATALOG}-keywords-0badcafe"
ACCEPTED = {"job_id": JOB, "catalog_id": CATALOG, "catalog_name": "shop"}


def job_status(state: str, **kw) -> dict:
    return {"job_id": JOB, "status": state} | kw


GENERATING = job_status(
    "pending",
    stage="top_terms",
    stage_state="generating 10/100",
    phase="generating",
    done=10,
    total=100,
    keywords="generating",
)
READY = job_status(
    "ready", stage="ingest", stage_state="done", phase="done", keywords="done"
)


def handler(calls: list[httpx.Request], statuses: list[dict]):
    statuses = list(statuses)

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.method == "POST":
            return httpx.Response(202, json=ACCEPTED)
        body = statuses.pop(0) if len(statuses) > 1 else statuses[0]
        return httpx.Response(200, json=body)

    return handle


def sync_client(statuses=(READY,)):
    calls: list[httpx.Request] = []
    client = UnboxAIClient(market="us", api_key="ubx_test", base_url=BASE)
    client._http_client._transport = httpx.MockTransport(handler(calls, statuses))
    return client, calls


def test_add_keywords_starts_a_job_for_the_catalog():
    client, calls = sync_client()
    job = client.add_keywords(CATALOG)

    assert job == AddKeywordsJobDetails(**ACCEPTED)
    (post,) = calls
    assert post.method == "POST"
    assert post.url.path == f"/v1/catalogs/{CATALOG}/keywords"
    assert post.content == b""


def test_add_keywords_waits_through_the_keywords_and_the_ingest():
    client, calls = sync_client([GENERATING, READY])
    seen: list[JobStatus] = []
    client.add_keywords(CATALOG, wait=True, interval=0.001, on_progress=seen.append)

    assert [s.stage for s in seen] == ["top_terms", "ingest"]
    assert seen[-1].status == "ready"
    assert all(c.url.path == f"/v1/embed-catalog/{JOB}" for c in calls[1:])


def test_a_failed_keywords_job_raises_with_its_reason():
    reason = "the catalog was uploaded again while its keywords were being generated"
    failed = job_status(
        "failed", stage="top_terms", stage_state="failed", phase="failed", error=reason
    )
    client, _ = sync_client([failed])
    with pytest.raises(UnboxAIError, match=reason):
        client.add_keywords(
            CATALOG, wait=True, interval=0.001, on_progress=lambda _: None
        )


def test_add_keywords_needs_a_catalog():
    client, calls = sync_client()
    with pytest.raises(ValueError, match="catalog_id"):
        client.add_keywords("")
    assert calls == []


def test_add_keywords_async():
    async def run():
        calls: list[httpx.Request] = []
        client = AsyncUnboxAIClient("us", api_key="ubx_test", base_url=BASE)
        client._http_client._transport = httpx.MockTransport(
            handler(calls, [GENERATING, READY])
        )
        async with client:
            job = await client.add_keywords(
                CATALOG, wait=True, interval=0.001, on_progress=lambda _: None
            )
        return job, calls

    job, calls = asyncio.run(run())
    assert job.job_id == JOB
    assert calls[0].url.path == f"/v1/catalogs/{CATALOG}/keywords"
    assert len(calls) == 3


# The progress bar


class Terminal(io.StringIO):
    def __init__(self, encoding: str = "utf-8") -> None:
        super().__init__()
        self._encoding = encoding

    @property
    def encoding(self) -> str:
        return self._encoding

    def isatty(self) -> bool:
        return True

    def screen(self) -> list[str]:
        """What is left on screen: every `\\r` redraws its line."""
        return [line.split("\r")[-1].rstrip() for line in self.getvalue().split("\n")]


def counts(done: int, total: int = 100, phase: str = "generating") -> JobStatus:
    return JobStatus(
        **job_status(
            "pending",
            stage="top_terms",
            stage_state=f"{phase} {done}/{total}",
            phase=phase,
            done=done,
            total=total,
        )
    )


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(catalogs_module.time, "monotonic", lambda: now[0])
    return now


def test_the_bar_redraws_one_line_and_estimates_the_time_left(clock):
    out = Terminal()
    bar = ProgressBar(out)
    bar(counts(10))
    clock[0] += 60
    bar(counts(30))

    (line,) = out.screen()
    assert line.startswith("keywords generating |")
    assert "30%" in line and "30/100" in line
    # 20 items a minute, 70 to go
    assert line.endswith("3m30s left")
    assert "█" in line and "░" in line


def test_each_step_keeps_its_own_line_and_ready_ends_it(clock):
    out = Terminal()
    bar = ProgressBar(out)
    bar(counts(50, phase="fetching"))
    bar(counts(5))
    clock[0] += 3
    bar(JobStatus(**READY))

    fetching, generating, ready, after = out.screen()
    assert fetching.startswith("keywords fetching |") and "50/100" in fetching
    assert generating.startswith("keywords generating |")
    assert ready.startswith("ingest done")
    assert after == ""


def test_a_shorter_line_wipes_the_longer_one_it_replaces(clock):
    out = Terminal()
    bar = ProgressBar(out)
    bar(counts(10))
    bar(counts(10))
    first, second = out.getvalue().split("\r")[1:]
    assert len(second) >= len(first)


def test_a_step_without_counts_shows_how_long_it_has_run(clock):
    out = Terminal()
    bar = ProgressBar(out)
    embedding = JobStatus(
        **job_status("pending", stage="embed", stage_state="embedding")
    )
    bar(embedding)
    clock[0] += 75
    bar(embedding)
    assert out.screen() == ["embed embedding  1m15s"]


def test_a_failure_prints_its_reason_and_ends_the_line(clock):
    out = Terminal()
    failed = job_status("failed", stage="top_terms", phase="failed", error="no model")
    ProgressBar(out)(JobStatus(**failed))
    assert out.screen() == ["keywords failed: no model", ""]


def test_a_console_without_unicode_gets_a_plain_bar(clock):
    out = Terminal(encoding="ascii")
    ProgressBar(out)(counts(50))
    (line,) = out.screen()
    assert "|############------------|" in line


def test_outside_a_terminal_each_state_is_printed_once(capsys):
    bar = ProgressBar()
    for status in (counts(10), counts(10), counts(20)):
        bar(status)
    assert capsys.readouterr().out.splitlines() == [
        "keywords generating 10/100",
        "keywords generating 20/100",
    ]


def test_a_notebook_gets_the_bar(monkeypatch, clock):
    out = io.StringIO()
    monkeypatch.setitem(sys.modules, "ipykernel", object())
    ProgressBar(out)(counts(10))
    assert out.getvalue().startswith("\rkeywords generating |")


def test_waiting_draws_the_bar_by_default(monkeypatch):
    client, _ = sync_client([GENERATING, READY])
    out = Terminal()
    monkeypatch.setattr(sys, "stdout", out)
    client.add_keywords(CATALOG, wait=True, interval=0.001)
    lines = out.screen()
    assert lines[0].startswith("keywords generating |")
    assert lines[1].startswith("ingest done")
