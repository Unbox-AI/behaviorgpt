import asyncio
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Optional

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from behaviorgpt._exceptions import UnboxAIError
from behaviorgpt.resources._shared import handle_response
from behaviorgpt.types import (
    AddItemsJobDetails,
    EmbedJobDetails,
    JobStatus,
    SimilarItemsRequest,
    UnboxAIResponse,
)

# The three states the catalogs.status CHECK constraint allows, server-side.
PENDING_STATUS = "pending"
READY_STATUS = "ready"
FAILED_STATUS = "failed"
QUEUED_STATUS = "queued"

# A status read that fails with one of these says nothing about the job.
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# Arrow type per column, see docs/catalog-format.md. The server checks column
# names only; a wrong type fails or degrades the job later, so check it here.
CATALOG_SCHEMA = {
    "id": pa.string(),
    "name": pa.string(),
    "brand": pa.string(),
    "categories": pa.string(),
    "image_url": pa.string(),
    "event_type": pa.string(),
    "group": pa.string(),
    "sales_since": pa.list_(pa.int64()),
    "timestamp": pa.int64(),
    "frequency": pa.int64(),
    "market": pa.string(),
    "price": pa.string(),
    "currency": pa.string(),
    "search_keywords": pa.list_(pa.string()),
    "keywords": pa.list_(pa.string()),
}


def check_catalog_schema(path: Path) -> None:
    """Raise `UnboxAIError` if the parquet's columns do not match `CATALOG_SCHEMA`.

    An entirely null column has Arrow type `null` and is accepted.
    """
    try:
        schema = pq.read_schema(path)
    except (pa.ArrowException, OSError) as exc:
        raise UnboxAIError(f"{path} is not a readable parquet file: {exc}") from exc
    problems = []
    for column, expected in CATALOG_SCHEMA.items():
        if column not in schema.names:
            problems.append(f"{column}: missing")
            continue
        actual = schema.field(column).type
        if actual != expected and not pa.types.is_null(actual):
            problems.append(f"{column}: expected {expected}, got {actual}")
    if problems:
        raise UnboxAIError(
            f"{path} does not match the catalog format:\n  " + "\n  ".join(problems)
        )


def format_reference_time(reference_time: datetime | str) -> str:
    """Format `reference_time` for the upload; a `datetime` must be timezone-aware."""
    if isinstance(reference_time, str):
        return reference_time
    if reference_time.tzinfo is None:
        raise ValueError("reference_time must be timezone-aware")
    return reference_time.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class ProgressPrinter:
    """Default `on_progress` callback: print each new state once.

    `wait_for_job` reads the status every few seconds; most reads repeat the
    previous one, so printing every read would flood a notebook cell.
    """

    def __init__(self) -> None:
        self.last: Optional[str] = None

    def __call__(self, status: JobStatus) -> None:
        line = status.describe()
        if line != self.last:
            print(line)
            self.last = line


def upload_form(path: Path, reference_time: Optional[datetime | str]) -> dict:
    """Check the parquet and return the upload's form fields."""
    check_catalog_schema(path)
    if reference_time is None:
        return {}
    return {"reference_time": format_reference_time(reference_time)}


def items_path(catalog_id: str) -> str:
    if not catalog_id:
        raise ValueError("catalog_id is required to add items")
    return f"/catalogs/{catalog_id}/items"


def similar_items_body(
    item_id: str, catalog_id: str, limit: int, offset: int, filters: Optional[dict]
) -> dict:
    req = SimilarItemsRequest(
        item_id=item_id,
        store_id=catalog_id,
        limit=limit,
        offset=offset,
        filters=filters or {},
    )
    return req.model_dump(mode="json")


def umap_page(response: httpx.Response) -> str:
    if response.is_error:
        handle_response(response)
    return response.text


class JobWatcher:
    """The rules `wait_for_job` polls by, for the sync and the async client alike.

    The caller reads the status, passes the result to `read` or the exception
    to `read_failed`, and hands what they return to `check`, which returns the
    ready status, raises, or returns None to poll again after `interval`.
    """

    def __init__(
        self,
        job_id: str,
        interval: float,
        timeout: float,
        on_progress: Optional[Callable[[JobStatus], None]],
        startup_grace: float,
        failed_grace: float,
    ):
        self.job_id = job_id
        self.interval = interval
        self.timeout = timeout
        self.on_progress = on_progress or ProgressPrinter()
        self.startup_grace = startup_grace
        self.failed_grace = failed_grace
        self.started = time.monotonic()
        self.deadline = self.started + timeout
        self.seen = False
        self.last_state = QUEUED_STATUS
        self.failed_since: Optional[float] = None
        self.read_error: Optional[Exception] = None

    def read(self, status: JobStatus) -> JobStatus:
        self.seen = True
        self.read_error = None
        return status

    def read_failed(self, exc: Exception) -> Optional[JobStatus]:
        """A status that stands in for the failed read, or None; raises the
        errors that end the wait."""
        if isinstance(exc, httpx.TransportError):
            self.read_error = exc
            return None
        if isinstance(exc, UnboxAIError):
            not_started = (
                exc.status_code == 404
                and not self.seen
                and time.monotonic() - self.started < self.startup_grace
            )
            if not_started:
                return JobStatus(job_id=self.job_id, status=QUEUED_STATUS)
            if exc.status_code in RETRYABLE_STATUS_CODES:
                self.read_error = exc
                return None
        raise exc

    def check(self, status: Optional[JobStatus]) -> Optional[JobStatus]:
        if status is not None:
            self.on_progress(status)
            self.last_state = status.status.lower()

            if self.last_state == READY_STATUS:
                return status

            if self.last_state == FAILED_STATUS:
                now = time.monotonic()
                if self.failed_since is None:
                    self.failed_since = now
                if status.error or now - self.failed_since >= self.failed_grace:
                    reason = f": {status.error}" if status.error else ""
                    raise UnboxAIError(
                        f"Embed job {self.job_id} failed{reason}",
                        response=status.model_dump(),
                    )
            else:
                self.failed_since = None

        if time.monotonic() + self.interval > self.deadline:
            message = (
                f"Embed job {self.job_id} still '{self.last_state}' "
                f"after {self.timeout:g}s"
            )
            if self.read_error:
                message += f"; last status read failed: {self.read_error}"
            raise UnboxAIError(message)
        return None


class Catalogs:
    """Upload a catalog parquet and embed it into the model's item space."""

    def __init__(self, http_client: httpx.Client):
        self._client = http_client

    def embed(
        self,
        path: Path,
        wait: bool = False,
        interval: float = 5.0,
        timeout: float = 600.0,
        on_progress: Optional[Callable[[JobStatus], None]] = None,
        startup_grace: float = 120.0,
        reference_time: Optional[datetime | str] = None,
    ) -> EmbedJobDetails:
        """Upload the parquet at `path`.

        The catalog id is minted server-side; `catalog_name` is the file's
        stem and is for display only.

        With `wait=True`, block until the job reaches a terminal state,
        polling every `interval` seconds and giving up after `timeout`.
        `on_progress` is called with each status read while waiting; by
        default each new state is printed once (see `ProgressPrinter`).
        """
        form = upload_form(path, reference_time)

        with open(path, "rb") as f:
            response = self._client.post(
                "/embed-catalog",
                files={"file": (path.name, f, "application/octet-stream")},
                data=form,
            )

        data = handle_response(response)

        job = EmbedJobDetails(**data)

        if wait:
            self.wait_for_job(
                job.job_id,
                interval=interval,
                timeout=timeout,
                on_progress=on_progress,
                startup_grace=startup_grace,
            )

        return job

    def add_items(
        self,
        path: Path,
        *,
        catalog_id: str,
        wait: bool = False,
        interval: float = 5.0,
        timeout: float = 600.0,
        on_progress: Optional[Callable[[JobStatus], None]] = None,
        startup_grace: float = 120.0,
    ) -> AddItemsJobDetails:
        """Upload the parquet at `path` and embed its items into the catalog
        `catalog_id`, without re-embedding the rest of it.

        The parquet has the catalog format. An item whose `id` the catalog
        already has is replaced; any other is added. The catalog keeps serving
        while the job runs, and the job's status reads like an embed's.
        `wait` and the arguments after it work as for `embed`.
        """
        check_catalog_schema(path)

        with open(path, "rb") as f:
            response = self._client.post(
                items_path(catalog_id),
                files={"file": (path.name, f, "application/octet-stream")},
            )

        job = AddItemsJobDetails(**handle_response(response))

        if wait:
            self.wait_for_job(
                job.job_id,
                interval=interval,
                timeout=timeout,
                on_progress=on_progress,
                startup_grace=startup_grace,
            )

        return job

    def get_job_status(self, job_id: str) -> JobStatus:
        """Read the job state once, without blocking."""
        response = self._client.get(f"/embed-catalog/{job_id}")
        data = handle_response(response)
        return JobStatus(**data)

    def wait_for_job(
        self,
        job_id: str,
        interval: float = 5.0,
        timeout: float = 600.0,
        on_progress: Optional[Callable[[JobStatus], None]] = None,
        startup_grace: float = 120.0,
        failed_grace: float = 180.0,
    ) -> JobStatus:
        """Block until the job succeeds; raise on failure or timeout.

        `on_progress` is called with every status read, terminal one included.
        It defaults to a `ProgressPrinter`, which prints each new state once;
        pass your own callable to render progress differently, or
        `lambda _: None` for silence.

        A job reads as 404 until a worker picks it up and writes its first
        state, which can be seconds after the upload returns (or longer if the
        GPU worker is busy). Within `startup_grace` seconds that 404 is
        reported as status `queued` and polling continues; after it, the 404
        is raised as it stands.

        A job that fails for good says why in `error`. A `failed` status
        without an error can be temporary, while the job is retried, so it is
        only final once it has lasted `failed_grace` seconds.

        A status read that fails with a 429, a 5xx or a network error says
        nothing about the job, so polling continues until `timeout`.
        """
        watcher = JobWatcher(
            job_id, interval, timeout, on_progress, startup_grace, failed_grace
        )
        while True:
            try:
                status = watcher.read(self.get_job_status(job_id))
            except (httpx.TransportError, UnboxAIError) as exc:
                status = watcher.read_failed(exc)
            if (ready := watcher.check(status)) is not None:
                return ready
            time.sleep(interval)

    def get_similar_items(
        self,
        item_id: str,
        limit: int = 10,
        offset: int = 0,
        *,
        catalog_id: str,
        filters: Optional[dict] = None,
        headers: Optional[dict] = None,
    ) -> UnboxAIResponse:
        response = self._client.post(
            "/vector/similar_products",
            json=similar_items_body(item_id, catalog_id, limit, offset, filters),
            headers=headers,
        )
        return UnboxAIResponse(**handle_response(response))

    def get_similar_products(
        self, product_id: str, *args: Any, **kwargs: Any
    ) -> UnboxAIResponse:
        warnings.warn(
            "`get_similar_products` is deprecated, use `get_similar_items`",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_similar_items(product_id, *args, **kwargs)

    def get_umap(self, *, catalog_id: str):
        return umap_page(self._client.get(f"/{catalog_id}/umap"))


class AsyncCatalogs:
    """The async counterpart of `Catalogs`; see it for what each method does."""

    def __init__(self, http_client: httpx.AsyncClient):
        self._client = http_client

    async def embed(
        self,
        path: Path,
        wait: bool = False,
        interval: float = 5.0,
        timeout: float = 600.0,
        on_progress: Optional[Callable[[JobStatus], None]] = None,
        startup_grace: float = 120.0,
        reference_time: Optional[datetime | str] = None,
    ) -> EmbedJobDetails:
        form = upload_form(path, reference_time)

        with open(path, "rb") as f:
            response = await self._client.post(
                "/embed-catalog",
                files={"file": (path.name, f, "application/octet-stream")},
                data=form,
            )

        job = EmbedJobDetails(**handle_response(response))

        if wait:
            await self.wait_for_job(
                job.job_id,
                interval=interval,
                timeout=timeout,
                on_progress=on_progress,
                startup_grace=startup_grace,
            )

        return job

    async def add_items(
        self,
        path: Path,
        *,
        catalog_id: str,
        wait: bool = False,
        interval: float = 5.0,
        timeout: float = 600.0,
        on_progress: Optional[Callable[[JobStatus], None]] = None,
        startup_grace: float = 120.0,
    ) -> AddItemsJobDetails:
        check_catalog_schema(path)

        with open(path, "rb") as f:
            response = await self._client.post(
                items_path(catalog_id),
                files={"file": (path.name, f, "application/octet-stream")},
            )

        job = AddItemsJobDetails(**handle_response(response))

        if wait:
            await self.wait_for_job(
                job.job_id,
                interval=interval,
                timeout=timeout,
                on_progress=on_progress,
                startup_grace=startup_grace,
            )

        return job

    async def get_job_status(self, job_id: str) -> JobStatus:
        response = await self._client.get(f"/embed-catalog/{job_id}")
        return JobStatus(**handle_response(response))

    async def wait_for_job(
        self,
        job_id: str,
        interval: float = 5.0,
        timeout: float = 600.0,
        on_progress: Optional[Callable[[JobStatus], None]] = None,
        startup_grace: float = 120.0,
        failed_grace: float = 180.0,
    ) -> JobStatus:
        watcher = JobWatcher(
            job_id, interval, timeout, on_progress, startup_grace, failed_grace
        )
        while True:
            try:
                status = watcher.read(await self.get_job_status(job_id))
            except (httpx.TransportError, UnboxAIError) as exc:
                status = watcher.read_failed(exc)
            if (ready := watcher.check(status)) is not None:
                return ready
            await asyncio.sleep(interval)

    async def get_similar_items(
        self,
        item_id: str,
        limit: int = 10,
        offset: int = 0,
        *,
        catalog_id: str,
        filters: Optional[dict] = None,
        headers: Optional[dict] = None,
    ) -> UnboxAIResponse:
        response = await self._client.post(
            "/vector/similar_products",
            json=similar_items_body(item_id, catalog_id, limit, offset, filters),
            headers=headers,
        )
        return UnboxAIResponse(**handle_response(response))

    async def get_umap(self, *, catalog_id: str) -> str:
        return umap_page(await self._client.get(f"/{catalog_id}/umap"))
