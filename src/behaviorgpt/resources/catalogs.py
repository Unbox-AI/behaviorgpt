import time
from pathlib import Path
from typing import Callable, Optional

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from behaviorgpt._exceptions import UnboxAIError
from behaviorgpt.resources._shared import handle_response
from behaviorgpt.types import (
    EmbedJobDetails,
    JobStatus,
    SimilarProductsRequest,
    UnboxAIResponse,
)

# The three states the catalogs.status CHECK constraint allows, server-side.
PENDING_STATUS = "pending"
READY_STATUS = "ready"
FAILED_STATUS = "failed"
QUEUED_STATUS = "queued"

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
    schema = pq.read_schema(path)
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


class Catalogs:
    """Upload a catalog parquet and embed it into the model's product space."""

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
    ) -> EmbedJobDetails:
        """Upload the parquet at `path`.

        The catalog id is minted server-side; `catalog_name` is the file's
        stem and is for display only.

        With `wait=True`, block until the job reaches a terminal state,
        polling every `interval` seconds and giving up after `timeout`.
        `on_progress` is called with each status read while waiting; by
        default each new state is printed once (see `ProgressPrinter`).
        """
        check_catalog_schema(path)

        with open(path, "rb") as f:
            response = self._client.post(
                "/embed-catalog",
                files={"file": (path.name, f, "application/octet-stream")},
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
        """
        if on_progress is None:
            on_progress = ProgressPrinter()

        started = time.monotonic()
        deadline = started + timeout
        seen = False

        while True:
            try:
                status = self.get_job_status(job_id)
            except UnboxAIError as exc:
                not_started = (
                    exc.status_code == 404
                    and not seen
                    and time.monotonic() - started < startup_grace
                )
                if not not_started:
                    raise
                status = JobStatus(job_id=job_id, status=QUEUED_STATUS)
            else:
                seen = True
            state = status.status.lower()

            on_progress(status)

            if state == READY_STATUS:
                return status

            if state == FAILED_STATUS:
                raise UnboxAIError(f"Embed job {job_id} failed")

            if time.monotonic() + interval > deadline:
                raise UnboxAIError(
                    f"Embed job {job_id} still '{state}' after {timeout:g}s"
                )

            time.sleep(interval)

    def get_similar_products(
        self,
        product_id: str,
        limit: int = 10,
        offset: int = 0,
        *,
        catalog_id: str,
        filters: Optional[dict] = None,
        headers: Optional[dict] = None,
    ) -> UnboxAIResponse:
        req = SimilarProductsRequest(
            product_id=product_id,
            store_id=catalog_id,
            limit=limit,
            offset=offset,
            filters=filters or {},
        )

        response = self._client.post(
            "/vector/similar_products",
            json=req.model_dump(mode="json"),
            headers=headers,
        )

        data = handle_response(response)

        return UnboxAIResponse(**data)

    def get_umap(self, *, catalog_id: str):
        response = self._client.get(
            f"/{catalog_id}/umap",
        )

        response.raise_for_status()
        return response.text
