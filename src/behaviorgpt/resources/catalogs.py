import ast
import asyncio
import sys
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Optional, TextIO
from urllib.parse import quote

import httpx
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq

from behaviorgpt._exceptions import UnboxAIError
from behaviorgpt.resources._shared import handle_response
from behaviorgpt.types import (
    AddItemsJobDetails,
    AddKeywordsJobDetails,
    CatalogSummary,
    EmbedJobDetails,
    JobStatus,
    SimilarItemsRequest,
    UnboxAIResponse,
)
from behaviorgpt.types.http import STAGE_LABELS

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


def _plain_type(t: pa.DataType) -> pa.DataType:
    """The embedder casts large, view and dictionary layouts to their plain
    forms before it checks a type, so the check here does the same."""
    if pa.types.is_dictionary(t):
        return _plain_type(t.value_type)
    if pa.types.is_large_string(t) or pa.types.is_string_view(t):
        return pa.string()
    if pa.types.is_large_binary(t) or pa.types.is_binary_view(t):
        return pa.binary()
    if (
        pa.types.is_list(t)
        or pa.types.is_large_list(t)
        or pa.types.is_list_view(t)
        or pa.types.is_large_list_view(t)
    ):
        return pa.list_(_plain_type(t.value_type))
    return t


def check_catalog_schema(path: Path) -> None:
    """Raise `UnboxAIError` if the parquet's columns do not match `CATALOG_SCHEMA`.

    An entirely null column has Arrow type `null` and is accepted, and so are
    the large, view and dictionary layouts of a type, which pandas 3 and polars
    write by default.
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
        if _plain_type(actual) != expected and not pa.types.is_null(actual):
            problems.append(f"{column}: expected {expected}, got {actual}")
    if problems:
        raise UnboxAIError(
            f"{path} does not match the catalog format:\n  " + "\n  ".join(problems)
        )


LIST_COLUMNS: dict[str, type] = {
    "sales_since": int,
    "search_keywords": str,
    "keywords": str,
}


def _parse_list(column: str, cell: Optional[str]) -> Optional[list]:
    """`[1, 2]` or `a, b` as a list of the column's item type; empty is null."""
    if cell is None or not cell.strip():
        return None
    cast = LIST_COLUMNS[column]
    try:
        items = (
            ast.literal_eval(cell) if cell.strip().startswith("[") else cell.split(",")
        )
        return [cast(str(item).strip()) for item in items if str(item).strip()]
    except (ValueError, SyntaxError) as exc:
        raise UnboxAIError(f"{column}: cannot read {cell!r} as a list") from exc


def catalog_from_csv(
    csv_path: Path | str, parquet_path: Optional[Path | str] = None
) -> Path:
    """Convert a CSV with the columns of docs/catalog-format.md to a catalog
    parquet, check it and return its path.

    Every cell is read as text, so prices keep their formatting. List columns
    take `[1, 2]` or `a, b`. Empty cells, and columns the CSV lacks, become
    null; other columns are left out. Duplicate ids keep their first row. The
    parquet goes next to the CSV unless `parquet_path` is given; its file name
    becomes the catalog's name.
    """
    csv_path = Path(csv_path)
    out = Path(parquet_path) if parquet_path else csv_path.with_suffix(".parquet")
    try:
        table = pa_csv.read_csv(
            csv_path,
            convert_options=pa_csv.ConvertOptions(
                column_types={column: pa.string() for column in CATALOG_SCHEMA},
                include_columns=list(CATALOG_SCHEMA),
                include_missing_columns=True,
                strings_can_be_null=True,
            ),
        )
    except pa.ArrowInvalid as exc:
        raise UnboxAIError(f"{csv_path} is not a readable CSV file: {exc}") from exc

    first_rows = {}
    for row, item_id in enumerate(table.column("id").to_pylist()):
        first_rows.setdefault(item_id, row)
    table = table.take(sorted(first_rows.values()))

    columns = {}
    for column, expected in CATALOG_SCHEMA.items():
        values = table.column(column)
        if column in LIST_COLUMNS:
            values = pa.array(
                [_parse_list(column, cell) for cell in values.to_pylist()],
                type=expected,
            )
        elif expected != pa.string():
            try:
                values = pc.cast(values, expected)
            except pa.ArrowInvalid as exc:
                raise UnboxAIError(f"{column}: expected {expected}: {exc}") from exc
        columns[column] = values

    pq.write_table(pa.table(columns, schema=pa.schema(CATALOG_SCHEMA)), out)
    check_catalog_schema(out)
    return out


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


def _interactive(stream: TextIO) -> bool:
    """A terminal or a notebook, where `\\r` rewrites the line. Not a log
    file, a CI job or a pipe, where it would leave every frame behind."""
    if "ipykernel" in sys.modules:
        return True
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds}s"


class ProgressBar:
    """Default `on_progress` callback: a progress bar with the step's items
    done, its share and the time left, redrawn in place, for example

        keywords generating |██████░░░░░░░░░░░░|  33%  400/1200  2m10s left

    Each step (embed fetching, keywords generating, ...) gets its own line,
    finished when the next one starts. A step that reports no counts shows
    how long it has run instead. Where the output is not a terminal or a
    notebook, each new state is printed once, as `ProgressPrinter` does.
    """

    WIDTH = 24

    def __init__(self, stream: Optional[TextIO] = None) -> None:
        self.stream = stream
        self.lines = ProgressPrinter()
        self.step: Optional[tuple] = None
        self.step_started = 0.0
        # the first count seen in this step, as (time, done), for the rate
        self.first: Optional[tuple[float, int]] = None
        self.drawn = 0

    @property
    def out(self) -> TextIO:
        return self.stream or sys.stdout

    def _blocks(self) -> tuple[str, str]:
        try:
            "█░".encode(getattr(self.out, "encoding", None) or "ascii")
        except (UnicodeEncodeError, LookupError):
            return "#", "-"
        return "█", "░"

    def render(self, status: JobStatus, now: float) -> str:
        step = (status.stage, status.phase or status.stage_state)
        if step != self.step:
            self.step, self.step_started, self.first = step, now, None
        done, total = status.done, status.total
        label = STAGE_LABELS.get(status.stage or "", status.stage or status.status)
        name = f"{label} {status.phase or status.stage_state or ''}".strip()
        if status.status.lower() in (READY_STATUS, FAILED_STATUS):
            return status.describe()
        if done is None or not total:
            return f"{status.describe()}  {_duration(now - self.step_started)}"
        if self.first is None or done < self.first[1]:
            self.first = (now, done)
        share = min(max(done / total, 0.0), 1.0)
        full, empty = self._blocks()
        filled = round(share * self.WIDTH)
        bar = full * filled + empty * (self.WIDTH - filled)
        line = f"{name} |{bar}| {share:4.0%}  {done}/{total}"
        since, done_then = self.first
        if done > done_then and now > since and done < total:
            left = (total - done) * (now - since) / (done - done_then)
            line += f"  {_duration(left)} left"
        return line

    def __call__(self, status: JobStatus) -> None:
        if not _interactive(self.out):
            self.lines(status)
            return
        now = time.monotonic()
        previous = self.step
        line = self.render(status, now)
        if previous is not None and self.step != previous and self.drawn:
            self.out.write("\n")
            self.drawn = 0
        self.out.write("\r" + line.ljust(self.drawn))
        self.drawn = len(line)
        state = status.status.lower()
        if state == READY_STATUS or (state == FAILED_STATUS and status.error):
            self.out.write("\n")
            self.drawn = 0
        self.out.flush()


def upload_form(
    path: Path, reference_time: Optional[datetime | str], keywords: bool = False
) -> dict:
    """Check the parquet and return the upload's form fields."""
    check_catalog_schema(path)
    form = {}
    if reference_time is not None:
        form["reference_time"] = format_reference_time(reference_time)
    if keywords:
        form["keywords"] = "true"
    return form


def keywords_path(catalog_id: str) -> str:
    if not catalog_id:
        raise ValueError("catalog_id is required to add keywords")
    return f"/catalogs/{quote(catalog_id, safe='')}/keywords"


def items_path(catalog_id: str) -> str:
    if not catalog_id:
        raise ValueError("catalog_id is required to add items")
    return f"/catalogs/{catalog_id}/items"


def catalog_path(catalog_id: str) -> str:
    if not catalog_id:
        raise ValueError("catalog_id is required to delete a catalog")
    return f"/catalogs/{quote(catalog_id, safe='')}"


def catalog_summaries(response: httpx.Response) -> list[CatalogSummary]:
    return [CatalogSummary(**c) for c in handle_response(response)["catalogs"]]


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
        self.on_progress = on_progress or ProgressBar()
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
                        f"Job {self.job_id} failed{reason}",
                        response=status.model_dump(),
                    )
            else:
                self.failed_since = None

        if time.monotonic() + self.interval > self.deadline:
            message = (
                f"Job {self.job_id} still '{self.last_state}' after {self.timeout:g}s"
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
        keywords: bool = False,
    ) -> EmbedJobDetails:
        """Upload the parquet at `path`.

        The catalog id is minted server-side; `catalog_name` is the file's
        stem and is for display only.

        With `wait=True`, block until the job reaches a terminal state,
        polling every `interval` seconds and giving up after `timeout`.
        `on_progress` is called with each status read while waiting; by
        default a progress bar is drawn (see `ProgressBar`).

        `keywords=True` also generates search keywords and browse categories
        for the items, which `sections` groups by. It adds time before the
        catalog is ready, so allow a longer `timeout`. If generation fails,
        the catalog still goes live, without keywords: the job status then
        reads `keywords_skipped`, with the reason in `keywords_error`.
        """
        form = upload_form(path, reference_time, keywords)

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

    def add_keywords(
        self,
        catalog_id: str,
        *,
        wait: bool = False,
        interval: float = 5.0,
        timeout: float = 4 * 3600.0,
        on_progress: Optional[Callable[[JobStatus], None]] = None,
        startup_grace: float = 120.0,
    ) -> AddKeywordsJobDetails:
        """Generate search keywords and browse categories for the ready
        catalog `catalog_id`, from the items it already has; nothing is
        re-embedded.

        The catalog keeps serving while the job runs and gets the keywords
        once it is ready. If the job fails, the catalog stays as it was.
        `wait` and the arguments after it work as for `embed`.
        """
        response = self._client.post(keywords_path(catalog_id))
        job = AddKeywordsJobDetails(**handle_response(response))

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
        It defaults to a `ProgressBar`, which draws a bar in a terminal or a
        notebook and prints each new state once elsewhere; pass
        `ProgressPrinter()` for plain lines, your own callable to render
        progress differently, or `lambda _: None` for silence.

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

    def list_catalogs(self) -> list[CatalogSummary]:
        return catalog_summaries(self._client.get("/catalogs"))

    def delete_catalog(self, catalog_id: str) -> None:
        handle_response(self._client.delete(catalog_path(catalog_id)))


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
        keywords: bool = False,
    ) -> EmbedJobDetails:
        form = upload_form(path, reference_time, keywords)

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

    async def add_keywords(
        self,
        catalog_id: str,
        *,
        wait: bool = False,
        interval: float = 5.0,
        timeout: float = 4 * 3600.0,
        on_progress: Optional[Callable[[JobStatus], None]] = None,
        startup_grace: float = 120.0,
    ) -> AddKeywordsJobDetails:
        response = await self._client.post(keywords_path(catalog_id))
        job = AddKeywordsJobDetails(**handle_response(response))

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

    async def list_catalogs(self) -> list[CatalogSummary]:
        return catalog_summaries(await self._client.get("/catalogs"))

    async def delete_catalog(self, catalog_id: str) -> None:
        handle_response(await self._client.delete(catalog_path(catalog_id)))
