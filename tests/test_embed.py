"""Catalogs.embed and wait_for_job against scripted server behaviour.

The scripts replay what a client can meet: a 404 until the job starts, a
temporary `failed` without an error, 5xx responses and dropped connections
while polling, and a plain-text 413 from a proxy for an oversized upload.
"""

import httpx
import pytest

from behaviorgpt._exceptions import UnboxAIError
from behaviorgpt.resources.catalogs import Catalogs

BASE = "http://sdk.test/v1"
UPLOADED = {
    "job_id": "job-1",
    "catalog_id": "cat_0123456789abcdef01234567",
    "catalog_name": "catalog",
    "uri": "https://blob.example/raw-catalogs/cat_0123456789abcdef01234567.parquet",
}


def status(state: str, **kw) -> dict:
    return {
        "job_id": "job-1",
        "status": state,
        "stage": "embed",
        "stage_state": state,
    } | kw


def scripted(steps: list):
    """Each status GET consumes one step (the last repeats): a dict is a 200
    body, an int a status with a JSON detail, an httpx.Response is returned
    as is, an exception is raised."""
    calls: list[httpx.Request] = []
    steps = list(steps)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.method == "POST":
            return httpx.Response(201, json=UPLOADED)
        step = steps.pop(0) if len(steps) > 1 else steps[0]
        if isinstance(step, BaseException):
            raise step
        if isinstance(step, httpx.Response):
            return step
        if isinstance(step, int):
            return httpx.Response(step, json={"detail": f"http {step}"})
        return httpx.Response(200, json=step)

    client = httpx.Client(base_url=BASE, transport=httpx.MockTransport(handler))
    return Catalogs(client), calls


def wait(steps, **kw):
    catalogs, calls = scripted(steps)
    seen = []
    kw = {"interval": 0.001, "timeout": 5, "on_progress": seen.append} | kw
    return catalogs.wait_for_job("job-1", **kw), seen, calls


# ---- polling ----------------------------------------------------------------


def test_startup_404_reads_queued_then_ready():
    result, seen, _ = wait([404, 404, status("fetching 10/100"), status("ready")])
    assert result.status == "ready"
    assert [s.status for s in seen][:2] == ["queued", "queued"]


def test_404_after_a_successful_read_raises():
    with pytest.raises(UnboxAIError) as exc:
        wait([status("pending"), 404])
    assert exc.value.status_code == 404


def test_404_past_startup_grace_raises():
    with pytest.raises(UnboxAIError):
        wait([404], startup_grace=0)


def test_failure_raises_with_the_reason():
    reason = "catalog parquet is missing columns: ['frequency']"
    with pytest.raises(UnboxAIError) as exc:
        wait([status("pending"), status("failed", error=reason)])
    assert reason in str(exc.value)
    assert exc.value.response["error"] == reason


def test_terminal_state_is_case_insensitive():
    result, _, _ = wait([status("READY")])
    assert result.status == "READY"


def test_timeout_names_the_last_state():
    with pytest.raises(UnboxAIError, match="still 'pending'"):
        wait([status("pending")], timeout=0.05, interval=0.01)


def test_progress_callback_sees_every_read():
    _, seen, calls = wait([status("reading"), status("embedding"), status("ready")])
    assert [s.status for s in seen] == ["reading", "embedding", "ready"]
    assert len(calls) == 3


def test_failed_without_reason_is_not_final():
    result, _, _ = wait(
        [status("pending"), status("failed"), status("processing"), status("ready")]
    )
    assert result.status == "ready"


def test_failed_without_reason_is_final_after_the_grace():
    with pytest.raises(UnboxAIError, match="failed$"):
        wait([status("pending"), status("failed")], failed_grace=0.02, interval=0.01)


def test_failed_with_reason_is_final_at_once():
    with pytest.raises(UnboxAIError, match="bad file"):
        wait([status("failed", error="bad file"), status("ready")])


@pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
def test_polling_survives_a_transient_status(code):
    result, _, _ = wait([status("pending"), code, code, status("ready")])
    assert result.status == "ready"


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("reset"),
        httpx.ReadTimeout("slow"),
        httpx.RemoteProtocolError("eof"),
    ],
    ids=["connect", "read-timeout", "protocol"],
)
def test_polling_survives_a_network_error(error):
    result, _, _ = wait([status("pending"), error, status("ready")])
    assert result.status == "ready"


def test_timeout_names_the_last_read_error():
    with pytest.raises(UnboxAIError, match="still 'pending'.*last status read failed"):
        wait([status("pending"), 503], timeout=0.05, interval=0.01)


@pytest.mark.parametrize("code", [400, 401, 403, 422])
def test_client_errors_stop_polling(code):
    with pytest.raises(UnboxAIError) as exc:
        wait([status("pending"), code, status("ready")])
    assert exc.value.status_code == code


def test_non_dict_error_body_is_an_sdk_error():
    with pytest.raises(UnboxAIError) as exc:
        wait([status("pending"), httpx.Response(400, json=["bad", "request"])])
    assert exc.value.status_code == 400
    assert "bad" in str(exc.value)


# ---- upload -----------------------------------------------------------------


def test_upload_is_a_multipart_file_field(catalog_path):
    catalogs, calls = scripted([status("ready")])
    job = catalogs.embed(catalog_path)
    assert job.catalog_id == UPLOADED["catalog_id"]
    post = calls[0]
    assert post.url.path == "/v1/embed-catalog"
    assert b'name="file"; filename="catalog.parquet"' in post.read()


def test_bad_schema_is_refused_before_any_request(write_catalog):
    from .conftest import catalog_table

    path = write_catalog(catalog_table().drop_columns(["frequency"]))
    catalogs, calls = scripted([status("ready")])
    with pytest.raises(UnboxAIError, match="frequency: missing"):
        catalogs.embed(path)
    assert calls == []


def test_ingress_413_is_an_sdk_error_with_status(catalog_path):
    def reject(request):
        return httpx.Response(413, text="Request Entity Too Large")

    catalogs = Catalogs(
        httpx.Client(base_url=BASE, transport=httpx.MockTransport(reject))
    )
    with pytest.raises(UnboxAIError) as exc:
        catalogs.embed(catalog_path)
    assert exc.value.status_code == 413
    assert "Too Large" in str(exc.value)


def test_embed_wait_returns_the_job(catalog_path):
    catalogs, _ = scripted([404, status("embedding"), status("ready")])
    job = catalogs.embed(
        catalog_path, wait=True, interval=0.001, on_progress=lambda _: None
    )
    assert job.job_id == "job-1"


def test_client_timeout_allows_a_large_upload():
    from behaviorgpt import UnboxAIClient

    timeout = UnboxAIClient(market="us", api_key="ubx_test")._http_client.timeout
    assert timeout.write >= 60 and timeout.read >= 60


def test_client_timeout_is_configurable():
    from behaviorgpt import UnboxAIClient

    client = UnboxAIClient(market="us", api_key="ubx_test", http_timeout=7.0)
    assert client._http_client.timeout.write == 7.0
