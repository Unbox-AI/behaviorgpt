import httpx
import pytest

from behaviorgpt import UnboxAIClient
from behaviorgpt._exceptions import UnboxAIError


def client_answering(response: httpx.Response) -> UnboxAIClient:
    client = UnboxAIClient("US", api_key="test", base_url="http://sdk.test/v1")
    client._http_client._transport = httpx.MockTransport(lambda _: response)
    return client


def test_umap_returns_the_page():
    client = client_answering(httpx.Response(200, text="<html>map</html>"))
    assert client.umap(catalog_id="cat") == "<html>map</html>"


def test_umap_error_raises_unboxai_error():
    client = client_answering(
        httpx.Response(404, json={"detail": "unknown catalog: cat"})
    )
    with pytest.raises(UnboxAIError, match="unknown catalog") as exc:
        client.umap(catalog_id="cat")
    assert exc.value.status_code == 404


def map_after(misses: int) -> list[httpx.Response]:
    """A UMAP endpoint that 404s `misses` times, then serves the page."""
    return [
        httpx.Response(404, json={"detail": "No UMAP for this catalog."})
    ] * misses + [httpx.Response(200, text="<html>map</html>")]


def client_answering_in_turn(
    responses: list[httpx.Response],
) -> tuple[UnboxAIClient, list]:
    client = UnboxAIClient("US", api_key="test", base_url="http://sdk.test/v1")
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return responses[min(len(calls), len(responses)) - 1]

    client._http_client._transport = httpx.MockTransport(handler)
    return client, calls


def test_umap_wait_asks_again_until_the_map_is_built(monkeypatch):
    monkeypatch.setattr("behaviorgpt._client.time.sleep", lambda _: None)
    client, calls = client_answering_in_turn(map_after(2))
    assert client.umap(catalog_id="cat", wait=60) == "<html>map</html>"
    assert len(calls) == 3


def test_umap_wait_gives_up_at_the_deadline(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("behaviorgpt._client.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "behaviorgpt._client.time.sleep", lambda s: clock.__setitem__(0, clock[0] + s)
    )
    client, calls = client_answering_in_turn(map_after(100))
    with pytest.raises(UnboxAIError, match="No UMAP"):
        client.umap(catalog_id="cat", wait=12)
    assert len(calls) == 3


def test_umap_without_wait_asks_once():
    client, calls = client_answering_in_turn(map_after(1))
    with pytest.raises(UnboxAIError):
        client.umap(catalog_id="cat")
    assert len(calls) == 1


def test_umap_wait_does_not_retry_other_errors(monkeypatch):
    monkeypatch.setattr("behaviorgpt._client.time.sleep", lambda _: None)
    client, calls = client_answering_in_turn(
        [httpx.Response(401, json={"detail": "bad key"})]
    )
    with pytest.raises(UnboxAIError):
        client.umap(catalog_id="cat", wait=60)
    assert len(calls) == 1


def test_error_classes_import_from_the_package_root():
    import behaviorgpt

    for name in ["AuthenticationError", "BadRequestError", "RateLimitError"]:
        assert issubclass(getattr(behaviorgpt, name), behaviorgpt.UnboxAIError)
    assert behaviorgpt.UnboxAIError is UnboxAIError
