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
