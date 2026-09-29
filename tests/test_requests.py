"""Recommendation requests: the optional prediction time and candidate count."""

import json
from datetime import UTC, datetime

import httpx
import pytest
from pydantic import ValidationError

from behaviorgpt import UnboxAIClient

PAGE = {"products": {"items": [], "offset": 0, "limit": 10}}


def sent(**fields) -> dict:
    """The JSON body the client posts for a request made with `fields`."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=PAGE)

    client = UnboxAIClient("US", api_key="test", base_url="http://sdk.test/v1")
    client._http_client._transport = httpx.MockTransport(handler)
    client.recommendations.create(payload=client.requester.make("cat", **fields))
    return json.loads(requests[0].content)


def test_unset_fields_are_left_out():
    body = sent()
    assert "timestamp" not in body and "num_candidates" not in body


def test_timestamp_and_candidate_count_are_sent():
    moment = datetime(2021, 11, 12, 13, tzinfo=UTC)
    body = sent(timestamp=moment, num_candidates=500)
    assert body["timestamp"] == "2021-11-12T13:00:00Z"
    assert body["num_candidates"] == 500


def test_naive_timestamp_is_refused():
    with pytest.raises(ValidationError, match="timezone-aware"):
        sent(timestamp=datetime(2021, 11, 12, 13))


@pytest.mark.parametrize("count", [0, 5_001])
def test_candidate_count_out_of_range_is_refused(count):
    with pytest.raises(ValidationError):
        sent(num_candidates=count)
