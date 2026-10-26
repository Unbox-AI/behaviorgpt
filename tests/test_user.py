"""Who the shopper is (`User`), sent with the prediction and every event of the
history: gender, age group, market and time zone, never a postal code."""

import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from behaviorgpt import AsyncUnboxAIClient, Search, UnboxAIClient, User, View
from behaviorgpt.types.user import AGE_GROUPS, age_group

PAGE = {"products": {"items": [], "offset": 0, "limit": 10}}
SECTIONS = {"sections": []}


def capture(client) -> list[httpx.Request]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = SECTIONS if request.url.path.endswith("/sections") else PAGE
        return httpx.Response(200, json=body)

    client._http_client._transport = httpx.MockTransport(handler)
    return requests


def client() -> UnboxAIClient:
    return UnboxAIClient(
        "US", api_key="test", base_url="http://sdk.test/v1", timezone="America/New_York"
    )


def test_without_a_user_the_clients_market_and_time_zone_are_sent():
    c = client()
    requests = capture(c)
    c.complete([View("a")])
    body = json.loads(requests[0].content)
    assert body["domains"] == {
        "market": "US", "postal_code": "unk", "gender": "unk", "age_group": "unk"
    }  # fmt: skip
    assert body["timezone"] == "America/New_York"
    assert body["history"][0]["domains"] == body["domains"]


@pytest.mark.parametrize("ends_in_search", [False, True])
def test_a_user_goes_with_the_prediction_and_its_history(ends_in_search: bool):
    c = client()
    requests = capture(c)
    user = User(gender="Female", age=34, market="SE", timezone="Europe/Stockholm")
    history = [View("a"), View("b")] + ([Search("sofa")] if ends_in_search else [])
    c.complete(history, user=user)
    body = json.loads(requests[0].content)
    assert requests[0].url.path.endswith(
        "/search" if ends_in_search else "/recommendations"
    )
    expected = {
        "market": "SE", "postal_code": "unk", "gender": "female", "age_group": "26-35"
    }  # fmt: skip
    assert body["domains"] == expected
    assert body["timezone"] == "Europe/Stockholm"
    assert [e["domains"] for e in body["history"]] == [expected, expected]
    assert {e["timezone"] for e in body["history"]} == {"Europe/Stockholm"}
    # the age itself is never sent
    assert '"age":' not in requests[0].content.decode()


def test_a_partial_user_keeps_the_clients_market_and_time_zone():
    c = client()
    requests = capture(c)
    c.sections([View("a")], user=User(age_group="66-75"))
    body = json.loads(requests[0].content)
    assert body["domains"] == {
        "market": "US", "postal_code": "unk", "gender": "unk", "age_group": "66-75"
    }  # fmt: skip
    assert body["timezone"] == "America/New_York"


def test_the_async_client_sends_the_user_too():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=PAGE)

    async def run() -> None:
        async with AsyncUnboxAIClient(
            "US", api_key="test", base_url="http://sdk.test/v1"
        ) as c:
            c._http_client._transport = httpx.MockTransport(handler)
            await c.complete([View("a")], user=User(gender="male", market="NO"))

    asyncio.run(run())
    body = json.loads(requests[0].content)
    assert (body["domains"]["gender"], body["domains"]["market"]) == ("male", "NO")


@pytest.mark.parametrize(
    ("age", "group"),
    [(0, "<18"), (17.9, "<18"), (18, "18-25"), (25, "18-25"), (26, "26-35"),
     (65, "56-65"), (66, "66-75"), (85, "76-85"), (86, ">85"), (120, ">85")],
)  # fmt: skip
def test_ages_fall_in_the_groups_behaviorgpt_knows(age: float, group: str):
    assert age_group(age) == group
    assert User(age=age).age_group == group


@pytest.mark.parametrize(
    "fields",
    [{"gender": "other"}, {"age_group": "30-40"}, {"age": 200}, {"age": -1},
     {"market": "Sweden"}, {"timezone": "CEST"}, {"age": 40, "age_group": "18-25"}],
)  # fmt: skip
def test_values_behaviorgpt_does_not_know_are_refused(fields: dict):
    with pytest.raises(ValidationError):
        User(**fields)


def test_the_groups_are_the_apis():
    assert AGE_GROUPS == (
        "<18", "18-25", "26-35", "36-45", "46-55", "56-65", "66-75", "76-85", ">85"
    )  # fmt: skip


def test_a_user_id_reads_their_stored_events_without_a_history():
    c = client()
    requests = capture(c)
    c.complete(user_id="u-1842", user=User(gender="male"))
    body = json.loads(requests[0].content)
    assert requests[0].url.path.endswith("/recommendations")
    assert (body["user_id"], body["history"]) == ("u-1842", [])
    assert body["domains"]["gender"] == "male"

    c.complete([Search("sofa")], user_id="u-1842")
    body = json.loads(requests[1].content)
    assert requests[1].url.path.endswith("/search")
    assert (body["user_id"], body["query"]) == ("u-1842", "sofa")

    c.sections(user_id="u-1842")
    assert json.loads(requests[2].content)["user_id"] == "u-1842"


def test_without_a_user_id_none_is_sent():
    c = client()
    requests = capture(c)
    c.complete([View("a")])
    assert "user_id" not in json.loads(requests[0].content)
