import httpx

from behaviorgpt.resources._shared import handle_response, request_body
from behaviorgpt.types import SectionsResponse, UnboxAIRequest

PATH = "/vector/sections"


def sections_body(
    payload: UnboxAIRequest,
    max_sections: int,
    max_products_per_section: int,
    min_products_per_section: int,
) -> dict:
    """The ranking request with the grouping options. Without a query the
    server groups recommendations, with one it groups search results."""
    return request_body(payload) | {
        "max_sections": max_sections,
        "max_products_per_section": max_products_per_section,
        "min_products_per_section": min_products_per_section,
    }


class Sectioner:
    def __init__(self, http_client: httpx.Client):
        self._client = http_client

    def create(self, body: dict, *, headers: dict | None = None) -> SectionsResponse:
        response = self._client.post(PATH, json=body, headers=headers)
        return SectionsResponse(**handle_response(response))


class AsyncSectioner:
    def __init__(self, http_client: httpx.AsyncClient):
        self._client = http_client

    async def create(
        self, body: dict, *, headers: dict | None = None
    ) -> SectionsResponse:
        response = await self._client.post(PATH, json=body, headers=headers)
        return SectionsResponse(**handle_response(response))
