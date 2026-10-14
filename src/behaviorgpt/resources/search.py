from typing import Union

import httpx

from behaviorgpt.resources._shared import (
    as_request,
    handle_response,
    ranked_response,
    request_body,
)
from behaviorgpt.types import UnboxAIRequest, UnboxAIResponse

PATH = "/vector/search"


class Searcher:
    def __init__(self, http_client: httpx.Client):
        self._client = http_client

    def create(
        self, payload: Union[UnboxAIRequest, dict], *, headers: dict | None = None
    ) -> UnboxAIResponse:
        payload = as_request(payload)
        response = self._client.post(PATH, json=request_body(payload), headers=headers)
        return ranked_response(payload, handle_response(response))


class AsyncSearcher:
    def __init__(self, http_client: httpx.AsyncClient):
        self._client = http_client

    async def create(
        self, payload: Union[UnboxAIRequest, dict], *, headers: dict | None = None
    ) -> UnboxAIResponse:
        payload = as_request(payload)
        response = await self._client.post(
            PATH, json=request_body(payload), headers=headers
        )
        return ranked_response(payload, handle_response(response))
