from typing import Union

import httpx

from behaviorgpt.resources._shared import handle_response
from behaviorgpt.types import UnboxAIRequest, UnboxAIResponse


class Searcher:
    def __init__(self, http_client: httpx.Client):
        self._client = http_client

    def create(
        self, payload: Union[UnboxAIRequest, dict], *, headers: dict | None = None
    ) -> UnboxAIResponse:
        if isinstance(payload, dict):
            payload = UnboxAIRequest(**payload)

        response = self._client.post(
            "/vector/search",
            json=payload.model_dump(mode="json", exclude_none=True),
            headers=headers,
        )

        data = handle_response(response)
        market_val = getattr(payload.domains, "market", None)

        return UnboxAIResponse(**data, market=market_val)
