from typing import Union

import httpx

from behaviorgpt._exceptions import (
    AuthenticationError,
    BadRequestError,
    RateLimitError,
    UnboxAIError,
)
from behaviorgpt.types import UnboxAIRequest, UnboxAIResponse


def as_request(payload: Union[UnboxAIRequest, dict]) -> UnboxAIRequest:
    return UnboxAIRequest(**payload) if isinstance(payload, dict) else payload


def request_body(payload: UnboxAIRequest) -> dict:
    return payload.model_dump(mode="json", exclude_none=True)


def ranked_response(payload: UnboxAIRequest, data: dict) -> UnboxAIResponse:
    market_val = getattr(payload.domains, "market", None)
    return UnboxAIResponse(**data, market=market_val)


def describe_detail(detail):
    """The API's error detail as text.

    A validation error lists one entry per failing field, each echoing the
    whole request; this keeps only `field: message`. When the server tries
    several request shapes, as for sections, an entry without a field only
    says which shape it ruled out, so those are dropped when another entry
    names the field at fault.
    """
    if not (
        isinstance(detail, list)
        and detail
        and all(isinstance(e, dict) and "msg" in e for e in detail)
    ):
        return detail

    def field(entry: dict) -> str:
        parts = [str(p) for p in entry.get("loc", [])[1:]]
        return ".".join(p for p in parts if "[" not in p)

    def from_shape(entry: dict) -> bool:
        return any("[" in str(p) for p in entry.get("loc", []))

    named = any(field(e) for e in detail)
    lines = []
    for entry in detail:
        if named and not field(entry) and from_shape(entry):
            continue
        msg = str(entry["msg"]).removeprefix("Value error, ")
        line = f"{field(entry)}: {msg}" if field(entry) else msg
        if line not in lines:
            lines.append(line)
    return "; ".join(lines)


def handle_response(response: httpx.Response):
    if response.status_code < 400:
        if response.status_code == 204:  # No Content
            return None
        return response.json()

    # the API answers {"detail": ...}; a proxy in front of it may answer plain
    # text or JSON of any shape
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict) and "detail" in body:
        error_data = describe_detail(body["detail"])
    else:
        error_data = response.text or "Unknown Error"

    if response.status_code == 401:
        raise AuthenticationError(
            f"Authentication failed: {error_data}", status_code=401
        )
    elif response.status_code == 429:
        raise RateLimitError(f"Rate limit exceeded: {error_data}", status_code=429)
    elif response.status_code == 400:
        raise BadRequestError(f"Bad request: {error_data}", status_code=400)

    raise UnboxAIError(f"API Error: {error_data}", status_code=response.status_code)
