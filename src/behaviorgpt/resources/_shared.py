import httpx

from behaviorgpt._exceptions import (
    AuthenticationError,
    BadRequestError,
    RateLimitError,
    UnboxAIError,
)


def handle_response(response: httpx.Response):
    if response.status_code < 400:
        if response.status_code == 204:  # No Content
            return None
        return response.json()

    try:
        error_data = response.json().get("detail", response.text)
    except ValueError:
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
