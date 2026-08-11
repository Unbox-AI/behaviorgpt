class UnboxAIError(Exception):
    def __init__(
        self, message: str, status_code: int | None = None, response: dict | None = None
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response = response or {}


class AuthenticationError(UnboxAIError):
    """Raised when the API key is missing, invalid, or revoked (HTTP 401)."""

    pass


class RateLimitError(UnboxAIError):
    """Raised when the user has sent too many requests (HTTP 429)."""

    pass


class BadRequestError(UnboxAIError):
    """Raised when the request is invalid or missing required parameters (HTTP 400)."""

    pass
