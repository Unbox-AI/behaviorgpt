import warnings
from typing import Any

from behaviorgpt.types import http
from behaviorgpt.types.domains import Domains
from behaviorgpt.types.events import (
    AddToCart,
    CartItem,
    Order,
    RemoveFromCart,
    Search,
    SessionEvent,
    UserEvent,
    UserHistoryInput,
    View,
)
from behaviorgpt.types.http import (
    EmbedJobDetails,
    Item,
    ItemsPage,
    ItemsRejected,
    JobStatus,
    RejectedItem,
    SimilarItemsRequest,
    UnboxAIRequest,
    UnboxAIResponse,
)

__all__ = [
    "UserEvent",
    "SessionEvent",
    "Search",
    "AddToCart",
    "View",
    "RemoveFromCart",
    "Order",
    "UnboxAIResponse",
    "UnboxAIRequest",
    "Domains",
    "UserHistoryInput",
    "SimilarItemsRequest",
    "EmbedJobDetails",
    "JobStatus",
    "ItemsRejected",
    "RejectedItem",
    "Item",
    "ItemsPage",
    "CartItem",
]


def __getattr__(name: str) -> Any:
    if name in http.DEPRECATED_NAMES:
        new = http.DEPRECATED_NAMES[name]
        warnings.warn(
            f"`{name}` is deprecated, use `{new}`", DeprecationWarning, stacklevel=2
        )
        return getattr(http, new)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
