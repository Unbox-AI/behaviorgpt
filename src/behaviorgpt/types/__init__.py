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
    AddItemsJobDetails,
    AddKeywordsJobDetails,
    AutocompleteResponse,
    CatalogMap,
    CatalogSummary,
    EmbedJobDetails,
    Item,
    ItemsPage,
    ItemsRejected,
    JobStatus,
    RejectedItem,
    Section,
    SectionsResponse,
    SimilarItemsRequest,
    Suggestion,
    TrackedEvent,
    UnboxAIRequest,
    UnboxAIResponse,
)
from behaviorgpt.types.user import AGE_GROUPS, AgeGroup, User

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
    "User",
    "AgeGroup",
    "AGE_GROUPS",
    "UserHistoryInput",
    "SimilarItemsRequest",
    "EmbedJobDetails",
    "AddItemsJobDetails",
    "AddKeywordsJobDetails",
    "CatalogMap",
    "CatalogSummary",
    "JobStatus",
    "ItemsRejected",
    "RejectedItem",
    "Item",
    "ItemsPage",
    "CartItem",
    "Section",
    "SectionsResponse",
    "Suggestion",
    "AutocompleteResponse",
    "TrackedEvent",
]


def __getattr__(name: str) -> Any:
    if name in http.DEPRECATED_NAMES:
        new = http.DEPRECATED_NAMES[name]
        warnings.warn(
            f"`{name}` is deprecated, use `{new}`", DeprecationWarning, stacklevel=2
        )
        return getattr(http, new)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
