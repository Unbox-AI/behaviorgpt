from datetime import UTC, datetime
from typing import List, Optional, Protocol, Sequence
from uuid import uuid4

import httpx

from behaviorgpt.resources._shared import handle_response
from behaviorgpt.types import Domains, Order, UserEvent
from behaviorgpt.types.http import RecordedCall, TrackedEvent
from behaviorgpt.types.user import User

PATH = "/events"
OWN_PATH = "/events/own"
CALLS_PATH = "/events/own/calls"
# the server's limit
MAX_LISTED = 200


class EventContext(Protocol):
    domains: Domains
    timezone: str
    default_catalog_id: str


def _event_catalog(event: UserEvent) -> Optional[str]:
    """The catalog the event's items say they are in, if they say."""
    if isinstance(event, Order):
        named = {entry.catalog_id for entry in event.items or ()} - {None}
        if len(named) > 1:
            raise ValueError(
                "an Order's items are in one catalog; track one Order per catalog"
            )
        return next(iter(named), None)
    return getattr(event, "catalog_id", None)


def event_body(
    context: EventContext,
    user_id: str,
    event: UserEvent,
    *,
    catalog_id: Optional[str],
    timestamp: Optional[datetime],
    user: Optional[User],
) -> dict:
    """The body `track` posts for `event`."""
    if timestamp is None:
        timestamp = datetime.now(tz=UTC)
    elif timestamp.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")

    payload = event.model_dump(mode="json", exclude_none=True)
    if isinstance(event, Order):
        if not event.items:
            raise ValueError("track an Order with the items bought: Order([item, ...])")
        # The API stores orders with a cart id and a unit price per item; the
        # model reads only which items were bought.
        payload["cart_id"] = event.cart_id or str(uuid4())
        payload["items"] = [
            {
                "product": entry.item,
                "quantity": entry.quantity,
                "unit_selling_price": 0.0,
            }
            for entry in event.items
        ]

    domains = user.domains(context.domains.market) if user else context.domains
    return {
        "user_id": user_id,
        "store_id": catalog_id or _event_catalog(event) or context.default_catalog_id,
        "event": payload,
        "domains": domains.model_dump(mode="json", exclude_none=True),
        "timezone": (user.timezone if user else None) or context.timezone,
        "timestamp": timestamp.isoformat(),
    }


def list_params(
    user_id: Optional[str],
    limit: int,
    types: Optional[Sequence[str]],
    stream: Optional[str],
) -> list[tuple[str, str]]:
    if not 1 <= limit <= MAX_LISTED:
        raise ValueError(f"limit must be between 1 and {MAX_LISTED}, got {limit}")
    params: list[tuple[str, str]] = [("limit", str(limit))]
    if user_id is not None:
        params.append(("user_id", user_id))
    params.extend(("type", t) for t in types or ())
    if stream is not None:
        params.append(("stream", stream))
    return params


def tracked_events(data: dict) -> list[TrackedEvent]:
    return [TrackedEvent(**entry) for entry in data["events"]]


def calls_params(
    user_id: Optional[str], limit: int, calls: Optional[Sequence[str]]
) -> list[tuple[str, str]]:
    if not 1 <= limit <= MAX_LISTED:
        raise ValueError(f"limit must be between 1 and {MAX_LISTED}, got {limit}")
    params: list[tuple[str, str]] = [("limit", str(limit))]
    if user_id is not None:
        params.append(("user_id", user_id))
    params.extend(("call", c) for c in calls or ())
    return params


def recorded_calls(data: dict) -> list[RecordedCall]:
    return [RecordedCall(**entry) for entry in data["calls"]]


class Events:
    """Your users' events. Record them with `track`; `complete(user_id=...)`
    and `sections(user_id=...)` then read them as that user's history, and
    record what they ran for them: a search as their event, the rest in
    `calls`."""

    def __init__(self, http_client: httpx.Client, context: EventContext):
        self._client = http_client
        self._context = context

    def track(
        self,
        user_id: str,
        event: UserEvent,
        *,
        catalog_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        user: Optional[User] = None,
    ) -> str:
        """Record that `user_id` did `event`, and return the event's id.

        `timestamp` is when it happened, now by default. Pass it when you send
        events later than they happen, so that they stay in order. The catalog
        is `catalog_id`, else the one an `Item` of a response names, else the
        client's default. `user` says who they are, as in `complete`.
        """
        body = event_body(
            self._context,
            user_id,
            event,
            catalog_id=catalog_id,
            timestamp=timestamp,
            user=user,
        )
        return handle_response(self._client.post(PATH, json=body))["id"]

    def list(
        self,
        *,
        user_id: Optional[str] = None,
        limit: int = 50,
        types: Optional[Sequence[str]] = None,
        stream: Optional[str] = None,
    ) -> list[TrackedEvent]:
        """Your account's latest events, newest first: all your users', or
        `user_id`'s. `types` keeps only those event types, such as
        `["viewItem", "search"]`, and `stream` only the events of one stream:
        a publishable key, "api" for those sent with your key, or an import
        such as "import_shop". `limit` can be at most 200."""
        params = list_params(user_id, limit, types, stream)
        response = self._client.get(OWN_PATH, params=params)
        return tracked_events(handle_response(response))

    def calls(
        self,
        *,
        user_id: Optional[str] = None,
        limit: int = 50,
        calls: Optional[Sequence[str]] = None,
    ) -> List[RecordedCall]:
        """The calls made for your users, newest first: all of them, or
        `user_id`'s. Each `complete`, `sections` and `autocomplete` with a
        `user_id` is recorded with what it was for and the items it showed,
        kept 28 days; histories never read them. `calls` keeps only those
        calls, such as `["search", "autocomplete"]`. `limit` can be at most
        200."""
        params = calls_params(user_id, limit, calls)
        response = self._client.get(CALLS_PATH, params=params)
        return recorded_calls(handle_response(response))


class AsyncEvents:
    """`Events` for `AsyncUnboxAIClient`."""

    def __init__(self, http_client: httpx.AsyncClient, context: EventContext):
        self._client = http_client
        self._context = context

    async def track(
        self,
        user_id: str,
        event: UserEvent,
        *,
        catalog_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        user: Optional[User] = None,
    ) -> str:
        """See `Events.track`."""
        body = event_body(
            self._context,
            user_id,
            event,
            catalog_id=catalog_id,
            timestamp=timestamp,
            user=user,
        )
        return handle_response(await self._client.post(PATH, json=body))["id"]

    async def list(
        self,
        *,
        user_id: Optional[str] = None,
        limit: int = 50,
        types: Optional[Sequence[str]] = None,
        stream: Optional[str] = None,
    ) -> list[TrackedEvent]:
        """See `Events.list`."""
        params = list_params(user_id, limit, types, stream)
        response = await self._client.get(OWN_PATH, params=params)
        return tracked_events(handle_response(response))

    async def calls(
        self,
        *,
        user_id: Optional[str] = None,
        limit: int = 50,
        calls: Optional[Sequence[str]] = None,
    ) -> List[RecordedCall]:
        """See `Events.calls`."""
        params = calls_params(user_id, limit, calls)
        response = await self._client.get(CALLS_PATH, params=params)
        return recorded_calls(handle_response(response))
