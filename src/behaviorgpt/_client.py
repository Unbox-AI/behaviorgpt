import os
import random
import warnings
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence, cast
from uuid import uuid4

import httpx

from behaviorgpt._exceptions import AuthenticationError
from behaviorgpt.resources.catalogs import AsyncCatalogs, Catalogs
from behaviorgpt.resources.recommendations import AsyncRecommender, Recommender
from behaviorgpt.resources.requests import UnboxAIRequester
from behaviorgpt.resources.search import AsyncSearcher, Searcher
from behaviorgpt.resources.sections import AsyncSectioner, Sectioner, sections_body
from behaviorgpt.types import (
    AddItemsJobDetails,
    AddKeywordsJobDetails,
    AddToCart,
    CartItem,
    CatalogSummary,
    Domains,
    EmbedJobDetails,
    Item,
    JobStatus,
    Order,
    RemoveFromCart,
    Search,
    SectionsResponse,
    SessionEvent,
    UnboxAIRequest,
    UnboxAIResponse,
    UserEvent,
    UserHistoryInput,
)
from behaviorgpt.types.http import RecommendationType

# An embed upload streams the whole parquet, and the API answers only once the
# file is stored and queued: httpx's 5 s default is too short for both.
DEFAULT_HTTP_TIMEOUT = httpx.Timeout(120.0, connect=10.0, write=300.0)
DEFAULT_BASE_URL = "https://behaviorgpt-northeurope.api.unboxai.com/v1"
DEFAULT_TRAILING_SEARCHES = 3

Predict = Literal["view", "purchase"]


class _BaseClient:
    """What the sync and the async client share: settings and request building."""

    def __init__(
        self,
        market: str,
        *,
        timezone: str = "UTC",
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        default_catalog_id: str = "retail_catalog",
    ):
        self.api_key = api_key or os.environ.get("UNBOXAI_API_KEY")
        if not self.api_key:
            msg = "API key is missing. Pass it or set UNBOXAI_API_KEY."
            raise AuthenticationError(msg)

        self.base_url = base_url
        self.default_catalog_id = default_catalog_id

        self.domains = Domains(market=market)
        self.timezone = timezone

        self.requester = UnboxAIRequester(domains=self.domains, timezone=self.timezone)

    def _catalog_headers(self, catalog_id: Optional[str]) -> tuple[str, dict]:
        active_catalog: str = catalog_id or self.default_catalog_id
        return active_catalog, {"x-catalog-id": active_catalog}

    def _complete_request(
        self,
        history: UserHistoryInput,
        limit: int,
        offset: int,
        catalog_id: Optional[str],
        filters: Optional[dict[str, Any]],
        register_event: bool,
        max_trailing_searches: int = DEFAULT_TRAILING_SEARCHES,
        predict: Predict = "view",
    ) -> tuple[bool, UnboxAIRequest, dict]:
        """The request `complete` sends, whether it goes to search, and its headers."""
        if predict not in ("view", "purchase"):
            raise ValueError(f"predict must be 'view' or 'purchase', not {predict!r}")
        active_catalog, request_headers = self._catalog_headers(catalog_id)

        history_events = self.__from_sequence(active_catalog, history)
        last_event = self._get_last_event(history)

        if isinstance(last_event, Search):
            if predict == "purchase":
                raise ValueError(
                    "predict='purchase' needs a history that does not end in a "
                    "search: search ranks the next item viewed"
                )
            req = self.requester.make(
                query=last_event.query,
                history=history_events[:-1],
                catalog_id=active_catalog,
                limit=limit,
                offset=offset,
                filters=filters,
                register_event=register_event,
                max_trailing_searches=max_trailing_searches,
            )
            return True, req, request_headers

        req = self.requester.make(
            history=history_events,
            catalog_id=active_catalog,
            limit=limit,
            offset=offset,
            filters=filters,
            register_event=register_event,
            recommendation_type=(
                RecommendationType(name="buy_next") if predict == "purchase" else None
            ),
        )
        return False, req, request_headers

    def _sections_request(
        self,
        history: UserHistoryInput,
        catalog_id: Optional[str],
        filters: Optional[dict[str, Any]],
        register_event: bool,
        max_sections: int,
        max_items_per_section: int,
        min_items_per_section: int,
    ) -> tuple[dict, dict]:
        """The body `sections` sends, and its headers."""
        _, req, headers = self._complete_request(
            history, 10, 0, catalog_id, filters, register_event
        )
        body = sections_body(
            req, max_sections, max_items_per_section, min_items_per_section
        )
        return body, headers

    def __from_sequence(
        self, catalog_id: str, context: UserHistoryInput
    ) -> list[SessionEvent]:
        events: list[SessionEvent] = []

        if not context:
            return events

        # Track cart state within a single session
        def _process_session_events(
            session_events: Sequence[UserEvent], session_id: str, session_offset: int
        ):
            cart_state: dict[str, int] = defaultdict(int)
            current_cart_id = str(uuid4())

            # Base time for the start of the sequence
            base_time = datetime.now(tz=UTC)

            for i, ev in enumerate(session_events):
                if isinstance(ev, AddToCart):
                    cart_state[ev.item] += 1
                elif isinstance(ev, RemoveFromCart):
                    if cart_state[ev.item] > 1:
                        cart_state[ev.item] -= 1
                    else:
                        cart_state.pop(ev.item)
                elif isinstance(ev, Order):
                    if not getattr(ev, "items", None):
                        ev.cart_id = current_cart_id
                        ev.items = [
                            CartItem(item, qty) for item, qty in cart_state.items()
                        ]
                    cart_state.clear()
                    current_cart_id = str(uuid4())

                # Add 1 millisecond for each event to guarantee network-safe distinct
                # ordering. Factors in a session offset just in case 2D arrays
                # were passed
                event_time = base_time + timedelta(
                    milliseconds=(session_offset * 1000) + i
                )

                events.append(
                    self.__to_session_evt(ev, session_id, catalog_id, event_time)
                )

        if isinstance(context[0], (list, tuple)):
            context_2d = cast(Sequence[Sequence[UserEvent]], context)
            for session_index, session_sequence in enumerate(context_2d):
                _process_session_events(session_sequence, str(uuid4()), session_index)
        else:
            context_1d = cast(Sequence[UserEvent], context)
            _process_session_events(context_1d, str(uuid4()), 0)

        return events

    def __to_session_evt(
        self, event: UserEvent, session_id: str, catalog_id: str, timestamp: datetime
    ) -> SessionEvent:
        return SessionEvent(
            store_id=catalog_id,
            domains=self.domains,
            timezone=self.timezone,
            event=event,
            session_id=session_id,
            timestamp=timestamp,
        )

    def _get_last_event(self, history: UserHistoryInput) -> UserEvent | None:
        if not history:
            return None

        if isinstance(history[0], (list, tuple)):
            history_2d = cast(Sequence[Sequence[UserEvent]], history)
            inner_seq = history_2d[-1]
            return inner_seq[-1] if inner_seq else None
        else:
            history_1d = cast(Sequence[UserEvent], history)
            return history_1d[-1]


class UnboxAIClient(_BaseClient):
    def __init__(
        self,
        market: str,
        *,
        timezone: str = "UTC",
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        default_catalog_id: str = "retail_catalog",
        http_timeout: float | httpx.Timeout = DEFAULT_HTTP_TIMEOUT,
    ):
        super().__init__(
            market,
            timezone=timezone,
            api_key=api_key,
            base_url=base_url,
            default_catalog_id=default_catalog_id,
        )

        headers = {"x-api-key": self.api_key}
        self._http_client = httpx.Client(
            base_url=self.base_url, headers=headers, timeout=http_timeout
        )

        self.recommendations = Recommender(self._http_client)
        self.search = Searcher(self._http_client)
        self.catalogs = Catalogs(self._http_client)
        self._sections = Sectioner(self._http_client)

    def complete(
        self,
        history: UserHistoryInput,
        limit: int = 10,
        offset: int = 0,
        *,
        catalog_id: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
        register_event: bool = False,
        max_trailing_searches: int = DEFAULT_TRAILING_SEARCHES,
        predict: Predict = "view",
    ) -> UnboxAIResponse:
        """Rank the items most likely to come after `history`.

        `predict` picks the question: `"view"`, the next item this person looks
        at, or `"purchase"`, what else goes in the same purchase as the items
        in the current session ("bought together"). For what they want after
        buying something, keep `"view"` and end the history with
        `Order([...])`. A history that ends in a search can only use `"view"`.

        When the history ends in searches, up to `max_trailing_searches` of
        them in a row, the query included, reach the model, so `nike` then
        `shoes` reads as one intent.
        """
        is_search, req, headers = self._complete_request(
            history,
            limit,
            offset,
            catalog_id,
            filters,
            register_event,
            max_trailing_searches,
            predict,
        )
        endpoint = self.search if is_search else self.recommendations
        return endpoint.create(payload=req, headers=headers)

    def sections(
        self,
        history: UserHistoryInput,
        *,
        catalog_id: Optional[str] = None,
        max_sections: int = 10,
        max_items_per_section: int = 25,
        min_items_per_section: int = 5,
        filters: Optional[dict[str, Any]] = None,
        register_event: bool = False,
    ) -> SectionsResponse:
        """The items most likely to come after `history`, grouped into titled
        sections such as "black coats", best section first.

        A history that ends in a search groups that search's results; any
        other history groups recommendations. Sections are built from item
        keywords: `retail_catalog` has them, and a catalog of your own has
        them if it was embedded with `keywords=True`; without them the list
        is empty.

        `max_sections` can be at most 15 and `max_items_per_section` at most
        100. Each item's `data` holds its generated search terms in
        `top_terms`, best first.
        """
        body, headers = self._sections_request(
            history,
            catalog_id,
            filters,
            register_event,
            max_sections,
            max_items_per_section,
            min_items_per_section,
        )
        return self._sections.create(body, headers=headers)

    def embed(
        self,
        path: Path | str,
        wait: bool = False,
        interval: float = 5.0,
        timeout: float = 600.0,
        on_progress: Optional[Callable[[JobStatus], None]] = None,
        startup_grace: float = 120.0,
        reference_time: Optional[datetime | str] = None,
        keywords: bool = False,
    ) -> EmbedJobDetails:
        """Upload and embed a catalog parquet.

        Pass `wait=True` to block until the embed job finishes. While waiting,
        a progress bar shows each step (each new state is printed once where
        the output is not a terminal or a notebook); pass `on_progress` to
        render progress yourself, or `on_progress=lambda _: None` for silence.

        `reference_time` is when item ages and sales are measured; the upload
        time by default.

        `keywords=True` also generates the search keywords and browse
        categories `sections` groups items by. Allow a longer `timeout`: it
        adds time before the catalog is ready, about 2 hours per 20,000
        items. If it fails the catalog still goes live, without them. Off by
        default; `add_keywords` generates them for the catalog later.
        """
        return self.catalogs.embed(
            Path(path),
            wait=wait,
            interval=interval,
            timeout=timeout,
            on_progress=on_progress,
            startup_grace=startup_grace,
            reference_time=reference_time,
            keywords=keywords,
        )

    def add_items(
        self,
        path: Path | str,
        *,
        catalog_id: str,
        wait: bool = False,
        interval: float = 5.0,
        timeout: float = 600.0,
        on_progress: Optional[Callable[[JobStatus], None]] = None,
        startup_grace: float = 120.0,
    ) -> AddItemsJobDetails:
        """Add items to, or replace items in, an embedded catalog without
        re-embedding the rest of it.

        The parquet has the same format as for `embed`. An item whose `id` is
        already in the catalog is replaced whole; any other is added. The
        catalog keeps serving while the job runs. Items are embedded at the
        catalog's own reference time, and the catalog's item limit still
        applies.

        A later `embed` of the catalog's file replaces the whole catalog,
        these items included: put them in that file too to keep them.

        `wait` and the arguments after it work as for `embed`.
        """
        return self.catalogs.add_items(
            Path(path),
            catalog_id=catalog_id,
            wait=wait,
            interval=interval,
            timeout=timeout,
            on_progress=on_progress,
            startup_grace=startup_grace,
        )

    def add_keywords(
        self,
        catalog_id: str,
        *,
        wait: bool = False,
        interval: float = 5.0,
        timeout: float = 4 * 3600.0,
        on_progress: Optional[Callable[[JobStatus], None]] = None,
        startup_grace: float = 120.0,
    ) -> AddKeywordsJobDetails:
        """Generate the search keywords and browse categories `sections`
        groups items by, for a catalog embedded without them.

        Nothing is re-embedded, and the catalog keeps serving while the job
        runs; it takes about 2 hours per 20,000 items, so the default
        `timeout` of 4 hours covers about 35,000. Once the job is ready
        the catalog serves with keywords. If it fails, the catalog stays as it
        was. Uploading the catalog again with `embed` replaces the keywords;
        pass `keywords=True` there to keep them.

        `wait` and the arguments after it work as for `embed`.
        """
        return self.catalogs.add_keywords(
            catalog_id,
            wait=wait,
            interval=interval,
            timeout=timeout,
            on_progress=on_progress,
            startup_grace=startup_grace,
        )

    def job_status(self, job_id: str) -> JobStatus:
        return self.catalogs.get_job_status(job_id)

    def list_catalogs(self) -> list[CatalogSummary]:
        """The catalogs this key can query: its own, in any status, and the
        shared ones that are ready. `owned` marks the key's own."""
        return self.catalogs.list_catalogs()

    def delete_catalog(self, catalog_id: str) -> None:
        """Delete one of this key's catalogs, its items and its index, and
        free its slot. This cannot be undone.

        A catalog that is not this key's raises `UnboxAIError` with status
        404, the same as an unknown one. While an embed or `add_items` job runs
        for the catalog, it raises with status 409; delete once the job is
        ready or failed.
        """
        self.catalogs.delete_catalog(catalog_id)

    def similar_items(
        self,
        item_id: str,
        limit: int = 10,
        offset: int = 0,
        *,
        catalog_id: Optional[str] = None,
    ) -> UnboxAIResponse:
        active_catalog, request_headers = self._catalog_headers(catalog_id)
        return self.catalogs.get_similar_items(
            item_id,
            catalog_id=active_catalog,
            limit=limit,
            offset=offset,
            headers=request_headers,
        )

    def similar_products(
        self, product_id: str, *args: Any, **kwargs: Any
    ) -> UnboxAIResponse:
        warnings.warn(
            "`similar_products` is deprecated, use `similar_items`",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.similar_items(product_id, *args, **kwargs)

    def random_item(self, catalog_id: Optional[str] = None, window: int = 100) -> Item:
        """One item from the embedded catalog, handy when
        you have no item id at hand.
        """
        top = self.complete(history=[], catalog_id=catalog_id, limit=window)
        return _pick_random(top, catalog_id or self.default_catalog_id)

    def random_product(self, *args: Any, **kwargs: Any) -> Item:
        warnings.warn(
            "`random_product` is deprecated, use `random_item`",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.random_item(*args, **kwargs)

    def umap(self, *, catalog_id: Optional[str] = None):
        """The catalog's UMAP, a 2D map of its items, as an HTML page.

        Maps exist only for catalogs of up to 100,000 items, and appear within
        about a minute after the catalog is ready: laying out more items than
        that does not fit in the embedding worker's memory, so larger catalogs
        (such as the shared `retail_catalog`) never get one. A missing map
        raises `UnboxAIError` with status 404 and says which case applies.
        """
        active_catalog: str = catalog_id or self.default_catalog_id
        return self.catalogs.get_umap(catalog_id=active_catalog)


class AsyncUnboxAIClient(_BaseClient):
    """`UnboxAIClient` for asyncio code: the same methods, awaited.

    Use it as `async with AsyncUnboxAIClient(...) as client:`, or call
    `await client.aclose()` when done, to release its connections.
    """

    def __init__(
        self,
        market: str,
        *,
        timezone: str = "UTC",
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        default_catalog_id: str = "retail_catalog",
        http_timeout: float | httpx.Timeout = DEFAULT_HTTP_TIMEOUT,
    ):
        super().__init__(
            market,
            timezone=timezone,
            api_key=api_key,
            base_url=base_url,
            default_catalog_id=default_catalog_id,
        )

        headers = {"x-api-key": self.api_key}
        self._http_client = httpx.AsyncClient(
            base_url=self.base_url, headers=headers, timeout=http_timeout
        )

        self.recommendations = AsyncRecommender(self._http_client)
        self.search = AsyncSearcher(self._http_client)
        self.catalogs = AsyncCatalogs(self._http_client)
        self._sections = AsyncSectioner(self._http_client)

    async def __aenter__(self) -> "AsyncUnboxAIClient":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http_client.aclose()

    async def complete(
        self,
        history: UserHistoryInput,
        limit: int = 10,
        offset: int = 0,
        *,
        catalog_id: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
        register_event: bool = False,
        max_trailing_searches: int = DEFAULT_TRAILING_SEARCHES,
        predict: Predict = "view",
    ) -> UnboxAIResponse:
        """Rank the items most likely to come after `history`; see
        `UnboxAIClient.complete`."""
        is_search, req, headers = self._complete_request(
            history,
            limit,
            offset,
            catalog_id,
            filters,
            register_event,
            max_trailing_searches,
            predict,
        )
        endpoint = self.search if is_search else self.recommendations
        return await endpoint.create(payload=req, headers=headers)

    async def sections(
        self,
        history: UserHistoryInput,
        *,
        catalog_id: Optional[str] = None,
        max_sections: int = 10,
        max_items_per_section: int = 25,
        min_items_per_section: int = 5,
        filters: Optional[dict[str, Any]] = None,
        register_event: bool = False,
    ) -> SectionsResponse:
        """Items grouped into titled sections; see `UnboxAIClient.sections`."""
        body, headers = self._sections_request(
            history,
            catalog_id,
            filters,
            register_event,
            max_sections,
            max_items_per_section,
            min_items_per_section,
        )
        return await self._sections.create(body, headers=headers)

    async def embed(
        self,
        path: Path | str,
        wait: bool = False,
        interval: float = 5.0,
        timeout: float = 600.0,
        on_progress: Optional[Callable[[JobStatus], None]] = None,
        startup_grace: float = 120.0,
        reference_time: Optional[datetime | str] = None,
        keywords: bool = False,
    ) -> EmbedJobDetails:
        """Upload and embed a catalog parquet; see `UnboxAIClient.embed`.

        With `wait=True` it waits without blocking the event loop.
        """
        return await self.catalogs.embed(
            Path(path),
            wait=wait,
            interval=interval,
            timeout=timeout,
            on_progress=on_progress,
            startup_grace=startup_grace,
            reference_time=reference_time,
            keywords=keywords,
        )

    async def add_items(
        self,
        path: Path | str,
        *,
        catalog_id: str,
        wait: bool = False,
        interval: float = 5.0,
        timeout: float = 600.0,
        on_progress: Optional[Callable[[JobStatus], None]] = None,
        startup_grace: float = 120.0,
    ) -> AddItemsJobDetails:
        """Add or replace items in an embedded catalog; see
        `UnboxAIClient.add_items`."""
        return await self.catalogs.add_items(
            Path(path),
            catalog_id=catalog_id,
            wait=wait,
            interval=interval,
            timeout=timeout,
            on_progress=on_progress,
            startup_grace=startup_grace,
        )

    async def add_keywords(
        self,
        catalog_id: str,
        *,
        wait: bool = False,
        interval: float = 5.0,
        timeout: float = 4 * 3600.0,
        on_progress: Optional[Callable[[JobStatus], None]] = None,
        startup_grace: float = 120.0,
    ) -> AddKeywordsJobDetails:
        """Generate keywords for a catalog embedded without them; see
        `UnboxAIClient.add_keywords`."""
        return await self.catalogs.add_keywords(
            catalog_id,
            wait=wait,
            interval=interval,
            timeout=timeout,
            on_progress=on_progress,
            startup_grace=startup_grace,
        )

    async def job_status(self, job_id: str) -> JobStatus:
        return await self.catalogs.get_job_status(job_id)

    async def list_catalogs(self) -> list[CatalogSummary]:
        """See `UnboxAIClient.list_catalogs`."""
        return await self.catalogs.list_catalogs()

    async def delete_catalog(self, catalog_id: str) -> None:
        """See `UnboxAIClient.delete_catalog`."""
        await self.catalogs.delete_catalog(catalog_id)

    async def similar_items(
        self,
        item_id: str,
        limit: int = 10,
        offset: int = 0,
        *,
        catalog_id: Optional[str] = None,
    ) -> UnboxAIResponse:
        active_catalog, request_headers = self._catalog_headers(catalog_id)
        return await self.catalogs.get_similar_items(
            item_id,
            catalog_id=active_catalog,
            limit=limit,
            offset=offset,
            headers=request_headers,
        )

    async def random_item(
        self, catalog_id: Optional[str] = None, window: int = 100
    ) -> Item:
        """One item from the embedded catalog; see `UnboxAIClient.random_item`."""
        top = await self.complete(history=[], catalog_id=catalog_id, limit=window)
        return _pick_random(top, catalog_id or self.default_catalog_id)

    async def umap(self, *, catalog_id: Optional[str] = None) -> str:
        """The catalog's UMAP as an HTML page; see `UnboxAIClient.umap`."""
        active_catalog: str = catalog_id or self.default_catalog_id
        return await self.catalogs.get_umap(catalog_id=active_catalog)


def _pick_random(top: UnboxAIResponse, catalog_id: str) -> Item:
    if not top.items:
        raise ValueError(f"catalog {catalog_id!r} returned no items")
    return random.choice(top.items)
