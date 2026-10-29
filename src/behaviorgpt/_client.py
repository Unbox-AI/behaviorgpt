import asyncio
import os
import random
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Optional, Sequence, cast
from uuid import uuid4

import httpx

from behaviorgpt._autocomplete import suggest
from behaviorgpt._exceptions import AuthenticationError, UnboxAIError
from behaviorgpt.resources.catalogs import AsyncCatalogs, Catalogs
from behaviorgpt.resources.events import AsyncEvents, Events
from behaviorgpt.resources.recommendations import AsyncRecommender, Recommender
from behaviorgpt.resources.requests import UnboxAIRequester
from behaviorgpt.resources.search import AsyncSearcher, Searcher
from behaviorgpt.resources.sections import AsyncSectioner, Sectioner, sections_body
from behaviorgpt.resources.shop import (
    DEFAULT_PORTAL_URL,
    AsyncShop,
    Shop,
    shop_http_options,
)
from behaviorgpt.types import (
    AddItemsJobDetails,
    AddKeywordsJobDetails,
    AddToCart,
    AutocompleteResponse,
    CartItem,
    CatalogMap,
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
    View,
)
from behaviorgpt.types.events import ItemRef
from behaviorgpt.types.http import RecommendationType
from behaviorgpt.types.user import User

# An embed upload streams the whole parquet, and the API answers only once the
# file is stored and queued: httpx's 5 s default is too short for both.
DEFAULT_HTTP_TIMEOUT = httpx.Timeout(120.0, connect=10.0, write=300.0)
DEFAULT_BASE_URL = "https://behaviorgpt-northeurope.api.unboxai.com/v1"
DEFAULT_TRAILING_SEARCHES = 3
# autocomplete reads this many ranked items for suggestions; at most the
# request's limit, 100
DEFAULT_SUGGESTION_CANDIDATES = 50
MAX_SUGGESTIONS = 20
SUGGESTION_FIELDS = ["name", "top_terms"]
# what an autocomplete call is recorded as for its user
AUTOCOMPLETE_CALL = "autocomplete"
UMAP_POLL = 5.0

Predict = Literal["view", "purchase"]
CATALOG_IDS_HEADER = "x-catalog-ids"


@dataclass(frozen=True)
class _Target:
    """The catalogs one request reads. Several (`catalog_ids=`) are ranked
    together, and then every item names the catalog it is in."""

    catalog_ids: tuple[str, ...]
    multi: bool = False

    @property
    def primary(self) -> str:
        return self.catalog_ids[0]

    @property
    def headers(self) -> dict:
        if self.multi:
            return {CATALOG_IDS_HEADER: ",".join(self.catalog_ids)}
        return {"x-catalog-id": self.primary}

    def of(self, catalog_id: Optional[str]) -> str:
        """The catalog of an item that names `catalog_id`, or none."""
        if catalog_id is None or catalog_id == self.primary:
            return self.primary
        if catalog_id not in self.catalog_ids:
            hint = "" if self.multi else "; pass catalog_ids= to rank several"
            raise ValueError(
                f"an item is in catalog {catalog_id!r}, but this request reads "
                f"{', '.join(map(repr, self.catalog_ids))}{hint}"
            )
        return catalog_id

    def label(self, items: Iterable[Item]) -> None:
        """Give each item of a response its catalog: the API names it only when
        the request reads several."""
        for item in items:
            if item.catalog_id is None:
                item.catalog_id = self.primary


class _BaseClient:
    """What the sync and the async client share: settings and request building."""

    def __init__(
        self,
        market: str,
        *,
        timezone: str = "UTC",
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        default_catalog_id: Optional[str] = None,
        portal_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("UNBOXAI_API_KEY")
        if not self.api_key:
            msg = "API key is missing. Pass it or set UNBOXAI_API_KEY."
            raise AuthenticationError(msg)

        self.base_url = base_url
        # where `shop` calls: the portal, whose shopping calls the widgets make too
        self.portal_url = (
            portal_url or os.environ.get("UNBOXAI_PORTAL_URL") or DEFAULT_PORTAL_URL
        )
        # the catalog of calls that name none
        self.default_catalog_id = (
            default_catalog_id
            or os.environ.get("UNBOXAI_CATALOG_ID")
            or "retail_catalog"
        )

        self.domains = Domains(market=market)
        self.timezone = timezone

        self.requester = UnboxAIRequester(domains=self.domains, timezone=self.timezone)

    def _target(
        self, catalog_id: Optional[str], catalog_ids: Optional[Sequence[str]]
    ) -> _Target:
        if catalog_ids is None:
            return _Target((catalog_id or self.default_catalog_id,))
        if catalog_id is not None:
            raise ValueError("pass catalog_id or catalog_ids, not both")
        if isinstance(catalog_ids, str):
            raise TypeError("catalog_ids takes a list of catalog ids, not a string")
        ids = tuple(dict.fromkeys(catalog_ids))
        if not ids:
            raise ValueError("catalog_ids names no catalog")
        return _Target(ids, multi=True)

    def _similar_request(
        self,
        item: ItemRef,
        catalog_id: Optional[str],
        catalog_ids: Optional[Sequence[str]],
    ) -> tuple[_Target, str, str]:
        """The catalogs a similar-items request reads, and the id and catalog
        of its item. The item's catalog is `catalog_id`, else the `Item`'s own;
        over `catalog_ids` it must be one of them, by default the first."""
        item_id = item if isinstance(item, str) else item.id
        item_catalog = catalog_id or (
            None if isinstance(item, str) else item.catalog_id
        )
        if catalog_ids is None:
            target = self._target(item_catalog, None)
            return target, item_id, target.primary
        target = self._target(None, catalog_ids)
        return target, item_id, target.of(item_catalog)

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
        catalog_ids: Optional[Sequence[str]] = None,
        user: Optional[User] = None,
        user_id: Optional[str] = None,
        record_call: Optional[bool] = None,
        call: Optional[str] = None,
    ) -> tuple[bool, UnboxAIRequest, _Target]:
        """The request `complete` sends, whether it goes to search, and the
        catalogs it reads."""
        if predict not in ("view", "purchase"):
            raise ValueError(f"predict must be 'view' or 'purchase', not {predict!r}")
        target = self._target(catalog_id, catalog_ids)
        # who: the request and every event of its history say the same
        domains = user.domains(self.domains.market) if user else self.domains
        timezone = (user.timezone if user else None) or self.timezone

        history_events = self.__from_sequence(target, history, domains, timezone)
        last_event = self._get_last_event(history)
        is_search = isinstance(last_event, Search)
        if user_id is not None and history_events[: -1 if is_search else None]:
            raise ValueError(
                "with user_id, BehaviorGPT reads the history it has stored for "
                "them: pass no history, or only the Search(...) to run for them"
            )

        if isinstance(last_event, Search):
            if predict == "purchase":
                raise ValueError(
                    "predict='purchase' needs a history that does not end in a "
                    "search: search ranks the next item viewed"
                )
            req = self.requester.make(
                query=last_event.query,
                history=history_events[:-1],
                catalog_id=target.primary,
                limit=limit,
                offset=offset,
                filters=filters,
                register_event=register_event,
                max_trailing_searches=max_trailing_searches,
                domains=domains,
                timezone=timezone,
                user_id=user_id,
                record_call=record_call,
                call=call,
            )
            return True, req, target

        req = self.requester.make(
            history=history_events,
            catalog_id=target.primary,
            limit=limit,
            offset=offset,
            filters=filters,
            register_event=register_event,
            recommendation_type=(
                RecommendationType(name="buy_next") if predict == "purchase" else None
            ),
            domains=domains,
            timezone=timezone,
            user_id=user_id,
            record_call=record_call,
            call=call,
        )
        return False, req, target

    def _autocomplete_request(
        self,
        text: str,
        history: UserHistoryInput,
        limit: int,
        candidates: int,
        catalog_id: Optional[str],
        catalog_ids: Optional[Sequence[str]],
        filters: Optional[dict[str, Any]],
        user: Optional[User],
        user_id: Optional[str],
        max_trailing_searches: int = DEFAULT_TRAILING_SEARCHES,
    ) -> tuple[bool, UnboxAIRequest, _Target]:
        """The ranking `autocomplete` reads: the history with the text typed so
        far as its last search, never recorded as a search. Empty text ranks
        what comes after the history. For a `user_id` the call is recorded as
        an "autocomplete" call, once: not the second ranking of the text alone
        (`max_trailing_searches=1`)."""
        if not 1 <= limit <= MAX_SUGGESTIONS:
            raise ValueError(
                f"limit must be between 1 and {MAX_SUGGESTIONS}, got {limit}"
            )
        if not limit <= candidates <= 100:
            raise ValueError(
                f"candidates must be between limit ({limit}) and 100, got {candidates}"
            )
        if text.strip():
            history = _with_search(history, text.strip())
        is_search, req, target = self._complete_request(
            history,
            candidates,
            0,
            catalog_id,
            filters,
            False,
            max_trailing_searches,
            catalog_ids=catalog_ids,
            user=user,
            user_id=user_id,
            record_call=_record_call(user_id, max_trailing_searches != 1),
            call=AUTOCOMPLETE_CALL,
        )
        req.fields_to_include = SUGGESTION_FIELDS
        return is_search, req, target

    def _sections_request(
        self,
        history: UserHistoryInput,
        catalog_id: Optional[str],
        filters: Optional[dict[str, Any]],
        register_event: bool,
        max_sections: int,
        max_items_per_section: int,
        min_items_per_section: int,
        catalog_ids: Optional[Sequence[str]] = None,
        user: Optional[User] = None,
        user_id: Optional[str] = None,
    ) -> tuple[dict, _Target]:
        """The body `sections` sends, and the catalogs it reads."""
        _, req, target = self._complete_request(
            history,
            10,
            0,
            catalog_id,
            filters,
            register_event,
            catalog_ids=catalog_ids,
            user=user,
            user_id=user_id,
            record_call=_record_call(user_id, register_event),
        )
        body = sections_body(
            req, max_sections, max_items_per_section, min_items_per_section
        )
        return body, target

    def __from_sequence(
        self,
        target: _Target,
        context: UserHistoryInput,
        domains: Domains,
        timezone: str,
    ) -> list[SessionEvent]:
        events: list[SessionEvent] = []

        if not context:
            return events

        sessions: Sequence[Sequence[UserEvent]] = (
            cast(Sequence[Sequence[UserEvent]], context)
            if isinstance(context[0], (list, tuple))
            else [cast(Sequence[UserEvent], context)]
        )
        # Events are 1 ms apart, sessions `gap_ms` apart, and the whole history ends
        # at least a second before now: the API reads only the events before the
        # request, by its own clock, so a history stamped from now on would lose
        # its later sessions and the tail of a long one.
        gap_ms = max(len(s) for s in sessions) + 1000
        start = datetime.now(tz=UTC) - timedelta(milliseconds=gap_ms * len(sessions))

        # Track cart state within a single session
        def _process_session_events(
            session_events: Sequence[UserEvent], session_id: str, session_offset: int
        ):
            cart_state: dict[tuple[Optional[str], str], int] = defaultdict(int)
            current_cart_id = str(uuid4())

            for i, ev in enumerate(session_events):
                if isinstance(ev, AddToCart):
                    cart_state[(ev.catalog_id, ev.item)] += 1
                elif isinstance(ev, RemoveFromCart):
                    key = (ev.catalog_id, ev.item)
                    if cart_state[key] > 1:
                        cart_state[key] -= 1
                    else:
                        cart_state.pop(key)
                elif isinstance(ev, Order):
                    if not getattr(ev, "items", None):
                        ev.cart_id = current_cart_id
                        ev.items = [
                            CartItem(item, qty, catalog_id=catalog)
                            for (catalog, item), qty in cart_state.items()
                        ]
                    cart_state.clear()
                    current_cart_id = str(uuid4())

                event_time = start + timedelta(milliseconds=session_offset * gap_ms + i)

                for j, (catalog, event) in enumerate(_by_catalog(ev, target)):
                    events.append(
                        self.__to_session_evt(
                            event,
                            session_id,
                            catalog,
                            event_time + timedelta(microseconds=j),
                            domains,
                            timezone,
                        )
                    )

        for session_index, session_sequence in enumerate(sessions):
            _process_session_events(session_sequence, str(uuid4()), session_index)

        return events

    def __to_session_evt(
        self,
        event: UserEvent,
        session_id: str,
        catalog_id: str,
        timestamp: datetime,
        domains: Domains,
        timezone: str,
    ) -> SessionEvent:
        return SessionEvent(
            store_id=catalog_id,
            domains=domains,
            timezone=timezone,
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
        default_catalog_id: Optional[str] = None,
        http_timeout: float | httpx.Timeout = DEFAULT_HTTP_TIMEOUT,
        portal_url: Optional[str] = None,
    ):
        super().__init__(
            market,
            timezone=timezone,
            api_key=api_key,
            base_url=base_url,
            default_catalog_id=default_catalog_id,
            portal_url=portal_url,
        )

        headers = {"x-api-key": self.api_key}
        self._http_client = httpx.Client(
            base_url=self.base_url, headers=headers, timeout=http_timeout
        )

        self.recommendations = Recommender(self._http_client)
        self.search = Searcher(self._http_client)
        self.catalogs = Catalogs(self._http_client)
        self.events = Events(self._http_client, self)
        self._sections = Sectioner(self._http_client)
        assert self.api_key is not None
        self._shop_client = httpx.Client(
            **shop_http_options(self.api_key, self.portal_url)
        )
        # search, shelves and the rest as the portal's widgets show them
        self.shop = Shop(self._shop_client, self)

    def complete(
        self,
        history: UserHistoryInput = (),
        limit: int = 10,
        offset: int = 0,
        *,
        catalog_id: Optional[str] = None,
        catalog_ids: Optional[Sequence[str]] = None,
        filters: Optional[dict[str, Any]] = None,
        register_event: Optional[bool] = None,
        max_trailing_searches: int = DEFAULT_TRAILING_SEARCHES,
        predict: Predict = "view",
        user: Optional[User] = None,
        user_id: Optional[str] = None,
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

        `catalog_ids` ranks several catalogs together instead of one, for
        instance your own and `retail_catalog`: each item of the result says
        which it is in (`item.catalog_id`). Items in the history are in the
        first catalog unless they name another, `View(item_id,
        catalog_id=...)`, or are `Item`s of an earlier response, which know
        theirs. In `filters`, `catalog_id` is the item's catalog, e.g.
        `{"catalog_id": "retail_catalog"}`.

        `user` says who the shopper is, `User(gender="female", age=34)`, and
        where, `User(market="SE", timezone="Europe/Stockholm")`: BehaviorGPT
        reads it with the history and the prediction. Left out, the gender and
        age are unknown and the market and time zone are the client's.

        `user_id` reads the events the API has stored for that person, sent by
        your site's script (the cookie `unbox_uid`) or with `events.track`, as
        the history: pass no `history` with it, or only the `Search(...)` to
        run for them.

        A call for a `user_id` is recorded for them. A search becomes their own
        event, so later calls read it; only the first page's, and only once:
        not again when your site's script sends the same search. And what the
        call showed is kept apart, listed by `events.calls`, never read as
        history. `register_event=False` records neither.
        """
        register = _registers(register_event, user_id)
        is_search, req, target = self._complete_request(
            history,
            limit,
            offset,
            catalog_id,
            filters,
            register,
            max_trailing_searches,
            predict,
            catalog_ids,
            user,
            user_id,
            record_call=_record_call(user_id, register),
        )
        endpoint = self.search if is_search else self.recommendations
        response = endpoint.create(payload=req, headers=target.headers)
        target.label(response.items)
        return response

    def autocomplete(
        self,
        text: str,
        history: UserHistoryInput = (),
        limit: int = 5,
        *,
        catalog_id: Optional[str] = None,
        catalog_ids: Optional[Sequence[str]] = None,
        filters: Optional[dict[str, Any]] = None,
        user: Optional[User] = None,
        user_id: Optional[str] = None,
        candidates: int = DEFAULT_SUGGESTION_CANDIDATES,
    ) -> AutocompleteResponse:
        """Search suggestions for `text`, what the person has typed so far,
        such as `"coffee m"` -> `"coffee maker"`, `"espresso machine"`.

        BehaviorGPT ranks the items this person most likely wants after
        `history` with `text` as their search, and each of the top
        `candidates` items, best first, offers the first of its search terms
        that continues `text`, allowing for a typo. So the same letters give
        different suggestions after different histories. Items without
        keywords (see `add_keywords`) offer their names instead.

        The text is never recorded as a search: call it on every keystroke,
        and `complete` with `Search(...)` once the person searches. For a
        `user_id`, each call is listed by `events.calls` as an
        `"autocomplete"` call, apart from their history. Empty `text`
        suggests what comes after the history. `limit` is at most 20.
        """
        is_search, req, target = self._autocomplete_request(
            text,
            history,
            limit,
            candidates,
            catalog_id,
            catalog_ids,
            filters,
            user,
            user_id,
        )
        endpoint = self.search if is_search else self.recommendations
        response = endpoint.create(payload=req, headers=target.headers)
        target.label(response.items)
        found = suggest(response.items, text, limit)
        if _needs_text_alone(found, limit, text, history, user_id):
            _, req, target = self._autocomplete_request(
                text,
                history,
                limit,
                candidates,
                catalog_id,
                catalog_ids,
                filters,
                user,
                user_id,
                max_trailing_searches=1,
            )
            response = self.search.create(payload=req, headers=target.headers)
            target.label(response.items)
            found = suggest(response.items, text, limit, found)
        return AutocompleteResponse(suggestions=found)

    def sections(
        self,
        history: UserHistoryInput = (),
        *,
        catalog_id: Optional[str] = None,
        catalog_ids: Optional[Sequence[str]] = None,
        max_sections: int = 10,
        max_items_per_section: int = 25,
        min_items_per_section: int = 5,
        filters: Optional[dict[str, Any]] = None,
        register_event: Optional[bool] = None,
        user: Optional[User] = None,
        user_id: Optional[str] = None,
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

        `catalog_ids` groups several catalogs together, as for `complete`: a
        section can hold items of any of them, each weighed by its catalog's
        size, so a small catalog does not crowd out a large one.

        `user_id` and `register_event` work as for `complete`: the search is
        recorded once even when `complete` ran it for them too.
        """
        body, target = self._sections_request(
            history,
            catalog_id,
            filters,
            _registers(register_event, user_id),
            max_sections,
            max_items_per_section,
            min_items_per_section,
            catalog_ids,
            user,
            user_id,
        )
        response = self._sections.create(body, headers=target.headers)
        target.label(item for section in response.sections for item in section.items)
        return response

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
        categories `sections` groups items by, as `add_keywords` does by
        default. Allow a longer `timeout`: it adds time before the catalog is
        ready, 12 minutes for a 19,000-item catalog. If it fails the catalog
        still goes live, without them. Off by default; `add_keywords`
        generates them for the catalog later.
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
        fast: bool = True,
        wait: bool = False,
        interval: float = 5.0,
        timeout: float = 4 * 3600.0,
        on_progress: Optional[Callable[[JobStatus], None]] = None,
        startup_grace: float = 120.0,
    ) -> AddKeywordsJobDetails:
        """Generate the search keywords and browse categories `sections`
        groups items by, for a catalog embedded without them.

        Nothing is re-embedded, and the catalog keeps serving while the job
        runs; a 19,000-item catalog took 12 minutes. Once the job is ready
        the catalog serves with keywords. If it fails, the catalog stays as it
        was. Uploading the catalog again with `embed` replaces the keywords;
        pass `keywords=True` there to keep them.

        By default the language model is asked once per group of about 16
        similar items. `fast=False` asks it once per item instead: each item
        gets more keywords (about 9 instead of 3), so sections have more to
        group by, but it is 6 to 10 times slower, 1 to 2 hours per 20,000
        items; the default `timeout` of 4 hours covers at least 35,000.

        `wait` and the arguments after it work as for `embed`.
        """
        return self.catalogs.add_keywords(
            catalog_id,
            fast=fast,
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
        item_id: ItemRef,
        limit: int = 10,
        offset: int = 0,
        *,
        catalog_id: Optional[str] = None,
        catalog_ids: Optional[Sequence[str]] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> UnboxAIResponse:
        """The items most like `item_id`, an item id or an `Item` of a
        response, most similar first.

        `catalog_id` is the catalog the item is in; for an `Item`, its own.
        `catalog_ids` looks for similar items in several catalogs, the item's
        among them (by default the first): each result says which it is in.
        """
        target, item, item_catalog = self._similar_request(
            item_id, catalog_id, catalog_ids
        )
        response = self.catalogs.get_similar_items(
            item,
            catalog_id=item_catalog,
            limit=limit,
            offset=offset,
            filters=filters,
            headers=target.headers,
        )
        target.label(response.items)
        return response

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

    def umap(
        self,
        *,
        catalog_id: Optional[str] = None,
        catalog_ids: Optional[Sequence[str]] = None,
        wait: float = 0.0,
    ) -> str:
        """The catalog's UMAP, a 2D map of its items, as an HTML page.

        Maps exist only for catalogs of up to 100,000 items, and appear within
        about a minute after the catalog is ready: laying out more items than
        that does not fit in the embedding worker's memory, so larger catalogs
        (such as the shared `retail_catalog`) never get one. A missing map
        raises `UnboxAIError` with status 404 and says which case applies.

        `catalog_ids` lays several catalogs out on one map instead, one colour
        each, so items of different catalogs that sit close together are
        alike. It is drawn on request, within a few minutes, and kept until one
        of the catalogs is uploaded again. Above 100,000 items every catalog is
        sampled evenly, so any catalog can be mapped this way, `retail_catalog`
        too, alone or with others. A map still being drawn when `wait` is up
        raises `UnboxAIError` with status 202; asking again later finds it.

        `wait` is how many seconds to keep asking while the map is missing,
        for a catalog that just became ready. A catalog that never gets a map
        raises only once that time is up.
        """
        deadline = time.monotonic() + wait
        if catalog_ids is not None:
            ids = self._target(catalog_id, catalog_ids).catalog_ids
            joint = self.catalogs.create_map(catalog_ids=ids)
            while not _map_ready(joint, deadline):
                time.sleep(UMAP_POLL)
                joint = self.catalogs.get_map(joint.map_id)
            return self.catalogs.get_map_page(joint.map_id)
        active_catalog: str = catalog_id or self.default_catalog_id
        while True:
            try:
                return self.catalogs.get_umap(catalog_id=active_catalog)
            except UnboxAIError as err:
                if err.status_code != 404 or time.monotonic() + UMAP_POLL > deadline:
                    raise
            time.sleep(UMAP_POLL)


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
        default_catalog_id: Optional[str] = None,
        http_timeout: float | httpx.Timeout = DEFAULT_HTTP_TIMEOUT,
        portal_url: Optional[str] = None,
    ):
        super().__init__(
            market,
            timezone=timezone,
            api_key=api_key,
            base_url=base_url,
            default_catalog_id=default_catalog_id,
            portal_url=portal_url,
        )

        headers = {"x-api-key": self.api_key}
        self._http_client = httpx.AsyncClient(
            base_url=self.base_url, headers=headers, timeout=http_timeout
        )

        self.recommendations = AsyncRecommender(self._http_client)
        self.search = AsyncSearcher(self._http_client)
        self.catalogs = AsyncCatalogs(self._http_client)
        self.events = AsyncEvents(self._http_client, self)
        self._sections = AsyncSectioner(self._http_client)
        assert self.api_key is not None
        self._shop_client = httpx.AsyncClient(
            **shop_http_options(self.api_key, self.portal_url)
        )
        # search, shelves and the rest as the portal's widgets show them
        self.shop = AsyncShop(self._shop_client, self)

    async def __aenter__(self) -> "AsyncUnboxAIClient":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http_client.aclose()
        await self._shop_client.aclose()

    async def complete(
        self,
        history: UserHistoryInput = (),
        limit: int = 10,
        offset: int = 0,
        *,
        catalog_id: Optional[str] = None,
        catalog_ids: Optional[Sequence[str]] = None,
        filters: Optional[dict[str, Any]] = None,
        register_event: Optional[bool] = None,
        max_trailing_searches: int = DEFAULT_TRAILING_SEARCHES,
        predict: Predict = "view",
        user: Optional[User] = None,
        user_id: Optional[str] = None,
    ) -> UnboxAIResponse:
        """Rank the items most likely to come after `history`; see
        `UnboxAIClient.complete`."""
        register = _registers(register_event, user_id)
        is_search, req, target = self._complete_request(
            history,
            limit,
            offset,
            catalog_id,
            filters,
            register,
            max_trailing_searches,
            predict,
            catalog_ids,
            user,
            user_id,
            record_call=_record_call(user_id, register),
        )
        endpoint = self.search if is_search else self.recommendations
        response = await endpoint.create(payload=req, headers=target.headers)
        target.label(response.items)
        return response

    async def autocomplete(
        self,
        text: str,
        history: UserHistoryInput = (),
        limit: int = 5,
        *,
        catalog_id: Optional[str] = None,
        catalog_ids: Optional[Sequence[str]] = None,
        filters: Optional[dict[str, Any]] = None,
        user: Optional[User] = None,
        user_id: Optional[str] = None,
        candidates: int = DEFAULT_SUGGESTION_CANDIDATES,
    ) -> AutocompleteResponse:
        """Search suggestions for `text`; see `UnboxAIClient.autocomplete`."""
        is_search, req, target = self._autocomplete_request(
            text,
            history,
            limit,
            candidates,
            catalog_id,
            catalog_ids,
            filters,
            user,
            user_id,
        )
        endpoint = self.search if is_search else self.recommendations
        response = await endpoint.create(payload=req, headers=target.headers)
        target.label(response.items)
        found = suggest(response.items, text, limit)
        if _needs_text_alone(found, limit, text, history, user_id):
            _, req, target = self._autocomplete_request(
                text,
                history,
                limit,
                candidates,
                catalog_id,
                catalog_ids,
                filters,
                user,
                user_id,
                max_trailing_searches=1,
            )
            response = await self.search.create(payload=req, headers=target.headers)
            target.label(response.items)
            found = suggest(response.items, text, limit, found)
        return AutocompleteResponse(suggestions=found)

    async def sections(
        self,
        history: UserHistoryInput = (),
        *,
        catalog_id: Optional[str] = None,
        catalog_ids: Optional[Sequence[str]] = None,
        max_sections: int = 10,
        max_items_per_section: int = 25,
        min_items_per_section: int = 5,
        filters: Optional[dict[str, Any]] = None,
        register_event: Optional[bool] = None,
        user: Optional[User] = None,
        user_id: Optional[str] = None,
    ) -> SectionsResponse:
        """Items grouped into titled sections; see `UnboxAIClient.sections`."""
        body, target = self._sections_request(
            history,
            catalog_id,
            filters,
            _registers(register_event, user_id),
            max_sections,
            max_items_per_section,
            min_items_per_section,
            catalog_ids,
            user,
            user_id,
        )
        response = await self._sections.create(body, headers=target.headers)
        target.label(item for section in response.sections for item in section.items)
        return response

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
        fast: bool = True,
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
            fast=fast,
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
        item_id: ItemRef,
        limit: int = 10,
        offset: int = 0,
        *,
        catalog_id: Optional[str] = None,
        catalog_ids: Optional[Sequence[str]] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> UnboxAIResponse:
        """The items most like `item_id`; see `UnboxAIClient.similar_items`."""
        target, item, item_catalog = self._similar_request(
            item_id, catalog_id, catalog_ids
        )
        response = await self.catalogs.get_similar_items(
            item,
            catalog_id=item_catalog,
            limit=limit,
            offset=offset,
            filters=filters,
            headers=target.headers,
        )
        target.label(response.items)
        return response

    async def random_item(
        self, catalog_id: Optional[str] = None, window: int = 100
    ) -> Item:
        """One item from the embedded catalog; see `UnboxAIClient.random_item`."""
        top = await self.complete(history=[], catalog_id=catalog_id, limit=window)
        return _pick_random(top, catalog_id or self.default_catalog_id)

    async def umap(
        self,
        *,
        catalog_id: Optional[str] = None,
        catalog_ids: Optional[Sequence[str]] = None,
        wait: float = 0.0,
    ) -> str:
        """The catalog's UMAP as an HTML page; see `UnboxAIClient.umap`."""
        deadline = time.monotonic() + wait
        if catalog_ids is not None:
            ids = self._target(catalog_id, catalog_ids).catalog_ids
            joint = await self.catalogs.create_map(catalog_ids=ids)
            while not _map_ready(joint, deadline):
                await asyncio.sleep(UMAP_POLL)
                joint = await self.catalogs.get_map(joint.map_id)
            return await self.catalogs.get_map_page(joint.map_id)
        active_catalog: str = catalog_id or self.default_catalog_id
        while True:
            try:
                return await self.catalogs.get_umap(catalog_id=active_catalog)
            except UnboxAIError as err:
                if err.status_code != 404 or time.monotonic() + UMAP_POLL > deadline:
                    raise
            await asyncio.sleep(UMAP_POLL)


def _map_ready(joint: CatalogMap, deadline: float) -> bool:
    """Whether the map can be read; raises when it failed, or when it is still
    being drawn and polling once more would pass `deadline`."""
    if joint.status == "ready":
        return True
    catalogs = ", ".join(joint.catalog_ids)
    if joint.status == "failed":
        raise UnboxAIError(
            f"the map of {catalogs} failed: {joint.error}",
            response=joint.model_dump(mode="json"),
        )
    if time.monotonic() + UMAP_POLL > deadline:
        raise UnboxAIError(
            f"the map of {catalogs} is still being drawn ({joint.map_id}); pass "
            "wait= (seconds) to wait for it, e.g. wait=300, or ask again later",
            status_code=202,
            response=joint.model_dump(mode="json"),
        )
    return False


def _registers(register_event: Optional[bool], user_id: Optional[str]) -> bool:
    """Whether a call records its search as the user's: by default when it is
    made for a user_id."""
    if register_event and user_id is None:
        raise ValueError(
            "register_event=True needs a user_id: a search is recorded as the "
            "search of the person it was made for"
        )
    return user_id is not None if register_event is None else register_event


def _record_call(user_id: Optional[str], record: bool) -> Optional[bool]:
    """What a request says about recording the call for its user: nothing, the
    API's default, unless it should not."""
    return False if user_id is not None and not record else None


def _needs_text_alone(
    found: list,
    limit: int,
    text: str,
    history: UserHistoryInput,
    user_id: Optional[str],
) -> bool:
    """Whether to fill out suggestions from a ranking of the text alone. With
    earlier searches, a few letters read as one with them ("golf clubs", "gl"
    as more golf clubs), so the top items may offer no term that continues
    them."""
    return len(found) < limit and bool(text.strip()) and bool(history or user_id)


def _with_search(history: UserHistoryInput, text: str) -> list:
    """`history` with a search for `text` at the end of its last session."""
    if history and isinstance(history[0], (list, tuple)):
        sessions = [list(session) for session in history]
        sessions[-1].append(Search(text))
        return sessions
    return [*history, Search(text)]


def _by_catalog(event: UserEvent, target: _Target) -> list[tuple[str, UserEvent]]:
    """An event as the API reads it, each with the catalog its items are in.
    An event names one catalog, so an order of items from several becomes one
    order per catalog."""
    if isinstance(event, Order) and event.items:
        groups: dict[str, list[CartItem]] = {}
        for item in event.items:
            groups.setdefault(target.of(item.catalog_id), []).append(item)
        if len(groups) == 1:
            return [(next(iter(groups)), event)]
        return [
            (catalog, event.model_copy(update={"items": items}))
            for catalog, items in groups.items()
        ]
    if isinstance(event, (View, AddToCart, RemoveFromCart)):
        return [(target.of(event.catalog_id), event)]
    return [(target.primary, event)]


def _pick_random(top: UnboxAIResponse, catalog_id: str) -> Item:
    if not top.items:
        raise ValueError(f"catalog {catalog_id!r} returned no items")
    return random.choice(top.items)
