import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, cast
from uuid import uuid4

import httpx

from behaviorgpt._exceptions import AuthenticationError
from behaviorgpt.resources.catalogs import Catalogs
from behaviorgpt.resources.recommendations import Recommender
from behaviorgpt.resources.requests import UnboxAIRequester
from behaviorgpt.resources.search import Searcher
from behaviorgpt.types import (
    AddToCart,
    CartItem,
    Domains,
    EmbedJobDetails,
    Item,
    JobStatus,
    Order,
    Search,
    SessionEvent,
    UnboxAIResponse,
    UserEvent,
    UserHistoryInput,
)


class UnboxAIClient:
    def __init__(
        self,
        market: str,
        *,
        timezone: str = "UTC",
        api_key: Optional[str] = None,
        base_url: str = "https://behaviorgpt-northeurope.api.unboxai.com/v1",
        default_catalog_id: str = "sample_catalog",
    ):
        self.api_key = api_key or os.environ.get("UNBOXAI_API_KEY")
        if not self.api_key:
            msg = "API key is missing. Pass it or set UNBOXAI_API_KEY."
            raise AuthenticationError(msg)

        self.base_url = base_url
        self.default_catalog_id = default_catalog_id

        headers = {"x-api-key": self.api_key}
        self._http_client = httpx.Client(base_url=self.base_url, headers=headers)

        self.domains = Domains(market=market)
        self.timezone = timezone

        self.requester = UnboxAIRequester(domains=self.domains, timezone=self.timezone)

        self.recommendations = Recommender(self._http_client)
        self.search = Searcher(self._http_client)
        self.catalogs = Catalogs(self._http_client)

    def complete(
        self,
        history: UserHistoryInput,
        limit: int = 10,
        offset: int = 0,
        *,
        catalog_id: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
        register_event: bool = False,
    ) -> UnboxAIResponse:
        active_catalog: str = catalog_id or self.default_catalog_id
        request_headers = {"x-catalog-id": active_catalog}

        history_events = self.__from_sequence(active_catalog, history)
        last_event = self._get_last_event(history)

        if isinstance(last_event, Search):
            req = self.requester.make(
                query=last_event.query,
                history=history_events[:-1],
                catalog_id=active_catalog,
                limit=limit,
                offset=offset,
                filters=filters,
                register_event=register_event,
            )
            return self.search.create(payload=req, headers=request_headers)

        req = self.requester.make(
            history=history_events,
            catalog_id=active_catalog,
            limit=limit,
            offset=offset,
            filters=filters,
            register_event=register_event,
        )
        return self.recommendations.create(payload=req, headers=request_headers)

    def embed(
        self,
        path: Path | str,
        wait: bool = False,
        interval: float = 5.0,
        timeout: float = 600.0,
        on_progress: Optional[Callable[[JobStatus], None]] = None,
        startup_grace: float = 120.0,
    ) -> EmbedJobDetails:
        """Upload and embed a catalog parquet.

        Pass `wait=True` to block until the embed job finishes. While waiting,
        each new job state is printed once; pass `on_progress` to render
        progress yourself, or `on_progress=lambda _: None` for silence.
        """
        if isinstance(path, str):
            path = Path(path)

        return self.catalogs.embed(
            path,
            wait=wait,
            interval=interval,
            timeout=timeout,
            on_progress=on_progress,
            startup_grace=startup_grace,
        )

    def job_status(self, job_id: str) -> JobStatus:
        return self.catalogs.get_job_status(job_id)

    def similar_products(
        self,
        product_id: str,
        limit: int = 10,
        offset: int = 0,
        *,
        catalog_id: Optional[str] = None,
    ) -> UnboxAIResponse:
        active_catalog: str = catalog_id or self.default_catalog_id
        request_headers = {"x-catalog-id": active_catalog}

        return self.catalogs.get_similar_products(
            product_id,
            catalog_id=active_catalog,
            limit=limit,
            offset=offset,
            headers=request_headers,
        )

    def random_product(
        self, catalog_id: Optional[str] = None, window: int = 100
    ) -> Item:
        """One product from the embedded catalog, highly likely
        you have no product_id at hand.
        """
        top = self.complete(history=[], catalog_id=catalog_id, limit=window)
        if not top.products.items:
            active = catalog_id or self.default_catalog_id
            raise ValueError(f"catalog {active!r} returned no products")
        return random.choice(top.products.items)

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

            from datetime import UTC, datetime, timedelta

            # Base time for the start of the sequence
            base_time = datetime.now(tz=UTC)

            for i, ev in enumerate(session_events):
                if isinstance(ev, AddToCart):
                    cart_state[ev.product] += 1
                elif isinstance(ev, Order):
                    if not getattr(ev, "items", None):
                        ev.cart_id = current_cart_id
                        ev.items = [
                            CartItem(prod, qty) for prod, qty in cart_state.items()
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

    def umap(self, *, catalog_id: Optional[str] = None):
        active_catalog: str = catalog_id or self.default_catalog_id
        return self.catalogs.get_umap(catalog_id=active_catalog)

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
