from datetime import datetime
from typing import List, Optional

from behaviorgpt.types import (
    Domains,
    SessionEvent,
    UnboxAIRequest,
)
from behaviorgpt.types.http import RecommendationType


class UnboxAIRequester:
    def __init__(self, domains: Domains, timezone: str = "UTC"):
        self.domains = domains
        self.timezone = timezone

    def make(
        self,
        catalog_id: str,
        query: str | None = None,
        history: List[SessionEvent] | None = None,
        domains: Optional[Domains] = None,
        limit: int = 10,
        offset: int = 0,
        filters: Optional[dict] = None,
        register_event: bool = False,
        timestamp: Optional[datetime] = None,
        num_candidates: Optional[int] = None,
        max_trailing_searches: Optional[int] = None,
        recommendation_type: Optional[RecommendationType] = None,
        timezone: Optional[str] = None,
        user_id: Optional[str] = None,
        record_call: Optional[bool] = None,
        call: Optional[str] = None,
    ) -> UnboxAIRequest:
        return UnboxAIRequest(
            query=query,
            history=history or [],
            store_id=catalog_id,
            domains=domains or self.domains,
            timezone=timezone or self.timezone,
            user_id=user_id,
            limit=limit,
            offset=offset,
            filters=filters or {},
            register_event=register_event,
            timestamp=timestamp,
            num_candidates=num_candidates,
            max_trailing_searches=max_trailing_searches,
            type=recommendation_type,
            record_call=record_call,
            call=call,
        )
