from typing import List, Optional

from behaviorgpt.types import (
    Domains,
    SessionEvent,
    UnboxAIRequest,
)


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
    ) -> UnboxAIRequest:
        return UnboxAIRequest(
            query=query,
            history=history or [],
            store_id=catalog_id,
            domains=domains or self.domains,
            timezone=self.timezone,
            limit=limit,
            offset=offset,
            filters=filters or {},
            register_event=register_event,
        )
