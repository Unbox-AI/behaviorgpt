from typing import List, Optional

import pandas as pd
from pydantic import AliasChoices, BaseModel, Field, model_validator

from behaviorgpt.types.domains import Domains
from behaviorgpt.types.events import SessionEvent


class Item(BaseModel):
    id: str
    data: dict
    score: float

    def extract_price(self, market: str | None = None) -> Optional[float]:
        if "price" in self.data:
            return self.data["price"]

        markets = self.data.get("markets", {})
        if isinstance(markets, dict):
            market_data = markets.get(market)
            if isinstance(market_data, dict) and "price" in market_data:
                return market_data["price"]
        return None


class ProductResponse(BaseModel):
    items: List[Item]
    offset: int
    limit: int


class UnboxAIRequest(BaseModel):
    query: Optional[str] = Field(default=None)

    history: List[SessionEvent] = Field(default_factory=list)
    store_id: str
    domains: Domains
    timezone: str = "UTC"
    limit: int = Field(default=10, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    filters: dict = Field(default_factory=dict)
    register_event: bool = True

    @model_validator(mode="after")
    def validate_pagination(self) -> "UnboxAIRequest":
        """Example: Prevent deep pagination attacks."""
        if self.offset + self.limit > 10000:
            raise ValueError("Offset + limit cannot exceed 10,000")
        return self


class UnboxAIResponse(BaseModel):
    products: ProductResponse

    market: Optional[str] = Field(default=None, exclude=True)

    @property
    def names(self) -> List[str]:
        return [item.data.get("name", "Unknown") for item in self.products.items]

    @property
    def mean_price(self) -> Optional[float]:
        prices = []
        for item in self.products.items:
            price = item.extract_price(self.market)
            if price is not None:
                try:
                    prices.append(float(price))
                except (ValueError, TypeError):
                    pass

        return sum(prices) / len(prices) if prices else None

    def to_pandas(self) -> pd.DataFrame:
        items_data = []
        for item in self.products.items:
            flat_item = {"id": item.id, "score": item.score, **item.data}
            flat_item["price"] = item.extract_price(self.market)
            items_data.append(flat_item)

        return pd.DataFrame(items_data)


class EmbedJobDetails(BaseModel):
    job_id: str
    catalog_id: str
    # display label derived from the uploaded file name, server-side
    catalog_name: str
    uri: str


class SimilarProductsRequest(BaseModel):
    product_id: str
    store_id: str
    limit: int = Field(default=10, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    filters: dict = Field(default_factory=dict)


class RejectedItem(BaseModel):
    id: str
    # why the index refused it, e.g. "failed to parse field [stock] of type
    # [long]"
    reason: str


class ItemsRejected(BaseModel):
    """Items left out of a catalog that is otherwise ready: the index refused
    them for their content. Fix those rows and re-upload to add them."""

    count: int
    total: int
    # the first of them (at most 100); "products" on the wire
    items: List[RejectedItem] = Field(
        default_factory=list, validation_alias=AliasChoices("items", "products")
    )

    def describe(self) -> str:
        ids = ", ".join(repr(i.id) for i in self.items[:3])
        more = f" and {self.count - 3} more" if self.count > 3 else ""
        return f"{self.count} of {self.total} items rejected: {ids}{more}"


class JobStatus(BaseModel):
    job_id: str
    status: str
    stage: Optional[str] = None
    stage_state: Optional[str] = None
    phase: Optional[str] = None
    done: Optional[int] = None
    total: Optional[int] = None
    # why the job failed, e.g. "catalog parquet is empty"; only set when
    # status is "failed", and not for every failure
    error: Optional[str] = None
    # items left out of a ready catalog; the job still succeeded.
    # "products_rejected" on the wire
    items_rejected: Optional[ItemsRejected] = Field(
        default=None,
        validation_alias=AliasChoices("items_rejected", "products_rejected"),
    )

    def describe(self) -> str:
        """One line for progress output: 'embed fetching 5000/20000',
        'embed failed: <reason>', or 'ingest done (2 of 20000 items
        rejected: ...)'."""
        if self.stage is None:
            return self.status
        step = self.phase or self.stage_state or ""
        if self.done is not None and self.total is not None:
            step = f"{step} {self.done}/{self.total}"
        line = f"{self.stage} {step}".strip()
        if self.error:
            line = f"{line}: {self.error}"
        if self.items_rejected:
            line = f"{line} ({self.items_rejected.describe()})"
        return line
