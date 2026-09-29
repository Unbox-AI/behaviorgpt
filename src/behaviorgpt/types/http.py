import warnings
from typing import TYPE_CHECKING, Any, List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    import pandas as pd

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


class ItemsPage(BaseModel):
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
        if self.offset + self.limit > 10000:
            raise ValueError("Offset + limit cannot exceed 10,000")
        return self


class UnboxAIResponse(BaseModel):
    items: List[Item]
    offset: int
    limit: int

    market: Optional[str] = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _flatten_products(cls, data: Any) -> Any:
        # the API nests the page under "products"
        if isinstance(data, dict) and "products" in data:
            data = dict(data)
            products = data.pop("products")
            if isinstance(products, ItemsPage):
                products = products.model_dump()
            data = {**products, **data}
        return data

    @property
    def products(self) -> ItemsPage:
        warnings.warn(
            "`response.products` is deprecated; use `response.items`, "
            "`response.offset` and `response.limit`",
            DeprecationWarning,
            stacklevel=2,
        )
        return ItemsPage(items=self.items, offset=self.offset, limit=self.limit)

    @property
    def names(self) -> List[str]:
        return [item.data.get("name", "Unknown") for item in self.items]

    @property
    def mean_price(self) -> Optional[float]:
        prices = []
        for item in self.items:
            price = item.extract_price(self.market)
            if price is not None:
                try:
                    prices.append(float(price))
                except (ValueError, TypeError):
                    pass

        return sum(prices) / len(prices) if prices else None

    def to_pandas(self) -> "pd.DataFrame":
        # pandas is optional: only this method needs it
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "to_pandas() needs pandas; install it with `pip install pandas`"
            ) from None

        items_data = []
        for item in self.items:
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


class SimilarItemsRequest(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True)

    item_id: str = Field(
        validation_alias=AliasChoices("item_id", "product_id"),
        serialization_alias="product_id",
    )
    store_id: str
    limit: int = Field(default=10, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    filters: dict = Field(default_factory=dict)


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

    def describe(self) -> str:
        """One line for progress output: 'embed fetching 5000/20000', or
        'embed failed: <reason>'."""
        if self.stage is None:
            return self.status
        step = self.phase or self.stage_state or ""
        if self.done is not None and self.total is not None:
            step = f"{step} {self.done}/{self.total}"
        line = f"{self.stage} {step}".strip()
        if self.error:
            line = f"{line}: {self.error}"
        return line


DEPRECATED_NAMES = {
    "ProductResponse": "ItemsPage",
    "SimilarProductsRequest": "SimilarItemsRequest",
}


def __getattr__(name: str) -> Any:
    if name in DEPRECATED_NAMES:
        new = DEPRECATED_NAMES[name]
        warnings.warn(
            f"`{name}` is deprecated, use `{new}`", DeprecationWarning, stacklevel=2
        )
        return globals()[new]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
