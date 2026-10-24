import warnings
from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Literal, Optional

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

if TYPE_CHECKING:
    import pandas as pd

from behaviorgpt.types.domains import Domains
from behaviorgpt.types.events import SessionEvent


class Item(BaseModel):
    id: str
    data: dict
    score: float
    # the catalog the item is in: an id is unique only within its catalog.
    # "store_id" on the wire.
    catalog_id: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("catalog_id", "store_id")
    )

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


class RecommendationType(BaseModel):
    name: Literal["view_next", "buy_next"]


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
    # The moment the prediction is made; the server uses the current time if unset.
    timestamp: Optional[datetime] = None
    # How many nearest-neighbour candidates the server explores.
    num_candidates: Optional[int] = Field(default=None, gt=0, le=5_000)
    # Search only: how many searches in a row at the end of the history, the
    # query included, reach the model. Unset, the deployment's history settings
    # apply.
    max_trailing_searches: Optional[int] = Field(default=None, ge=1, le=10)
    # Recommendations only: `buy_next` predicts what else goes in the same
    # purchase. Unset, the server predicts the next item viewed.
    type: Optional[RecommendationType] = None

    @field_validator("timestamp")
    @classmethod
    def require_timezone(cls, timestamp: Optional[datetime]) -> Optional[datetime]:
        if timestamp is not None and timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return timestamp

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
            flat_item = {
                "id": item.id,
                "catalog_id": item.catalog_id,
                "score": item.score,
                **item.data,
            }
            flat_item["price"] = item.extract_price(self.market)
            items_data.append(flat_item)

        return pd.DataFrame(items_data)


class EmbedJobDetails(BaseModel):
    job_id: str
    catalog_id: str
    # display label derived from the uploaded file name, server-side
    catalog_name: str
    uri: str


class AddItemsJobDetails(EmbedJobDetails):
    # distinct item ids in the upload, new and replaced alike
    products: int


class AddKeywordsJobDetails(BaseModel):
    job_id: str
    catalog_id: str
    catalog_name: str


class CatalogSummary(BaseModel):
    catalog_id: str
    # display label derived from the uploaded file name, server-side
    catalog_name: str
    # 'pending', 'ready' or 'failed'; shared catalogs are listed only when ready
    status: str
    # uploaded with this API key, so it counts against the key's catalog slots
    # and this key can delete it
    owned: bool


class SimilarItemsRequest(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True)

    item_id: str = Field(
        validation_alias=AliasChoices("item_id", "product_id"),
        serialization_alias="product_id",
    )
    # the catalog of the item; the results can come from several
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
    # only for an upload with keywords=True: how keyword generation went,
    # e.g. "generating", "done", or "skipped" when the catalog went live
    # without keywords; None until it starts
    keywords: Optional[str] = None
    # why keywords were skipped; only set when keywords is "skipped"
    keywords_error: Optional[str] = None

    def describe(self) -> str:
        """One line for progress output: 'embed fetching 5000/20000',
        'keywords generating 400/2000', 'embed failed: <reason>', or
        'ingest done (2 of 20000 items rejected: ...)'."""
        if self.stage is None:
            return self.status
        step = self.phase or self.stage_state or ""
        if self.done is not None and self.total is not None:
            step = f"{step} {self.done}/{self.total}"
        stage = STAGE_LABELS.get(self.stage, self.stage)
        line = f"{stage} {step}".strip()
        if self.error:
            line = f"{line}: {self.error}"
        skipped_reason = f": {self.keywords_error}" if self.keywords_error else ""
        notes = []
        if self.keywords_skipped:
            if self.stage == KEYWORDS_STAGE:
                line = f"{line}{skipped_reason}"
            else:
                notes.append(f"keywords skipped{skipped_reason}")
        if self.items_rejected:
            notes.append(self.items_rejected.describe())
        if notes:
            line = f"{line} ({'; '.join(notes)})"
        return line

    @property
    def keywords_skipped(self) -> bool:
        """The catalog is live without keywords, so sections cannot use them;
        `keywords_error` says why."""
        return self.keywords == KEYWORDS_SKIPPED


# the server's stage name for keyword generation, and the step it ends in
# when it gives up and the catalog goes live without keywords
KEYWORDS_STAGE = "top_terms"
KEYWORDS_SKIPPED = "skipped"
STAGE_LABELS = {KEYWORDS_STAGE: "keywords"}


class Section(BaseModel):
    """Items grouped under a title, e.g. 'black coats'."""

    title: str
    # "products" on the wire
    items: List[Item] = Field(validation_alias=AliasChoices("items", "products"))


class SectionsResponse(BaseModel):
    sections: List[Section]

    @property
    def titles(self) -> List[str]:
        return [s.title for s in self.sections]


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
