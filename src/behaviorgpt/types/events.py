import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Sequence, Tuple, Union

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from behaviorgpt.types.domains import Domains

if TYPE_CHECKING:
    from behaviorgpt.types.http import Item


class ExternalItem(BaseModel):
    """An item in none of your catalogs, by what it is: its name, with its brand
    and categories as a catalog row has them (see the catalog format), and its
    picture. It needs a name or a picture. Put it in a history where an item id
    goes: `View(ExternalItem("Linen shirt", brand="Acme"))`,
    `Order([ExternalItem(image="photo.jpg")])`. BehaviorGPT reads it as it reads
    a catalog item with that row; `similar_items(ExternalItem(image=...))` finds
    the catalog items that look most like a picture.

    `image` is an https link to the picture, or the picture itself: a file's
    path, its bytes or a data: URL, at most 1024 pixels a side. It is not stored,
    and the item is in no results."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Optional[str] = None
    brand: Optional[str] = None
    # a comma-separated path, e.g. "Clothing, Shirts"
    categories: Optional[str] = None
    # an https link, or a data:image/...;base64, URL
    image: Optional[str] = None

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        image: Union[str, bytes, Path, None] = None,
        **data: Any,
    ):
        if name is not None:
            data["name"] = name
        if image is not None:
            data["image"] = _picture(image)
        super().__init__(**data)

    @model_validator(mode="after")
    def _something_to_embed(self) -> "ExternalItem":
        if not self.name and not self.image:
            raise ValueError("an ExternalItem needs a name or an image")
        return self


def _picture(image: Union[str, bytes, Path]) -> str:
    """A picture as the API takes it: an https link, or a data: URL of the
    picture (read from a file's path or bytes)."""
    from behaviorgpt.resources.shop import picture_source

    source = picture_source(image)
    if "url" in source:
        if not source["url"].startswith("https://"):
            raise ValueError("an ExternalItem's image link must be https")
        return source["url"]
    return source["image"]


# an item id, an `Item` of a response, which knows its catalog, or an item in
# none of the catalogs
ItemRef = Union[str, "Item", ExternalItem]


class _Timed(BaseModel):
    """An event's optional `timestamp`: when it happened, timezone-aware. Not
    sent as part of the event: it becomes the time the API reads it at."""

    timestamp: Optional[datetime] = Field(default=None, exclude=True)

    @field_validator("timestamp")
    @classmethod
    def _aware(cls, value: Optional[datetime]) -> Optional[datetime]:
        if value is not None and value.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return value


class _ItemEvent(BaseModel):
    # the API calls the item "product"
    model_config = ConfigDict(serialize_by_alias=True)

    item: Union[str, ExternalItem] = Field(
        validation_alias=AliasChoices("item", "product"),
        serialization_alias="product",
    )
    # the catalog the item is in, for a request over several catalogs
    # (`catalog_ids=`); unset, the request's first catalog. Not sent as part of
    # the event: it becomes the event's store_id.
    catalog_id: Optional[str] = Field(default=None, exclude=True)

    def __init__(self, item: Optional[ItemRef] = None, **data):
        if "product" in data:
            warnings.warn(
                "`product=` is deprecated, use `item=`",
                DeprecationWarning,
                stacklevel=2,
            )
            product = data.pop("product")
            item = product if item is None else item
        if item is not None and not isinstance(item, (str, ExternalItem)):
            if data.get("catalog_id") is None:
                data["catalog_id"] = item.catalog_id
            item = item.id
        if item is not None:
            data["item"] = item
        super().__init__(**data)

    @model_serializer(mode="wrap")
    def _wire(self, handler: SerializerFunctionWrapHandler) -> dict:
        """An id is sent as `product`; an item in no catalog as `item`."""
        data = handler(self)
        if isinstance(self.item, ExternalItem):
            data.pop("product", None)
            data.pop("item", None)
            data["item"] = self.item.model_dump(mode="json", exclude_none=True)
        return data

    @property
    def product(self) -> Union[str, ExternalItem]:
        warnings.warn(
            "`.product` is deprecated, use `.item`", DeprecationWarning, stacklevel=2
        )
        return self.item


class CartItem(_ItemEvent):
    quantity: int

    def __init__(
        self, item: Optional[ItemRef] = None, quantity: Optional[int] = None, **data
    ):
        if quantity is not None:
            data["quantity"] = quantity
        super().__init__(item, **data)


class Search(_Timed):
    type: Literal["search"] = "search"
    query: str

    def __init__(self, query: Optional[str] = None, **data):
        if query is not None:
            data["query"] = query
        super().__init__(**data)


class View(_ItemEvent, _Timed):
    type: Literal["viewItem"] = "viewItem"


class AddToCart(_ItemEvent, _Timed):
    type: Literal["addToCart"] = "addToCart"

    def __init__(self, item: Optional[ItemRef] = None, **data):
        warnings.warn(
            "`AddToCart` is deprecated and will be removed in 0.7: the model reads "
            "it the same as `View`. Send a `View`, and list what was bought in "
            "`Order([...])`",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(item, **data)


class RemoveFromCart(_ItemEvent, _Timed):
    type: Literal["removeFromCart"] = "removeFromCart"

    def __init__(self, item: Optional[ItemRef] = None, **data):
        warnings.warn(
            "`RemoveFromCart` is deprecated and will be removed in 0.7: it does not "
            "change results. Leave the item out of `Order([...])` instead",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(item, **data)


OrderItem = Union[ItemRef, CartItem, Tuple[ItemRef, int]]


class Order(_Timed):
    """A finished purchase of `items`: item ids, `Item`s, `(item, quantity)`
    pairs or `CartItem`s. The model reads only which items were bought;
    quantities and `cart_id` do not change results.

    In a request over several catalogs the items can come from any of them:
    each is in the catalog its `Item` or `CartItem(catalog_id=)` names."""

    type: Literal["order"] = "order"

    cart_id: Optional[str] = None
    items: Optional[List[CartItem]] = None

    def __init__(
        self,
        items: Optional[Union[str, Sequence[OrderItem]]] = None,
        cart_id: Optional[str] = None,
        **data,
    ):
        if isinstance(items, str):
            warnings.warn(
                "`Order`'s first argument is now the list of items bought; pass "
                "`cart_id=` by name. A single string is still read as the cart id "
                "until 0.7",
                DeprecationWarning,
                stacklevel=2,
            )
            items, cart_id = cart_id, items

        if cart_id is not None:
            data["cart_id"] = cart_id

        if items is None and "items" not in data:
            warnings.warn(
                "`Order()` without items is deprecated and will be removed in 0.7: "
                "list what was bought, `Order([item, ...])`",
                DeprecationWarning,
                stacklevel=2,
            )
        elif items is not None:
            data["items"] = [_order_item(entry) for entry in items]

        super().__init__(**data)


def _order_item(entry: OrderItem) -> CartItem:
    if isinstance(entry, CartItem):
        return entry
    if isinstance(entry, tuple):
        item, quantity = entry
        return CartItem(item, quantity)
    return CartItem(entry, 1)


UserEvent = Union[Search, AddToCart, View, RemoveFromCart, Order]


class SessionEvent(BaseModel):
    store_id: str
    domains: Domains
    timezone: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    event: UserEvent
    session_id: str


UserHistoryInput = Union[Sequence[UserEvent], Sequence[Sequence[UserEvent]]]
