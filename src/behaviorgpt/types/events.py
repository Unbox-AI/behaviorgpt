import warnings
from datetime import UTC, datetime
from typing import List, Literal, Optional, Sequence, Tuple, Union

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from behaviorgpt.types.domains import Domains


class _ItemEvent(BaseModel):
    # the API calls the item "product"
    model_config = ConfigDict(serialize_by_alias=True)

    item: str = Field(
        validation_alias=AliasChoices("item", "product"),
        serialization_alias="product",
    )

    def __init__(self, item: Optional[str] = None, **data):
        if "product" in data:
            warnings.warn(
                "`product=` is deprecated, use `item=`",
                DeprecationWarning,
                stacklevel=2,
            )
            product = data.pop("product")
            item = product if item is None else item
        if item is not None:
            data["item"] = item
        super().__init__(**data)

    @property
    def product(self) -> str:
        warnings.warn(
            "`.product` is deprecated, use `.item`", DeprecationWarning, stacklevel=2
        )
        return self.item


class CartItem(_ItemEvent):
    quantity: int

    def __init__(
        self, item: Optional[str] = None, quantity: Optional[int] = None, **data
    ):
        if quantity is not None:
            data["quantity"] = quantity
        super().__init__(item, **data)


class Search(BaseModel):
    type: Literal["search"] = "search"
    query: str

    def __init__(self, query: Optional[str] = None, **data):
        if query is not None:
            data["query"] = query
        super().__init__(**data)


class View(_ItemEvent):
    type: Literal["viewItem"] = "viewItem"


class AddToCart(_ItemEvent):
    type: Literal["addToCart"] = "addToCart"

    def __init__(self, item: Optional[str] = None, **data):
        warnings.warn(
            "`AddToCart` is deprecated and will be removed in 0.7: the model reads "
            "it the same as `View`. Send a `View`, and list what was bought in "
            "`Order([...])`",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(item, **data)


class RemoveFromCart(_ItemEvent):
    type: Literal["removeFromCart"] = "removeFromCart"

    def __init__(self, item: Optional[str] = None, **data):
        warnings.warn(
            "`RemoveFromCart` is deprecated and will be removed in 0.7: it does not "
            "change results. Leave the item out of `Order([...])` instead",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(item, **data)


OrderItem = Union[str, CartItem, Tuple[str, int]]


class Order(BaseModel):
    """A finished purchase of `items`: item ids, `(id, quantity)` pairs or
    `CartItem`s. The model reads only which items were bought; quantities and
    `cart_id` do not change results."""

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
    if isinstance(entry, str):
        return CartItem(entry, 1)
    if isinstance(entry, tuple):
        item_id, quantity = entry
        return CartItem(item_id, quantity)
    return entry


UserEvent = Union[Search, AddToCart, View, RemoveFromCart, Order]


class SessionEvent(BaseModel):
    store_id: str
    domains: Domains
    timezone: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    event: UserEvent
    session_id: str


UserHistoryInput = Union[Sequence[UserEvent], Sequence[Sequence[UserEvent]]]
