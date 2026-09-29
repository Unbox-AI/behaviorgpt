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


class RemoveFromCart(_ItemEvent):
    type: Literal["removeFromCart"] = "removeFromCart"


class Order(BaseModel):
    type: Literal["order"] = "order"

    cart_id: Optional[str] = None
    items: Optional[List[CartItem]] = None

    def __init__(
        self,
        cart_id: Optional[str] = None,
        items: Optional[List[Union[CartItem, Tuple[str, int]]]] = None,
        **data,
    ):
        if cart_id is not None:
            data["cart_id"] = cart_id

        if items is not None:
            parsed_items = []
            for entry in items:
                if isinstance(entry, tuple):
                    item_id, quantity = entry
                    parsed_items.append(CartItem(item_id, quantity))
                else:
                    parsed_items.append(entry)

            data["items"] = parsed_items

        super().__init__(**data)


UserEvent = Union[Search, AddToCart, View, RemoveFromCart, Order]


class SessionEvent(BaseModel):
    store_id: str
    domains: Domains
    timezone: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    event: UserEvent
    session_id: str


UserHistoryInput = Union[Sequence[UserEvent], Sequence[Sequence[UserEvent]]]
