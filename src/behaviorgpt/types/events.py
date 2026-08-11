from datetime import UTC, datetime
from typing import List, Literal, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Field

from behaviorgpt.types.domains import Domains


class CartItem(BaseModel):
    product: str
    quantity: int

    def __init__(
        self, product: Optional[str] = None, quantity: Optional[int] = None, **data
    ):
        if product is not None:
            data["product"] = product
        if quantity is not None:
            data["quantity"] = quantity
        super().__init__(**data)


class Search(BaseModel):
    type: Literal["search"] = "search"
    query: str

    def __init__(self, query: Optional[str] = None, **data):
        if query is not None:
            data["query"] = query
        super().__init__(**data)


class View(BaseModel):
    type: Literal["viewItem"] = "viewItem"
    product: str

    def __init__(self, product: Optional[str] = None, **data):
        if product is not None:
            data["product"] = product
        super().__init__(**data)


class AddToCart(BaseModel):
    type: Literal["addToCart"] = "addToCart"
    product: str

    def __init__(self, product: Optional[str] = None, **data):
        if product is not None:
            data["product"] = product
        super().__init__(**data)


class RemoveFromCart(BaseModel):
    type: Literal["removeFromCart"] = "removeFromCart"
    product: str

    def __init__(self, product: Optional[str] = None, **data):
        if product is not None:
            data["product"] = product
        super().__init__(**data)


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
            for item in items:
                if isinstance(item, tuple):
                    product_id, quantity = item
                    parsed_items.append(
                        CartItem(product_id=product_id, quantity=quantity)
                    )
                else:
                    parsed_items.append(item)

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
