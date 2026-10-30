"""What `client.shop` answers: the cards the portal's widgets show, ready for a
page, an app or an email."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

ShelfKind = Literal["for_you", "similar", "bought_together", "cart", "after_purchase"]
PictureKind = Literal["look", "for_you"]


class Only(BaseModel):
    """Which items a call may show: in any of `categories` (at any level of an
    item's category path), of any of `brands`, priced within the bounds. Values
    are matched exactly, as the catalog has them; every condition given must hold.

        Only(categories=["Sofas", "Armchairs"], max_price=900)
    """

    model_config = ConfigDict(frozen=True)

    categories: list[str] = Field(default_factory=list, max_length=10)
    brands: list[str] = Field(default_factory=list, max_length=10)
    min_price: Optional[float] = Field(default=None, ge=0)
    max_price: Optional[float] = Field(default=None, ge=0)


class ShopItem(BaseModel):
    """An item as a shop shows it."""

    id: str
    name: str
    brand: Optional[str] = None
    # an http(s) address, or None
    image: Optional[str] = None
    price: Optional[float] = None
    currency: Optional[str] = None
    # the item's page: from the catalog's file, its links file or the tracking script
    url: Optional[str] = None
    # `item()` only: its search terms, best first
    keywords: Optional[list[str]] = None
    # bought_together and after_purchase: the name of its kind ("Floor lamps")
    tag: Optional[str] = None
    # `visual()` only: the item a link was the very picture of
    exact: Optional[bool] = None


class ShopSuggestion(BaseModel):
    """A search to offer as the shopper types."""

    text: str
    # one of a ranked item's keywords ("keyword"), or its name made a phrase ("name")
    source: str
    # the item that offered it
    item: str


class ShopPage(BaseModel):
    """A page of search results or of a category."""

    items: list[ShopItem]
    offset: int
    # whether another page follows
    more: bool


class ShopSection(BaseModel):
    """A titled shelf, such as "Linen sofas"."""

    title: str
    items: list[ShopItem]


class ShopPicture(BaseModel):
    """The products a picture looks like, once BehaviorGPT has read it."""

    status: Literal["ready", "failed"]
    # the picture's item in BehaviorGPT; None for a link to the catalog's own picture
    picture: Optional[str] = None
    items: list[ShopItem] = Field(default_factory=list)
    # why it failed
    error: Optional[str] = None
