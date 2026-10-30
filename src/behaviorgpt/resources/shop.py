"""`client.shop`: the shopping calls the portal's widgets make, from a server.

    client = UnboxAIClient(market="us", default_catalog_id="cat_...")
    client.shop.suggest("lin", user_id=uid)
    client.shop.search("linen sofa", user_id=uid, only=Only(max_price=900))
    client.shop.recommend("bought_together", item="sku-123")

`complete()` ranks a catalog for a history and answers raw items; these answer
what a shop shows, from the same code as the widgets on a page and in an app:
suggestions phrased as searches, pages with `more`, a shelf without the items
the shopper already saw or the same product twice, bought-together as one
product of each kind with its kind's name, titled sections, an item with its
keywords, a search by picture. They call the portal
(`https://portal.unboxai.com/api/v1/shop/`) with your API key, and record
nothing: a call never becomes the shopper's history.

`user_id` names a shopper whose events you record (`events.track`, the tracking
script's `unbox_uid`, an app's tracker): their recorded searches and views come
first. `history` adds what they did that is not recorded yet, as `Search` and
`View` events, oldest first. `user` says who they are, as in `complete`.
"""

import asyncio
import base64
import mimetypes
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Union

import httpx

from behaviorgpt.resources._shared import handle_response
from behaviorgpt.types.events import Search, View
from behaviorgpt.types.shop import (
    Only,
    PictureKind,
    ShelfKind,
    ShopItem,
    ShopPage,
    ShopPicture,
    ShopSection,
    ShopSuggestion,
)
from behaviorgpt.types.user import User

if TYPE_CHECKING:
    from behaviorgpt._client import _BaseClient

DEFAULT_PORTAL_URL = "https://portal.unboxai.com"
SHOP_PATH = "/api/v1/shop/"
SHOP_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
# the portal keeps a shopper's last few events of a call's history
MAX_HISTORY = 20
# a picture as a data: URL, as the portal takes it
MAX_PICTURE_CHARS = 8_000_000
PICTURE_POLL = 1.5
PICTURE_WAIT = 180.0

Picture = Union[str, Path, bytes]
History = Sequence[Union[Search, View]]


def shop_http_options(api_key: str, portal_url: str) -> dict[str, Any]:
    """The portal's shop API, with the key as its bearer."""
    return {
        "base_url": portal_url.rstrip("/") + SHOP_PATH,
        "headers": {"authorization": f"Bearer {api_key}"},
        "timeout": SHOP_TIMEOUT,
    }


def _event(event: Union[Search, View]) -> dict:
    if isinstance(event, Search):
        return {"type": "search", "query": event.query}
    if isinstance(event, View):
        return {"type": "view", "item": event.item}
    raise TypeError(
        f"history takes Search and View events, not {type(event).__name__}: "
        "for what goes with an order use recommend('after_purchase', items=...)"
    )


def _who(context: "_BaseClient", user: Optional[User]) -> dict:
    """Who the shopper is: `user`, with the client's market and time zone where
    it names none."""
    who: dict[str, Any] = user.model_dump(exclude_none=True) if user else {}
    market = who.get("market") or context.domains.market
    if isinstance(market, str) and len(market) == 2 and market.isalpha():
        who["market"] = market.upper()
    else:
        who.pop("market", None)
    who.setdefault("timezone", context.timezone)
    return who


def _only(only: Optional[Only]) -> Optional[dict]:
    if only is None:
        return None
    body = only.model_dump(exclude_none=True)
    body = {k: v for k, v in body.items() if v not in ([], None)}
    return body or None


def shop_body(
    context: "_BaseClient",
    *,
    catalog_id: Optional[str],
    user_id: Optional[str],
    history: History,
    user: Optional[User],
    **fields: Any,
) -> dict:
    """A call's JSON: the catalog, the shopper, and `fields` without the unset."""
    body: dict[str, Any] = {
        "catalog": catalog_id or context.default_catalog_id,
        "who": _who(context, user),
    }
    if user_id is not None:
        body["user"] = str(user_id)
    events = [_event(e) for e in history][-MAX_HISTORY:]
    if events:
        body["history"] = events
    body.update({k: v for k, v in fields.items() if v is not None})
    return body


def picture_source(picture: Picture) -> dict:
    """A picture as the portal takes it: a link as `url`, else the picture
    itself as a data: URL in `image`. A string that is not a link or a data:
    URL is a file's path."""
    if isinstance(picture, str):
        text = picture.strip()
        if text.startswith(("http://", "https://")):
            return {"url": text}
        if text.startswith("data:"):
            return _sized({"image": text})
        picture = Path(text)
    if isinstance(picture, Path):
        if not picture.is_file():
            raise ValueError(f"{picture} is not a file, a link or a data: URL")
        kind = mimetypes.guess_type(picture.name)[0] or "image/jpeg"
        data = picture.read_bytes()
    else:
        kind, data = _sniff(bytes(picture)), bytes(picture)
    encoded = base64.b64encode(data).decode("ascii")
    return _sized({"image": f"data:{kind};base64,{encoded}"})


def _sized(source: dict) -> dict:
    if len(source["image"]) > MAX_PICTURE_CHARS:
        raise ValueError(
            "the picture is too large to send: make it at most 1024 pixels a side first"
        )
    return source


def _sniff(data: bytes) -> str:
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"GIF8"):
        return "image/gif"
    return "image/jpeg"


def _picture(answer: dict) -> ShopPicture:
    return ShopPicture(
        status="ready" if answer.get("status") == "ready" else "failed",
        picture=answer.get("picture"),
        items=answer.get("items") or [],
        error=answer.get("error"),
    )


def _pending(answer: dict) -> bool:
    return answer.get("status") in ("pending", "embedding")


class Shop:
    """BehaviorGPT's shopping calls, as the portal's widgets make them; see the
    module's description."""

    def __init__(self, http_client: httpx.Client, context: "_BaseClient"):
        self._client = http_client
        self._context = context

    def _call(self, path: str, body: dict) -> dict:
        return handle_response(self._client.post(path, json=body))

    def suggest(
        self,
        text: str = "",
        *,
        limit: int = 6,
        user_id: Optional[str] = None,
        history: History = (),
        user: Optional[User] = None,
        catalog_id: Optional[str] = None,
    ) -> list[ShopSuggestion]:
        """Searches to offer for the text typed so far, for this shopper: an
        item's keywords, or its name made a short phrase. With no text, what
        they likely search next."""
        body = shop_body(
            self._context,
            catalog_id=catalog_id,
            user_id=user_id,
            history=history,
            user=user,
            text=text[:100],
            limit=limit,
        )
        return [ShopSuggestion(**s) for s in self._call("suggest", body)["suggestions"]]

    def search(
        self,
        query: str,
        *,
        limit: int = 12,
        offset: int = 0,
        only: Optional[Only] = None,
        user_id: Optional[str] = None,
        history: History = (),
        user: Optional[User] = None,
        catalog_id: Optional[str] = None,
    ) -> ShopPage:
        """The results for `query`, best for this shopper first, a page at a time."""
        body = shop_body(
            self._context,
            catalog_id=catalog_id,
            user_id=user_id,
            history=history,
            user=user,
            query=query.strip()[:200],
            limit=limit,
            offset=offset,
            only=_only(only),
        )
        return ShopPage(**self._call("search", body))

    def browse(
        self,
        only: Only,
        *,
        limit: int = 12,
        offset: int = 0,
        user_id: Optional[str] = None,
        history: History = (),
        user: Optional[User] = None,
        catalog_id: Optional[str] = None,
    ) -> ShopPage:
        """A category page: the items `only` admits, best for this shopper first.
        A shopper nothing is known about gets the category's best for anyone."""
        if _only(only) is None:
            raise ValueError("browse needs a category, a brand or a price in `only`")
        body = shop_body(
            self._context,
            catalog_id=catalog_id,
            user_id=user_id,
            history=history,
            user=user,
            only=_only(only),
            limit=limit,
            offset=offset,
        )
        return ShopPage(**self._call("browse", body))

    def recommend(
        self,
        kind: ShelfKind = "for_you",
        *,
        item: Optional[str] = None,
        items: Optional[Sequence[str]] = None,
        limit: int = 8,
        only: Optional[Only] = None,
        user_id: Optional[str] = None,
        history: History = (),
        user: Optional[User] = None,
        catalog_id: Optional[str] = None,
    ) -> list[ShopItem]:
        """A shelf. `for_you`: what the shopper likely wants next. `similar`:
        what people view after `item`. `bought_together`: what goes in the same
        purchase as `item`, one product of each kind, each with its kind's name
        as `tag`. `cart`: what goes with `items`. `after_purchase`: what the
        shopper wants next now that they bought `items`, one of each kind. No
        shelf shows what the shopper viewed or bought, or one product twice."""
        body = shop_body(
            self._context,
            catalog_id=catalog_id,
            user_id=user_id,
            history=history,
            user=user,
            kind=kind,
            item=item,
            items=list(items) if items else None,
            limit=limit,
            only=_only(only),
        )
        return [ShopItem(**i) for i in self._call("recommend", body)["items"]]

    def sections(
        self,
        query: str = "",
        *,
        sections: int = 4,
        limit: int = 8,
        only: Optional[Only] = None,
        user_id: Optional[str] = None,
        history: History = (),
        user: Optional[User] = None,
        catalog_id: Optional[str] = None,
    ) -> list[ShopSection]:
        """Titled shelves, best first: what the shopper likely wants next, or
        with `query` its results, grouped by kind ("Linen sofas"). The titles
        come from the catalog's keywords, so a catalog without them has none."""
        body = shop_body(
            self._context,
            catalog_id=catalog_id,
            user_id=user_id,
            history=history,
            user=user,
            query=query.strip()[:200] or None,
            sections=sections,
            limit=limit,
            only=_only(only),
        )
        return [ShopSection(**s) for s in self._call("sections", body)["sections"]]

    def item(self, item_id: str, *, catalog_id: Optional[str] = None) -> ShopItem:
        """One item's card, with its search terms."""
        body = shop_body(
            self._context,
            catalog_id=catalog_id,
            user_id=None,
            history=(),
            user=None,
            item=item_id,
        )
        return ShopItem(**self._call("item", body)["item"])

    def visual(
        self,
        picture: Picture,
        *,
        kind: PictureKind = "look",
        words: str = "",
        limit: int = 12,
        user_id: Optional[str] = None,
        history: History = (),
        user: Optional[User] = None,
        catalog_id: Optional[str] = None,
        on_progress: Optional[Callable[[str], None]] = None,
        timeout: float = PICTURE_WAIT,
    ) -> ShopPicture:
        """The products a picture looks like (`look`), or what this shopper would
        view next after it (`for_you`). `picture` is a file's path, its bytes, a
        data: URL, or a link to a picture or to a page with one. A new picture
        takes BehaviorGPT a little while to read; `on_progress` hears what it is
        doing, and this waits for it up to `timeout` seconds."""
        common = dict(
            catalog_id=catalog_id, user_id=user_id, history=history, user=user
        )
        body = shop_body(
            self._context,
            **common,
            **picture_source(picture),
            words=words or None,
            kind=kind,
            limit=limit,
        )
        answer = self._call("visual", body)
        until = time.monotonic() + timeout
        while _pending(answer):
            if on_progress:
                on_progress(answer.get("phase") or "BehaviorGPT is reading the picture")
            if time.monotonic() > until:
                raise TimeoutError("BehaviorGPT took too long to read the picture")
            time.sleep(PICTURE_POLL)
            status = shop_body(
                self._context,
                **common,
                picture=answer["picture"],
                kind=kind,
                limit=limit,
            )
            answer = self._call("visual/status", status)
        return _picture(answer)


class AsyncShop:
    """`Shop` for asyncio code: the same calls, awaited."""

    def __init__(self, http_client: httpx.AsyncClient, context: "_BaseClient"):
        self._client = http_client
        self._context = context

    async def _call(self, path: str, body: dict) -> dict:
        return handle_response(await self._client.post(path, json=body))

    async def suggest(
        self,
        text: str = "",
        *,
        limit: int = 6,
        user_id: Optional[str] = None,
        history: History = (),
        user: Optional[User] = None,
        catalog_id: Optional[str] = None,
    ) -> list[ShopSuggestion]:
        """See `Shop.suggest`."""
        body = shop_body(
            self._context,
            catalog_id=catalog_id,
            user_id=user_id,
            history=history,
            user=user,
            text=text[:100],
            limit=limit,
        )
        answer = await self._call("suggest", body)
        return [ShopSuggestion(**s) for s in answer["suggestions"]]

    async def search(
        self,
        query: str,
        *,
        limit: int = 12,
        offset: int = 0,
        only: Optional[Only] = None,
        user_id: Optional[str] = None,
        history: History = (),
        user: Optional[User] = None,
        catalog_id: Optional[str] = None,
    ) -> ShopPage:
        """See `Shop.search`."""
        body = shop_body(
            self._context,
            catalog_id=catalog_id,
            user_id=user_id,
            history=history,
            user=user,
            query=query.strip()[:200],
            limit=limit,
            offset=offset,
            only=_only(only),
        )
        return ShopPage(**await self._call("search", body))

    async def browse(
        self,
        only: Only,
        *,
        limit: int = 12,
        offset: int = 0,
        user_id: Optional[str] = None,
        history: History = (),
        user: Optional[User] = None,
        catalog_id: Optional[str] = None,
    ) -> ShopPage:
        """See `Shop.browse`."""
        if _only(only) is None:
            raise ValueError("browse needs a category, a brand or a price in `only`")
        body = shop_body(
            self._context,
            catalog_id=catalog_id,
            user_id=user_id,
            history=history,
            user=user,
            only=_only(only),
            limit=limit,
            offset=offset,
        )
        return ShopPage(**await self._call("browse", body))

    async def recommend(
        self,
        kind: ShelfKind = "for_you",
        *,
        item: Optional[str] = None,
        items: Optional[Sequence[str]] = None,
        limit: int = 8,
        only: Optional[Only] = None,
        user_id: Optional[str] = None,
        history: History = (),
        user: Optional[User] = None,
        catalog_id: Optional[str] = None,
    ) -> list[ShopItem]:
        """See `Shop.recommend`."""
        body = shop_body(
            self._context,
            catalog_id=catalog_id,
            user_id=user_id,
            history=history,
            user=user,
            kind=kind,
            item=item,
            items=list(items) if items else None,
            limit=limit,
            only=_only(only),
        )
        answer = await self._call("recommend", body)
        return [ShopItem(**i) for i in answer["items"]]

    async def sections(
        self,
        query: str = "",
        *,
        sections: int = 4,
        limit: int = 8,
        only: Optional[Only] = None,
        user_id: Optional[str] = None,
        history: History = (),
        user: Optional[User] = None,
        catalog_id: Optional[str] = None,
    ) -> list[ShopSection]:
        """See `Shop.sections`."""
        body = shop_body(
            self._context,
            catalog_id=catalog_id,
            user_id=user_id,
            history=history,
            user=user,
            query=query.strip()[:200] or None,
            sections=sections,
            limit=limit,
            only=_only(only),
        )
        answer = await self._call("sections", body)
        return [ShopSection(**s) for s in answer["sections"]]

    async def item(self, item_id: str, *, catalog_id: Optional[str] = None) -> ShopItem:
        """See `Shop.item`."""
        body = shop_body(
            self._context,
            catalog_id=catalog_id,
            user_id=None,
            history=(),
            user=None,
            item=item_id,
        )
        return ShopItem(**(await self._call("item", body))["item"])

    async def visual(
        self,
        picture: Picture,
        *,
        kind: PictureKind = "look",
        words: str = "",
        limit: int = 12,
        user_id: Optional[str] = None,
        history: History = (),
        user: Optional[User] = None,
        catalog_id: Optional[str] = None,
        on_progress: Optional[Callable[[str], None]] = None,
        timeout: float = PICTURE_WAIT,
    ) -> ShopPicture:
        """See `Shop.visual`."""
        common = dict(
            catalog_id=catalog_id, user_id=user_id, history=history, user=user
        )
        body = shop_body(
            self._context,
            **common,
            **picture_source(picture),
            words=words or None,
            kind=kind,
            limit=limit,
        )
        answer = await self._call("visual", body)
        until = time.monotonic() + timeout
        while _pending(answer):
            if on_progress:
                on_progress(answer.get("phase") or "BehaviorGPT is reading the picture")
            if time.monotonic() > until:
                raise TimeoutError("BehaviorGPT took too long to read the picture")
            await asyncio.sleep(PICTURE_POLL)
            status = shop_body(
                self._context,
                **common,
                picture=answer["picture"],
                kind=kind,
                limit=limit,
            )
            answer = await self._call("visual/status", status)
        return _picture(answer)
