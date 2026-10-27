---
name: portal-app
description: Build a web app that opens from the user's Home on portal.unboxai.com ("Add an app") and reads their catalogs with BehaviorGPT. Use when the user wants to make their own app, tool, page or dashboard for the portal, add an app to the portal, or read the catalog, compare or streams parameters a portal launch passes.
---

# An app that opens from the portal

Goal: a web app the user hosts themselves, added on their portal Home with "Add an app", that opens on whatever catalogs Home has selected and ranks them with the SDK.

The portal is the source of truth for how an app is opened. Its guide is the "Your own app" card under "Build with it" on Home, and "Add an app" links to it. If what it says differs from this file, follow the portal.

## How the portal opens it

The user adds the app with a name, an address, what it does, and which catalogs and events it opens on (one catalog, or several compared; no events, optional, or required). Open sends the browser to the app's own address with what Home had selected in the query, comma-separated:

| Parameter | Holds | Missing when |
|---|---|---|
| `catalog` | the base catalog id | the app opens on no catalog |
| `compare` | up to 4 catalog ids ranked with the base | nothing is compared, or the app opens on one catalog |
| `streams` | event stream names, as on the portal's Events page | none is selected, which means all of them |

Any other parameters in the address the user saved are kept, so `?view=table` still arrives.

Nothing is signed and no key is sent. So:

- The app calls the API with its own `UNBOXAI_API_KEY`: the user's key, kept on the app's server, never in the page. Never print or log it.
- The parameters are what Home had selected, not a permission. Anyone can type them into the address. The key is what limits what can be read, so a deployed app should be private to the user (behind their own sign-in) or not mind showing what the key can read.
- The link does not expire, so it can be bookmarked or opened again without the portal.

## Ranking the selection

The base goes first in `catalog_ids`, then the compared ones. One ranking comes back and each item says which catalog it is in:

```python
from behaviorgpt import Search, UnboxAIClient

client = UnboxAIClient(market="us")  # reads UNBOXAI_API_KEY

def selection(params) -> list[str]:
    ids = [params.get("catalog", ""), *params.get("compare", "").split(",")]
    return [i for i in ids if i]

res = client.complete([Search("rain jacket")], catalog_ids=selection(params), limit=10)
for item in res.items:
    print(item.catalog_id, item.data["name"])
```

For an app that reads events, read each stream with `client.events.list(stream=...)`, or every stream when `streams` is missing; `user_id=` narrows it to one shopper. `client.complete(user_id=...)` ranks for a shopper from their stored events.

With one id, `catalog_ids=[cat]` and `catalog_id=cat` are the same. If the app opens on no catalog and none arrives, fall back to `UNBOXAI_CATALOG_ID` or ask; don't guess an id. The API refuses a catalog the key can't read and catalogs that can't be ranked together; show its message rather than an empty result.

## Workflow

1. Ask what the app should do, and from that which catalogs and events it opens on, since that is what the user picks when adding it.
2. Build it with the server reading the query and calling the SDK. Any framework works; keep the key server side.
3. Run it locally. The portal accepts `http://localhost` addresses, so the user can add `http://localhost:<port>/` and open it from Home while building. Other addresses must be https.
4. Tell the user to add it: Home, "Add an app", with the address and the same catalogs and events choices. It is listed only for their account, tagged External and Private. An account has up to 20.
5. Once deployed, they edit the app on Home to its https address.
