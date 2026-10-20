# BehaviorGPT

**The foundation model for behavior.**

BehaviorGPT is Unbox AI's Large Behavioral Model. It is trained on long sequences of things people actually did (viewed this, added that, bought the other) and predicts what comes next, so search, recommendations and personalization are the same call with different histories. BehaviorGPT-v4 is a single 12.5B-parameter model pretrained on 150 billion user actions across retail, engagement and payments, and it ranks catalogs it has never seen with no training.

[Website](https://unboxai.com/behaviorgpt) · [Live demo](https://behaviorgpt.unboxai.com/) · [Research](https://research.unboxai.com) · [Notebook](https://github.com/Unbox-AI/behaviorgpt/blob/main/notebooks/showcase.ipynb)

This repository holds the Python SDK for the Unbox AI API and a notebook that walks through the model step by step.

Three apps built on this SDK: two that recommend, on a store's catalog and a museum's, and one that reads the same rankings as a fraud detector. More on each in [Demo and examples](#demo-and-examples).

<table>
  <tr>
    <td width="33%"><a href="https://behaviorgpt.unboxai.com/"><img src="https://raw.githubusercontent.com/Unbox-AI/behaviorgpt/main/docs/images/storefront.png" alt="The storefront demo: after a search for golf clubs, a search for gloves returns golf gloves"></a></td>
    <td width="33%"><a href="https://github.com/Unbox-AI/aic-artworks"><img src="https://raw.githubusercontent.com/Unbox-AI/behaviorgpt/main/docs/images/curate-my-wall.png" alt="Curate my wall: three Hokusai prints hung on the wall, and the fashion they lead to: printed scarves, a tiger-print tee and a kimono"></a></td>
    <td width="33%"><a href="https://github.com/Unbox-AI/behaviorgpt-anomaly"><img src="https://raw.githubusercontent.com/Unbox-AI/behaviorgpt/main/docs/images/anomaly-detection.png" alt="Anomaly detection: an account takeover after headphone views, flagged at step 8 when a carted lawn mower pushes the session's risk past the barrier"></a></td>
  </tr>
  <tr>
    <td><b>Storefront</b> (<a href="https://behaviorgpt.unboxai.com/">live demo</a>). Search results and recommendations change with what the shopper did before, and a panel shows the history the model read.</td>
    <td><b>Curate my wall</b> (<a href="https://aic-artworks.streamlit.app/">live app</a>, <a href="https://github.com/Unbox-AI/aic-artworks">source</a>). Hang three Hokusai prints, and the "Fashion for you" tab suggests scarves and kimonos from those art clicks alone.</td>
    <td><b>Anomaly detection</b> (<a href="https://github.com/Unbox-AI/behaviorgpt-anomaly">source</a>). An account takeover: after the owner's headphone views, someone carts a phone and then a lawn mower, and the session is flagged at the mower, the step where its risk crosses the barrier.</td>
  </tr>
</table>

## Requirements

- An Unbox AI API key. Create and copy the key at [unboxai.com/behaviorgpt](https://unboxai.com/behaviorgpt); the key also arrives by email.
- Python 3.11 or newer.

## Quick start

```sh
pip install behaviorgpt
export UNBOXAI_API_KEY=...   # or pass api_key= to UnboxAIClient
```

```python
from behaviorgpt import Search, UnboxAIClient, View

client = UnboxAIClient(market="us")

# a history of one search
res = client.complete([Search("running shoes")], limit=5)
print(res.names)

# the same call, personalized: a viewed item (Nike pants), then a query
res = client.complete([View("paB08NYK61PJ"), Search("shoes")], limit=5)
print(res.names)

# items close to one another in the model's space
print(client.similar_items(res.items[0].id).names)
```

`complete` takes the history and returns the items most likely to come next. Each result in `res.items` has an `id`, a `score` and the catalog row in `data`; `res.to_pandas()` gives a DataFrame if pandas is installed.

### Async

In asyncio code, such as a FastAPI or aiohttp server, use `AsyncUnboxAIClient`. It takes the same arguments and has the same methods, awaited, so a call never blocks the event loop and many calls can run at once:

```python
import asyncio

from behaviorgpt import AsyncUnboxAIClient, Search

async def main():
    async with AsyncUnboxAIClient(market="us") as client:
        shoes, gloves = await asyncio.gather(
            client.complete([Search("running shoes")], limit=5),
            client.complete([Search("gloves")], limit=5),
        )
        print(shoes.names, gloves.names)

asyncio.run(main())
```

Create one client and reuse it, for example for the lifetime of your server: `async with` closes its connections at the end, or call `await client.aclose()`. `embed(..., wait=True)` waits for the job without blocking other requests.

## How it works

- **One call, many features.** Every request asks the same question: given what this person did, which items are they most likely to act on next? The history picks the feature. End it with a `Search` for search results, with a `View` for "more like this", with an `Order` for what is bought next; leave it empty for a cold start.
- **The whole sequence counts.** The model reads the history in order, so the same query means different things after different actions: "gloves" after "golf clubs" returns golf gloves. `market` on the client says where the user is, which helps most when the history is short.
- **Items are read from their content, not their ids.** Each item is embedded from its name, brand, categories, image and price. That is why the model can rank a catalog it has never seen as soon as it is uploaded, with no interaction data and no training, and why those columns decide how well it does ([catalog format](https://github.com/Unbox-AI/behaviorgpt/blob/main/docs/catalog-format.md)).
- **One pass per request.** Item vectors are computed once when the catalog is embedded, so ranking the whole catalog costs one forward pass over the history rather than one per item: the model itself takes under a millisecond per query, and most of an API call's time is the network.

## The notebook

[`notebooks/showcase.ipynb`](https://github.com/Unbox-AI/behaviorgpt/blob/main/notebooks/showcase.ipynb) covers the model in eight steps: setup, a first query, one call for many features, how context changes the answer, then embedding your own catalog, inspecting the embedding space, querying it, and trying it in the demo.

Needs [uv](https://github.com/astral-sh/uv). Clone, install, add your key, register the kernel, start JupyterLab:

```sh
git clone https://github.com/Unbox-AI/behaviorgpt.git
cd behaviorgpt
uv sync
cp .env.example .env   # paste your key as UNBOXAI_API_KEY
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=behaviorgpt-env
uv run jupyter lab
```

JupyterLab opens in your browser. Open the notebook and pick the `behaviorgpt-env` kernel.

## Your own catalog

The model can only return items it knows about. Out of the box the client uses `retail_catalog`, a sample retail catalog. To use the model on your own data, upload your catalog as a parquet file. A catalog doesn't have to be products for sale: artworks, articles, listings or anything else people browse and pick from work the same way.

- [`docs/catalog-format.md`](https://github.com/Unbox-AI/behaviorgpt/blob/main/docs/catalog-format.md) lists the required columns and types.
- A catalog can hold at most 100,000 items.
- Each API key can hold 2 catalogs. The file name is the catalog's name: uploading a file with the same name again refreshes that catalog and keeps its `catalog_id`, and a new name takes one of the 2 slots. A failed upload does not use up a slot. Once both slots are in use, an upload under a new name is refused until you delete one of them (see below); to raise the limit, contact alexander@unboxai.com.
- [`examples/sample_catalog.csv`](https://github.com/Unbox-AI/behaviorgpt/blob/main/examples/sample_catalog.csv) is an example only, showing what each column should contain. Your own file will look different depending on your catalog. A CSV with the documented column names converts to the parquet structure with `catalog_from_csv`.

```python
from behaviorgpt import catalog_from_csv

path = catalog_from_csv("my_catalog.csv")  # writes my_catalog.parquet
job = client.embed(path, wait=True, timeout=1800.0)
client.complete([Search("...")], catalog_id=job.catalog_id)
```

With `wait=True`, a progress bar shows each step of the job and the time left; where the output is not a terminal or a notebook, each new state is printed once instead. Pass `on_progress=lambda _: None` for silence.

### Adding or updating items

To add items to an embedded catalog, or change some of its items, upload just those items in the same format. Only they are embedded, so the job takes about as long as the file is big, and the catalog keeps answering queries while it runs:

```python
client.add_items("new_items.parquet", catalog_id=job.catalog_id, wait=True)
```

- An item whose `id` is already in the catalog is replaced with the new row; any other item is added. Nothing is removed.
- The new items are embedded at the catalog's own reference time, so their ages and sales are measured the same way as the rest of the catalog's.
- The 100,000-item limit counts the whole catalog: the items it holds plus the new ones.
- One `add_items` job runs per catalog at a time, and the catalog cannot be refreshed or deleted while it runs.
- Uploading the catalog's file again with `embed` replaces the whole catalog, added items included. Put them in that file too to keep them.
- A catalog embedded before `add_items` existed needs one more full upload before items can be added to it.

### Listing and deleting catalogs

```python
for c in client.list_catalogs():
    print(c.catalog_id, c.catalog_name, c.status, c.owned)

client.delete_catalog("cat_...")
```

- `list_catalogs()` returns your own catalogs, in any status, and the shared ones that are ready. `owned` marks your own.
- `delete_catalog()` removes one of your catalogs with its items and frees its slot. It cannot be undone.
- Deleting another key's catalog fails with status 404, the same as an unknown id. While an `embed`, `add_items` or `add_keywords` job runs for the catalog, deleting fails with status 409; try again once the job is ready or failed.

### Keywords and sections

Sections group the items the model ranks under titles such as "black coats", best section first. They are built from item keywords, which `embed` does not generate by default because they take time. Generate them for an embedded catalog with `add_keywords`:

```python
job = client.embed(path, wait=True, timeout=1800.0)
client.add_keywords(job.catalog_id, wait=True)
response = client.sections([Search("coat")], catalog_id=job.catalog_id)
for section in response.sections:
    print(section.title, [item.data["name"] for item in section.items])
```

Or generate them during the upload, so the catalog goes live with them:

```python
job = client.embed(path, keywords=True, wait=True, timeout=4 * 3600.0)
```

- Keyword generation takes roughly an hour per 25,000 items. `add_keywords` waits up to 4 hours by default; with `embed(..., keywords=True)`, allow a longer `timeout`.
- `add_keywords` re-embeds nothing, and the catalog keeps answering queries while it runs. The keywords appear when the job is ready. If it fails, the catalog stays as it was.
- With `embed(..., keywords=True)`, if keyword generation fails, the catalog still goes live, without keywords. The job's status then reads `keywords_skipped`, with the reason in `keywords_error`, and `sections` returns an empty list for that catalog.
- Uploading the catalog again with `embed` replaces its keywords. Pass `keywords=True` there, or call `add_keywords` again, to keep them.
- A history that ends in a search groups that search's results; any other history groups recommendations.
- Items added or replaced with `add_items` have no keywords until you call `add_keywords` again or upload the catalog with `embed(..., keywords=True)`; until then they are left out of sections.
- One `add_items` or `add_keywords` job runs per catalog at a time, and the catalog cannot be refreshed or deleted while it runs.

If mapping your data to the format is hard, open a coding agent in this directory and ask it to embed your catalog. The repo ships a skill at [`.claude/skills/embed-catalog/SKILL.md`](https://github.com/Unbox-AI/behaviorgpt/blob/main/.claude/skills/embed-catalog/SKILL.md) that walks the agent through mapping your columns, building and validating the parquet, checking image URLs, and uploading. Claude Code picks it up automatically; for other agents, point them at that file.

### Events

A history is a list of `Search`, `View`, `AddToCart`, `RemoveFromCart` and `Order` events. The names come from a store because that is what the API accepts, but they work for any catalog: map your users' actions by meaning. Curate my wall sends "look closer" as `View`, "hang it on the wall" as `AddToCart` and "take it down" as `RemoveFromCart`.

## Demo and examples

More on the three apps shown at the top of this page:

- [behaviorgpt.unboxai.com](https://behaviorgpt.unboxai.com/): a storefront, with recommendations, personalized search and a cart driven by the shopper's clicks. To try it on your catalog once it is embedded, select the **BehaviorGPT V4.0-12.5B** model, choose **Bring your own catalog** and enter your API key. Images must be publicly reachable for the demo to display them.
- [Curate my wall](https://github.com/Unbox-AI/aic-artworks) ([live app](https://aic-artworks.streamlit.app/)): an art recommender over 19,000 public-domain artworks from the Art Institute of Chicago. It shows the whole path for a catalog that isn't products: turning an open dataset into a catalog, embedding it, and driving recommendations, search, similar items and the embedding map from a visitor's clicks.
  - **Art to fashion.** Its fashion pilot puts H&M clothing in the same catalog as the art, and the "Fashion for you" tab ranks it from the visitor's art clicks alone: kimonos for Japanese prints, beaded bracelets for ancient Egypt, statement earrings for portraits. No event in the history involves clothing. It works because the art and the clothing share one catalog; a `filters` query keeps the art grids art-only.
- [Anomaly detection](https://github.com/Unbox-AI/behaviorgpt-anomaly): scores every step of a shopping session by how likely the model finds it given the history so far, and flags the step where a session crosses a calibrated risk barrier, naming the items that pushed it there. It needs no fraud labels, rules or training. It runs on the public `retail_catalog`, and its bundled examples work without an API key.
