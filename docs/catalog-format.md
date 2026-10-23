# Catalog format

`client.embed(path)` uploads one parquet file. The server places every row in the model's item space. This page describes what the file must contain and how to build it from a CSV.

An item doesn't have to be a product for sale. The columns are named for a store, but they map onto other catalogs too. [Curate my wall](https://github.com/Unbox-AI/aic-artworks) embeds artworks with the title as `name`, the artist as `brand`, department, medium and style as `categories`, and the museum's view counts standing in for sales.

```python
job = client.embed("my_catalog.parquet", wait=True, timeout=1800.0)
```

Item ages (from `timestamp`) and sales (from `sales_since`) are measured at the upload time. For a historical dataset, pass the moment to measure them at:

```python
from datetime import UTC, datetime

job = client.embed(
    "my_catalog.parquet", wait=True, reference_time=datetime(2020, 7, 15, tzinfo=UTC)
)
```

## File

- Only parquet files are accepted.
- All 15 columns below must be present. The server checks column names only. A missing column fails the job before any work starts. A column that exists but is entirely null passes.
- The client checks column types before upload, since the tokenizer depends on them. A wrong type fails the job before any work starts.
- If none of the first 500 fetched images can be used, the job fails instead of embedding text only. This usually means the image host blocks automated requests, for example a site behind bot protection. The error names the host, e.g. `Embed job … failed: no usable image in 500 of 18974 urls fetched: {'unreachable (http 403)': 500}; failing hosts: www.example.com (500)`. Serve the images from a host that allows plain HTTPS fetches.
- A failed job raises `UnboxAIError`, whose message carries the reason when the server gives one. `client.job_status(job_id).error` holds the same text.

## Columns

| Column | Arrow type | Nullable | Role |
|---|---|---|---|
| `id` | string | no | Item key. Must be unique. Duplicate ids collapse to the first row. |
| `name` | string | no | Item title. Primary text input to the embedding. |
| `brand` | string | yes | Appended to the embedding text. |
| `categories` | string | yes | Comma-separated path, e.g. `Clothing, Trousers`. Appended to the embedding text and split into keywords. |
| `image_url` | string | yes | Full `https://` URL, fetchable without a login or browser. Fetched and encoded into the embedding. Null or unreachable means text-only embedding for that row, but if none of the first 500 fetched images is usable the job fails (see below). |
| `event_type` | string | no | Set every row to `product`, whatever the catalog holds. |
| `group` | string | no | Set every row to `product`, whatever the catalog holds. |
| `sales_since` | list of int64 | yes | 12 integers: sales in the trailing 30, 60, ..., 330 days, then the all-time total. Popularity signal. Null is treated as all zeros. |
| `timestamp` | int64 | yes | Epoch seconds of the first sale. Recency signal. |
| `frequency` | int64 | yes | Total interactions with the item. Popularity signal. |
| `market` | string | yes | Market code, e.g. `SE`. |
| `price` | string | yes | Display only. Keep as string, e.g. `"299.00"`. |
| `currency` | string | yes | Display only, e.g. `SEK`. |
| `search_keywords` | list of string | yes | Extra search terms. Augmentation only. |
| `keywords` | list of string | yes | Extra tags. Augmentation only. |

The large, view and dictionary layouts of these types are accepted too, so a parquet written by pandas 3 or polars (`large_string`, `large_list`, `string_view`, pandas `category`) needs no cast.

## Minimum viable catalog

`id`, `name`, `event_type = "product"`, `group = "product"`. Every other column may be null. The job completes and the embedding is built from the title alone.

## What raises embedding quality

Listed in order of impact.

1. `image_url`. For visual categories this carries most of the signal. The URL must be a complete `https://` address that returns the image without authentication. Relative paths, `gs://`, `s3://`, and login-gated URLs are never fetched.
2. `categories` and `brand`. They join the title in the embedding text.
3. `sales_since`, `timestamp`, `frequency`. These drive popularity and recency. Without them all items rank as equally popular.

## Catalog size and the UMAP

`client.umap()` returns a 2D map of the catalog's items. The map is built only for catalogs of up to 100,000 items, within about a minute after the catalog reads `ready`. Laying out more items than that does not fit in the embedding worker's memory, so larger catalogs, such as the shared `retail_catalog`, never get one. Embedding and every query work the same at any size; only the map is skipped. For a catalog without a map, `umap()` raises `UnboxAIError` with status 404. Right after an embed, pass `wait=` (seconds) to keep asking until the map is built, for example `client.umap(catalog_id=job.catalog_id, wait=180)`. Items added with `client.add_items()` are not drawn on the map until the next full upload of the catalog.

## Hosting your catalog

The parquet itself is uploaded, but images are fetched from their `image_url` by UnboxAI's servers. Every URL must therefore be public, `https://`, and reachable from a server, not only from a browser. Some sites put images behind bot protection (Cloudflare, for example) that serves browsers but blocks servers; the job then fails when all image fetches come back 403. Test a few URLs with `curl` from a cloud machine if in doubt.

If your images are not reachable that way, re-host copies on any public host: an S3, R2, GCS or Azure bucket with public read, a GitHub repo, or a Hugging Face dataset. Only re-host images you have the rights to publish.

To use Hugging Face with your own account:

1. Create a write token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and add it to `.env` as `HF_TOKEN=...`. Never commit it.
2. Upload your image folder as a public dataset:

   ```python
   from dotenv import load_dotenv
   from huggingface_hub import HfApi

   load_dotenv()
   api = HfApi()
   api.create_repo("you/my-catalog-images", repo_type="dataset", exist_ok=True)
   api.upload_folder(
       repo_id="you/my-catalog-images",
       repo_type="dataset",
       folder_path="images",
       path_in_repo="images",
   )
   ```

   A folder on Hugging Face holds at most 10,000 files, so split larger sets into subfolders.
3. Point `image_url` at `https://huggingface.co/datasets/you/my-catalog-images/resolve/main/images/<file>`.

Embed a small slice first (a few hundred rows) to confirm the images are fetched before uploading the full catalog.

## Example catalog

`examples/sample_catalog.csv` has 100 rows with the correct column names and realistic values. It exists to be read: open it to see what each column should contain. It cannot be uploaded as-is, because CSV carries no types; convert it with `catalog_from_csv` (below). Read naively, three places come out wrong:

| Column | In the CSV | Parquet requires |
|---|---|---|
| `price` | number, `175` | string, `"175"` |
| `sales_since` | one text field, `"[61, 78, ...]"` | list of 12 int64 |
| `market`, `search_keywords`, `keywords` | empty | string or list of string; all-null is accepted |

Every other column is already a string or an integer and passes through unchanged.

## From CSV to parquet

`catalog_from_csv` turns a CSV with the column names above into an accepted parquet, checks it and returns its path. The parquet goes next to the CSV unless you pass a second path, and its file name becomes the catalog's name.

```python
from behaviorgpt import catalog_from_csv

path = catalog_from_csv("examples/sample_catalog.csv")  # examples/sample_catalog.parquet
job = client.embed(path, wait=True, timeout=1800.0)
```

It does what the format needs, so the result does not depend on how a CSV reader guesses types:

1. Every cell is read as text, so `price` and ids keep their formatting.
2. List columns become real lists: `sales_since` of integers, `search_keywords` and `keywords` of strings. A cell may be `[1, 2]` or `a, b`.
3. `timestamp` and `frequency` become int64. A value that is not a whole number raises `UnboxAIError` naming the column.
4. Empty cells, and columns the CSV lacks, become null. Columns not in the table above are left out.
5. Duplicate `id` rows keep the first, as the server would.

If the source uses other column names, rename them to the names in the table first, and set `event_type` and `group` to `"product"` on every row. For a source that is not a CSV, build the table with `pa.schema(CATALOG_SCHEMA)` from `behaviorgpt.resources.catalogs` so types are forced rather than inferred, write it with `pyarrow.parquet.write_table`, and run `check_catalog_schema(path)` before uploading.
