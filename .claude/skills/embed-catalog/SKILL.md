---
name: embed-catalog
description: Convert a user's product catalog (CSV, TSV, JSON, Excel, Shopify/WooCommerce export, Google Merchant feed, database dump) into the BehaviorGPT catalog parquet and embed it with the UnboxAI API. Use when the user wants to bring, upload, embed, or use their own catalog or products with BehaviorGPT, or asks to build a catalog parquet.
---

# Embed your own catalog

Goal: take whatever product data the user has, produce one parquet that passes `check_catalog_schema`, upload it with `client.embed`, and hand back the `catalog_id`.

`docs/catalog-format.md` is the source of truth for columns, types and failure modes. Read it before starting. `examples/sample_catalog.csv` shows realistic values for every column. `src/behaviorgpt/resources/catalogs.py` holds `CATALOG_SCHEMA` and `check_catalog_schema`.

## Environment

- Run Python through the project env: `uv sync` once, then `uv run python <script>`. pandas is in the dev group, which `uv sync` installs.
- The API key comes from `UNBOXAI_API_KEY`, set in `.env` (copied from `.env.example`). Never print, log, or echo the key. If it is missing, ask the user to add it; do not ask them to paste it in chat.
- Put conversion scripts and the output parquet in `.untracked/` (gitignored). Never commit the user's catalog.

## Workflow

### 1. Inspect the source

Load a sample (first ~20 rows) and report: row count, column names, a few example values per column, and which columns look like lists, prices, dates or image links. Do not guess types from the file extension alone.

### 2. Propose a column mapping and confirm it

Map source columns to the 15 target columns. Show the mapping as a table and wait for the user to confirm before converting. Flag every target column that will be all-null.

Common sources:

| Target | Shopify export | WooCommerce export | Google Merchant feed |
|---|---|---|---|
| `id` | `Handle` | `ID` or `SKU` | `id` |
| `name` | `Title` | `Name` | `title` |
| `brand` | `Vendor` | brand attribute | `brand` |
| `categories` | `Product Category` or `Type` | `Categories` (`>` separators become `, `) | `product_type` or `google_product_category` (`>` separators become `, `) |
| `image_url` | `Image Src` | first URL in `Images` | `image_link` |
| `price` | `Variant Price` | `Regular price` | number part of `price` |
| `currency` | ask the user | ask the user | currency part of `price`, e.g. `299.00 SEK` |

Source-specific traps:

- Shopify exports one row per variant and per extra image. Group by `Handle` and keep the first row that has a `Title`.
- Variant-level catalogs: decide with the user whether a product is the parent or each variant. Default to the parent; variants mostly differ by size and add near-duplicate vectors.
- `categories` is one comma-separated string, general to specific, e.g. `Clothing, Trousers`.
- Set `event_type` and `group` to `"product"` on every row.

### 3. Fill the popularity columns when order data exists

Ask whether the user has order or sales history. Without it, leave `sales_since`, `timestamp`, `frequency` null and say that every product will rank as equally popular.

With order lines (product id + date + quantity), compute per product, relative to today:

- `sales_since`: 12 int64 values. Units sold in the trailing 30, 60, 90, ..., 330 days (11 cumulative windows, so values never decrease), then the all-time total.
- `timestamp`: epoch seconds of the first sale.
- `frequency`: total interactions. Use the all-time total if nothing better exists.

### 4. Enforce the limits

- At most 20 000 products. If the source is larger, ask the user how to select, and suggest the top 20 000 by `frequency` or recent sales.
- Unique `id`. Report how many duplicates were dropped; the server would silently keep the first.
- `id` and `name` must be non-null. Drop rows missing either and report the count.

### 5. Build and validate the parquet

Follow the "From CSV to parquet" recipe in `docs/catalog-format.md`: read everything as text, parse list columns into real lists, cast `timestamp` and `frequency` to int64, keep `price` as a string, build the table with `pa.schema(CATALOG_SCHEMA)`, write it, then run `check_catalog_schema(path)`. Fix and rerun until it passes.

### 6. Check images before uploading

`image_url` carries most of the signal, and the job fails if none of the first 500 fetched images is usable. Fetch a random sample of ~20 URLs with a plain HTTPS GET, no cookies, no browser headers, and report the success rate and status codes per host.

- Relative paths: prefix the store's domain.
- `gs://`, `s3://`, login-gated or bot-protected hosts (403, 429, challenge pages): these will not be fetched. Tell the user they need a public image host or CDN URL.
- Protocol-relative `//cdn...`: prefix `https:`.

### 7. Upload, only after explicit confirmation

Uploading sends the catalog to UnboxAI. State the file path, row count, and image success rate, and ask before running:

```python
from behaviorgpt import Search, UnboxAIClient

client = UnboxAIClient(market="SE")  # market code of the user's store
job = client.embed(".untracked/my_catalog.parquet", wait=True, timeout=1800.0)
print(job.catalog_id)
```

Load `.env` first (`from dotenv import load_dotenv; load_dotenv()`) if the key is not already in the environment. Embedding takes several minutes; run it in the background if the tool allows.

On failure, `UnboxAIError` carries the server's reason. `client.job_status(job.job_id).error` holds the same text. Map it back to a fix from `docs/catalog-format.md`, most often the image host.

### 8. Smoke test and hand over

Run one query and one similarity lookup against the new catalog, and show the top names:

```python
res = client.complete(history=[Search("<something the store sells>")], catalog_id=job.catalog_id, limit=5)
print(res.names)

pick = client.random_product(job.catalog_id)
print(pick.data["name"], "->", client.similar_products(pick.id, catalog_id=job.catalog_id).names[:5])
```

Finish by giving the user the `catalog_id` (private to their API key), and point them to steps 6 to 8 of `notebooks/showcase.ipynb` and the demo at https://behaviorgpt.unboxai.com/ ("Bring your own catalog").
