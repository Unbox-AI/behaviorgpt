# Catalog format

`client.embed(path)` uploads one parquet file. The server places every row in the model's product space. This page describes what the file must contain.

```python
job = client.embed("my_catalog.parquet", wait=True, timeout=1800.0)
```

## File

- Only parquet files are allowed. (as of now)
- All 15 columns below must be present. The check is on column names only. A missing column fails the job before any work starts. A column that exists but is entirely null passes.
- Types are checked at upload time since the tokenizer depends on them. Wrong types fails the job before any work starts.

## Columns

| Column | Arrow type | Nullable | Role |
|---|---|---|---|
| `id` | string | no | Product key. Must be unique. Duplicate ids collapse to the first row. |
| `name` | string | no | Product title. Primary text input to the embedding. |
| `brand` | string | yes | Appended to the embedding text. |
| `categories` | string | yes | Comma-separated path, e.g. `Clothing, Trousers`. Appended to the embedding text and split into keywords. |
| `image_url` | string | yes | Full `https://` URL. Fetched and encoded into the embedding. Null or unreachable means text-only embedding for that row. |
| `event_type` | string | no | Set every row to `product`. |
| `group` | string | no | Set every row to `product`. |
| `sales_since` | list of int64 | yes | 12 integers: sales in the trailing 30, 60, ..., 330 days, then the all-time total. Popularity signal. Null is treated as all zeros. |
| `timestamp` | int64 | yes | Epoch seconds of the first sale. Recency signal. |
| `frequency` | int64 | yes | Total interactions with the product. Popularity signal. |
| `market` | string | yes | Market code, e.g. `SE`. |
| `price` | string | yes | Display only. Keep as string, e.g. `"299.00"`. |
| `currency` | string | yes | Display only, e.g. `SEK`. |
| `search_keywords` | list of string | yes | Extra search terms. Augmentation only. |
| `keywords` | list of string | yes | Extra tags. Augmentation only. |

## Minimum viable catalog

`id`, `name`, `event_type = "product"`, `group = "product"`. Every other column may be null. The job completes and the embedding is built from the title alone.

## What raises embedding quality

Listed in order of impact.

1. `image_url`. For visual categories this carries most of the signal. The URL must be a complete `https://` address that returns the image without authentication. Relative paths, `gs://`, `s3://`, and login-gated URLs are never fetched.
2. `categories` and `brand`. They join the title in the embedding text.
3. `sales_since`, `timestamp`, `frequency`. These drive popularity and recency. Without them all products rank as equally popular.

## Rules the upload check does not enforce

- One row per `id`. Duplicates are silently dropped after the first.
- `frequency`, `timestamp`, and every element of `sales_since` must be integers, not floats or strings.
- `price` must be a string, not a number.
- List columns must be real parquet lists. A string such as `"['a', 'b']"` is not a list.

## Example

`examples/sample_catalog.parquet` contains 100 rows in this exact schema. Inspect it with:

```python
import pyarrow.parquet as pq

t = pq.read_table("examples/sample_catalog.parquet")
print(t.schema)
print(t.slice(0, 3).to_pylist())
```
