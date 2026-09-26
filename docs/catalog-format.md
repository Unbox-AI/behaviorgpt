# Catalog format

`client.embed(path)` uploads one parquet file. The server places every row in the model's product space. This page describes what the file must contain and how to build it from a CSV.

```python
job = client.embed("my_catalog.parquet", wait=True, timeout=1800.0)
```

## File

- Only parquet files are accepted.
- All 15 columns below must be present. The server checks column names only. A missing column fails the job before any work starts. A column that exists but is entirely null passes.
- The client checks column types before upload, since the tokenizer depends on them. A wrong type fails the job before any work starts.

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

`examples/sample_catalog.csv` has 100 rows with the correct column names and realistic values. It exists to be read: open it to see what each column should contain. It cannot be uploaded as-is, because CSV carries no types. Read naively, three places come out wrong:

| Column | In the CSV | Parquet requires |
|---|---|---|
| `price` | number, `175` | string, `"175"` |
| `sales_since` | one text field, `"[61, 78, ...]"` | list of 12 int64 |
| `market`, `search_keywords`, `keywords` | empty | string or list of string; all-null is accepted |

Every other column is already a string or an integer and passes through unchanged.

## From CSV to parquet

Recipe for turning a CSV with the column names above into an accepted parquet. Written so a coding agent can follow it directly.

1. Read every column as text. Do not let the CSV reader guess types.
2. Parse list columns into real lists: `sales_since` to integers, `search_keywords` and `keywords` to strings. Empty cells become null. A string such as `"['a', 'b']"` is not a list.
3. Cast `timestamp` and `frequency` to int64. Leave `price` as a string.
4. Drop duplicate `id` rows. The server silently keeps only the first.
5. Build the table with the explicit schema shipped in the package, so types are forced rather than inferred, and run the package's own check before uploading.

```python
import ast
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from behaviorgpt.resources.catalogs import CATALOG_SCHEMA, check_catalog_schema

CSV_PATH = "examples/sample_catalog.csv"
CATALOG_PATH = "examples/sample_catalog.parquet"

df = pd.read_csv(CSV_PATH, dtype=str)  # step 1

def to_list(cell, cast):
    """'[1, 2]' or 'a, b' -> list; empty -> None."""
    if pd.isna(cell) or not cell.strip():
        return None
    items = ast.literal_eval(cell) if cell.strip().startswith("[") else cell.split(",")
    return [cast(str(x).strip()) for x in items]

df["sales_since"] = df["sales_since"].map(lambda c: to_list(c, int))       # step 2
df["search_keywords"] = df["search_keywords"].map(lambda c: to_list(c, str))
df["keywords"] = df["keywords"].map(lambda c: to_list(c, str))
df["timestamp"] = df["timestamp"].astype("Int64")                          # step 3
df["frequency"] = df["frequency"].astype("Int64")
df = df.drop_duplicates("id")                                              # step 4

table = pa.Table.from_pandas(df, schema=pa.schema(CATALOG_SCHEMA), preserve_index=False)
pq.write_table(table, CATALOG_PATH)                                        # step 5
check_catalog_schema(CATALOG_PATH)
```

If the source uses other column names, rename them to the names in the table first. Add any column the source lacks as all-null. Set `event_type` and `group` to `"product"` on every row.
