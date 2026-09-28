from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from behaviorgpt.resources.catalogs import CATALOG_SCHEMA

ROOT = Path(__file__).resolve().parent.parent


def catalog_table(rows: int = 3) -> pa.Table:
    """A small catalog in exactly the documented schema."""
    data = {
        "id": [f"sku-{i}" for i in range(rows)],
        "name": [f"Product {i}" for i in range(rows)],
        "brand": ["Brand"] * rows,
        "categories": ["Clothing, Trousers"] * rows,
        "image_url": [f"https://images.example.com/{i}.jpg" for i in range(rows)],
        "event_type": ["product"] * rows,
        "group": ["product"] * rows,
        "sales_since": [[1] * 12] * rows,
        "timestamp": [1_700_000_000] * rows,
        "frequency": [12] * rows,
        "market": ["SE"] * rows,
        "price": ["299.00"] * rows,
        "currency": ["SEK"] * rows,
        "search_keywords": [["trousers"]] * rows,
        "keywords": [["b1"]] * rows,
    }
    return pa.Table.from_pydict(data, schema=pa.schema(CATALOG_SCHEMA))


@pytest.fixture
def write_catalog(tmp_path):
    def write(table: pa.Table, name: str = "catalog.parquet") -> Path:
        path = tmp_path / name
        pq.write_table(table, path)
        return path

    return write


@pytest.fixture
def catalog_path(write_catalog) -> Path:
    return write_catalog(catalog_table())
