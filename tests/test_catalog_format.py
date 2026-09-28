"""The catalog format: docs/catalog-format.md, CATALOG_SCHEMA and
check_catalog_schema must say the same thing."""

import re

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from behaviorgpt._exceptions import UnboxAIError
from behaviorgpt.resources.catalogs import CATALOG_SCHEMA, check_catalog_schema

from .conftest import ROOT, catalog_table

DOC = ROOT / "docs" / "catalog-format.md"


def _doc_type(t: pa.DataType) -> str:
    return f"list of {t.value_type}" if pa.types.is_list(t) else str(t)


def test_doc_column_table_matches_schema():
    rows = re.findall(
        r"^\|\s*`([a-z_]+)`\s*\|\s*([a-z0-9 ]+?)\s*\|", DOC.read_text(), re.M
    )
    assert dict(rows) == {name: _doc_type(t) for name, t in CATALOG_SCHEMA.items()}


def test_valid_catalog_passes(catalog_path):
    check_catalog_schema(catalog_path)


def test_extra_columns_and_order_are_fine(write_catalog):
    t = catalog_table()
    t = t.select(list(reversed(t.column_names))).append_column(
        "color", pa.array(["r"] * 3)
    )
    check_catalog_schema(write_catalog(t))


def test_all_null_column_passes(write_catalog):
    t = catalog_table()
    i = t.schema.get_field_index("brand")
    check_catalog_schema(write_catalog(t.set_column(i, "brand", pa.nulls(3))))


def test_every_problem_is_reported_at_once(write_catalog):
    t = catalog_table().drop_columns(["frequency", "keywords"])
    i = t.schema.get_field_index("timestamp")
    t = t.set_column(i, "timestamp", pc.cast(t.column(i), pa.string()))
    with pytest.raises(UnboxAIError) as exc:
        check_catalog_schema(write_catalog(t))
    msg = str(exc.value)
    assert "frequency: missing" in msg
    assert "keywords: missing" in msg
    assert "timestamp: expected int64, got string" in msg


@pytest.mark.parametrize(
    "column,cast",
    [
        ("id", pa.large_string()),  # pa.Table.from_pandas(df) under pandas 3
        ("name", pa.dictionary(pa.int32(), pa.string())),  # pandas category
        ("timestamp", pa.int32()),
        ("sales_since", pa.large_list(pa.int64())),
    ],
)
def test_near_miss_types_are_rejected(write_catalog, column, cast):
    """The embedding pipeline needs the exact types, so the client refuses
    near misses before upload rather than letting the job fail later."""
    t = catalog_table()
    i = t.schema.get_field_index(column)
    t = t.set_column(i, column, t.column(i).cast(cast))
    with pytest.raises(UnboxAIError, match=column):
        check_catalog_schema(write_catalog(t))


def test_non_parquet_is_an_sdk_error(tmp_path):
    path = tmp_path / "catalog.parquet"
    path.write_text('{"id": "x"}\n')
    with pytest.raises(UnboxAIError, match="not a readable parquet file"):
        check_catalog_schema(path)


def test_missing_file_is_an_sdk_error(tmp_path):
    with pytest.raises(UnboxAIError, match="not a readable parquet file"):
        check_catalog_schema(tmp_path / "nope.parquet")
