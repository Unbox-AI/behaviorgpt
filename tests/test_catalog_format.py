"""The catalog format: docs/catalog-format.md, CATALOG_SCHEMA and
check_catalog_schema must say the same thing."""

import re

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from behaviorgpt import catalog_from_csv
from behaviorgpt._exceptions import UnboxAIError
from behaviorgpt.resources.catalogs import CATALOG_SCHEMA, check_catalog_schema

from .conftest import ROOT, SAMPLE_CSV, catalog_table

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
        ("brand", pa.string_view()),
        ("name", pa.dictionary(pa.int32(), pa.string())),  # pandas category
        ("sales_since", pa.large_list(pa.int64())),
        ("keywords", pa.large_list(pa.large_string())),
    ],
)
def test_wide_layouts_of_the_right_type_pass(write_catalog, column, cast):
    """The embedder casts these to the plain layout before it checks types."""
    t = catalog_table()
    i = t.schema.get_field_index(column)
    t = t.set_column(i, column, t.column(i).cast(cast))
    check_catalog_schema(write_catalog(t))


def test_a_catalog_written_by_pandas_passes(tmp_path):
    path = tmp_path / "catalog.parquet"
    pd.DataFrame(catalog_table().to_pylist()).to_parquet(path, index=False)
    check_catalog_schema(path)


@pytest.mark.parametrize(
    "column,cast",
    [
        ("timestamp", pa.int32()),
        ("sales_since", pa.large_list(pa.int32())),
        ("price", pa.float64()),
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


# ---- catalog_from_csv -------------------------------------------------------


def from_csv(tmp_path, text: str) -> pa.Table:
    path = tmp_path / "catalog.csv"
    path.write_text(text)
    return pq.read_table(catalog_from_csv(path))


def test_sample_csv_converts_to_the_schema(tmp_path):
    out = catalog_from_csv(SAMPLE_CSV, tmp_path / "sample_catalog.parquet")
    table = pq.read_table(out)
    assert table.schema == pa.schema(CATALOG_SCHEMA)
    assert table.num_rows == 100
    assert all(len(s) == 12 for s in table.column("sales_since").to_pylist())
    assert table.column("price").to_pylist()[0] == "175"


def test_parquet_goes_next_to_the_csv_by_default(tmp_path):
    csv = tmp_path / "my_catalog.csv"
    csv.write_text("id,name,event_type,group\na,A,product,product\n")
    assert catalog_from_csv(csv) == tmp_path / "my_catalog.parquet"


def test_cells_are_text_and_lists_are_parsed(tmp_path):
    table = from_csv(
        tmp_path,
        "id,name,price,sales_since,keywords,timestamp\n"
        '007,Bond,299.00,"[1, 2, 3]","red, blue",1700000000\n',
    )
    row = table.to_pylist()[0]
    assert row["id"] == "007" and row["price"] == "299.00"
    assert row["sales_since"] == [1, 2, 3]
    assert row["keywords"] == ["red", "blue"]
    assert row["timestamp"] == 1_700_000_000


def test_empty_cells_and_missing_columns_are_null(tmp_path):
    table = from_csv(tmp_path, "id,name,brand,keywords,color\na,A,,,red\n")
    row = table.to_pylist()[0]
    assert row["brand"] is None and row["keywords"] is None
    assert row["frequency"] is None and row["image_url"] is None
    assert "color" not in table.column_names


def test_duplicate_ids_keep_the_first_row(tmp_path):
    table = from_csv(tmp_path, "id,name\na,first\nb,other\na,second\n")
    assert table.column("name").to_pylist() == ["first", "other"]


@pytest.mark.parametrize(
    "column,cell",
    [("timestamp", "yesterday"), ("sales_since", "[1, two]"), ("frequency", "1.5")],
)
def test_unreadable_values_name_their_column(tmp_path, column, cell):
    with pytest.raises(UnboxAIError, match=column):
        from_csv(tmp_path, f'id,name,{column}\na,A,"{cell}"\n')
