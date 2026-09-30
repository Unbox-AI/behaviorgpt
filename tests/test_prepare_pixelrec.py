"""scripts/prepare_pixelrec.py on a handful of synthetic PixelRec rows."""

import csv
import importlib.util
from datetime import date

import pyarrow.parquet as pq

from behaviorgpt.resources.catalogs import check_catalog_schema

from .conftest import ROOT

_spec = importlib.util.spec_from_file_location(
    "prepare_pixelrec", ROOT / "scripts" / "prepare_pixelrec.py"
)
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)

DAY_A = (date(2021, 11, 19) - date(1970, 1, 1)).days  # the cutoff day
DAY_B = DAY_A + 1
ITEM_COLUMNS = ["item_id", "title", "tag", "description", *prep.COUNTERS.values()]


def users_by_side() -> tuple[str, str, str]:
    """One user id hashed into train and two into test."""
    ids = [f"u{i}" for i in range(1000)]
    train = [u for u in ids if prep.split_of(u) == "train"]
    test = [u for u in ids if prep.split_of(u) == "test"]
    return train[0], test[0], test[1]


def at(day: int, hour: int, second: int) -> int:
    """Unix seconds for a day number, an hour and a second within it."""
    return day * prep.DAY + hour * 3600 + second


def write_raw(root, events: list[tuple[str, str, int]]) -> None:
    """The two PixelRec CSVs: the events, and one item row per item id."""
    (root / "interactions").mkdir(parents=True)
    (root / "items").mkdir()
    with open(root / "interactions" / "Pixel200K.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "user_id", "timestamp"])
        writer.writerows((item, user, second) for user, item, second in events)
    with open(root / "items" / "Pixel200K.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(ITEM_COLUMNS)
        for item in sorted({item for _, item, _ in events} | {"unused"}):
            writer.writerow(
                [item, f"title {item}", "Pets, Dogs", "", 10, 1, 1, 3, 0, 0, 0]
            )


def build(tmp_path, image_url_prefix=None):
    """Run the script's steps on synthetic data; return its outputs and the user ids."""
    train, test, dropped = users_by_side()
    events = [
        (train, "x", at(DAY_A, 5, 0)),
        (train, "x", at(DAY_A, 5, 1)),  # a repeat in a row, kept once
        (train, "y", at(DAY_A, 5, 2)),
        (train, "w", at(DAY_B, 9, 0)),  # after the cutoff: not a training session
        *((test, "p", at(DAY_A, 7, k)) for k in range(4)),  # before it: not a test one
        (test, "z", at(DAY_B, 13, 0)),
        (dropped, "q", at(DAY_A, 8, 0)),  # a test user with no test session
    ]
    write_raw(tmp_path / "raw", events)
    interactions = prep.read_strings(
        tmp_path / "raw" / "interactions" / "Pixel200K.csv",
        ["item_id", "user_id", "timestamp"],
    )
    rows, available_from, histograms, cutoff, dropped_users = prep.build_sequences(
        interactions
    )
    catalog = prep.build_catalog(
        tmp_path / "raw" / "items" / "Pixel200K.csv", available_from, histograms
    )
    prep.write_api_catalog(
        tmp_path, catalog, prep.sequence_items(rows), image_url_prefix
    )
    return rows, catalog, cutoff, dropped_users, (train, test)


def test_split_is_deterministic_and_close_to_the_share():
    users = [f"user{i}" for i in range(20_000)]
    sides = [prep.split_of(u) for u in users]
    assert sides == [prep.split_of(u) for u in users]
    assert abs(sides.count("test") / len(users) - prep.TEST_FRAC) < 0.01


def test_session_text():
    text = prep.session_text(DAY_A, 8, ["i1", "i2"])
    assert text == (
        "[gender:unk] [age_group:unk] [postal_code:unk] [market:unk] "
        "[year:2021] [Nov] [day:19] [Fri] [hour:8] i1 i2 <end>"
    )


def test_sessions_follow_the_cutoff(tmp_path):
    rows, _, cutoff, dropped_users, (train, test) = build(tmp_path)
    assert cutoff == DAY_A  # 8 of the 10 events happen on or before it
    assert [r["user_id"] for r in rows["train"]] == [train]
    assert rows["train"][0]["text"] == f"{train}, " + prep.session_text(
        DAY_A, 5, ["x", "y"]
    )
    assert [r["user_id"] for r in rows["test"]] == [test]
    assert rows["test"][0]["text"] == f"{test}, " + prep.session_text(DAY_B, 13, ["z"])
    assert dropped_users == {"test": 1}


def test_catalog_dates_and_sales(tmp_path):
    _, catalog, *_ = build(tmp_path)
    rows = {r["id"]: r for r in catalog.to_pylist()}
    assert set(rows) == {"x", "y", "w", "p", "z", "q", "unused"}
    assert rows["x"]["timestamp"] == DAY_A * prep.DAY
    assert rows["z"]["timestamp"] == DAY_B * prep.DAY  # first seen in either split
    assert rows["x"]["frequency"] == 1 and rows["x"]["sales_since"] == [1] * 12
    assert rows["z"]["frequency"] is None  # no training events
    assert rows["x"]["price"] == "3.00000000" and rows["x"]["categories"] == "Pets Dogs"
    assert rows["unused"]["timestamp"] is None


def test_api_catalog(tmp_path):
    build(tmp_path, image_url_prefix="https://images.example.com/")
    path = tmp_path / "catalog_api.parquet"
    check_catalog_schema(path)
    api = pq.read_table(path)
    assert api.column_names == prep.API_COLUMNS
    assert sorted(api.column("id").to_pylist()) == ["x", "y", "z"]
    for item, url in zip(
        api.column("id").to_pylist(), api.column("image_url").to_pylist()
    ):
        assert url == f"https://images.example.com/{item}.jpg"


def test_api_catalog_has_no_image_urls_by_default(tmp_path):
    build(tmp_path)
    api = pq.read_table(tmp_path / "catalog_api.parquet")
    assert api.column("image_url").null_count == api.num_rows
