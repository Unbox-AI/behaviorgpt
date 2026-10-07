"""Build the PixelRec benchmark split read by notebooks/benchmark_reproduction.ipynb.

Input (PixelRec release, one variant):

    <raw>/interactions/Pixel200K.csv   item_id, user_id, timestamp (unix seconds)
    <raw>/items/Pixel200K.csv          item_id, title, tag, description, counters

Output, the layout the notebook reads:

    <out>/sequences/{train,test}/shard_0_1.parquet    one row per user
    <out>/catalog/shard_0_1.arrow                     one row per item
    <out>/meta.json
    <out>/catalog_api.parquet                         the API catalog, see below

The API catalog is the catalog in the format the API's embed endpoint takes: every
item that occurs in the sequences, in the endpoint's columns. Its image URLs are
`<prefix><id>.jpg` with --image-url-prefix and empty without.

Protocol: users are hashed into train (90 %) and test (10 %); the cutoff is the day by
which 80 % of all events have happened; train keeps sessions on or before it, test
keeps sessions after it. A session is one user's events on one UTC day, stamped with
the hour of its first event; repeats of an item in a row are kept once. Users left with
no sessions on their side are dropped. An item's `timestamp` is the first day it occurs
in either split; its sales timeline counts training events only.

Usage:
    python prepare_pixelrec.py --raw <raw> --out <out> [--image-url-prefix <url>]
"""

import argparse
import json
import zlib
from collections import Counter
from datetime import date
from itertools import groupby
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

VARIANT = "Pixel200K"
TEST_FRAC, CUTOFF_SHARE = 0.1, 0.8
TIMELINE_DAYS, TIMELINE_PERIODS = 30, 11
DAY = 86_400
EPOCH = date(1970, 1, 1).toordinal()
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
WEEKDAYS = "Mon Tue Wed Thu Fri Sat Sun".split()
PROFILE = "[gender:unk] [age_group:unk] [postal_code:unk] [market:unk]"
COUNTERS = {
    "views": "view_number",
    "likes": "thumbup_number",
    "saves": "favorite_number",
    "coins": "coin_number",
    "comments": "comment_number",
    "shares": "share_number",
    "danmaku": "barrage_number",
}
CATALOG_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("name", pa.string()),
        ("market", pa.string()),
        ("image_url", pa.string()),
        ("brand", pa.string()),
        ("price", pa.string()),
        ("currency", pa.string()),
        ("search_keywords", pa.list_(pa.string())),
        ("categories", pa.string()),
        ("keywords", pa.list_(pa.string())),
        ("event_type", pa.string()),
        ("group", pa.string()),
        ("sales_since", pa.list_(pa.int64())),
        ("timestamp", pa.int64()),
        ("frequency", pa.int64()),
    ]
    + [(f"count_{label}", pa.int64()) for label in COUNTERS]
    + [("description", pa.string())]
)
API_COLUMNS = [
    "id",
    "name",
    "brand",
    "categories",
    "image_url",
    "market",
    "price",
    "currency",
    "search_keywords",
    "keywords",
    "event_type",
    "group",
    "timestamp",
    "sales_since",
    "frequency",
]


def read_strings(path: Path, columns: list[str]) -> pa.Table:
    """Read CSV columns as strings, so ids keep their exact spelling."""
    options = pacsv.ConvertOptions(
        include_columns=columns, column_types=dict.fromkeys(columns, pa.string())
    )
    return pacsv.read_csv(path, convert_options=options)


def crc32(text: str) -> int:
    """Reproducible string hash."""
    return zlib.crc32(text.encode())


def split_of(user_id: str) -> str:
    """Hash a user into train or test."""
    return (
        "test"
        if crc32(f"{user_id}|split") % 10_000 < round(TEST_FRAC * 10_000)
        else "train"
    )


def to_date(day: int) -> date:
    """A UTC day number (days since 1970-01-01) as a date."""
    return date.fromordinal(EPOCH + day)


def session_text(day: int, hour: int, items: list[str]) -> str:
    """One session: nine domain tokens, the items, <end>."""
    d = to_date(day)
    month, weekday = MONTHS[d.month - 1], WEEKDAYS[d.weekday()]
    when = f"[year:{d.year}] [{month}] [day:{d.day}] [{weekday}] [hour:{hour}]"
    return f"{PROFILE} {when} {' '.join(items)} <end>"


def timeline_slot(offset_days: int) -> int:
    """sales_since slot for an occurrence this many days after the item's first day."""
    return (
        0
        if offset_days <= 0
        else min((offset_days - 1) // TIMELINE_DAYS, TIMELINE_PERIODS)
    )


def build_sequences(
    interactions: pa.Table,
) -> tuple[dict[str, list[dict]], dict[str, int], dict[str, list[int]], int, Counter]:
    """Split, cut and render every user.

    Returns the rows per split, each item's first day, each item's training histogram,
    the cutoff day and the number of users dropped per split. Days are UTC day numbers.
    """
    items = interactions.column("item_id").to_pylist()
    users = interactions.column("user_id").to_pylist()
    seconds = pc.cast(interactions.column("timestamp"), pa.int64()).to_pylist()
    days = [second // DAY for second in seconds]
    split = {user: split_of(user) for user in set(users)}

    available_from, first_seen = {}, {}
    for item, user, day in zip(items, users, days):
        available_from[item] = min(day, available_from.get(item, day))
        if split[user] == "train":
            first_seen[item] = min(day, first_seen.get(item, day))

    events_per_day, target, running = Counter(days), len(days) * CUTOFF_SHARE, 0
    for cutoff in sorted(events_per_day):
        running += events_per_day[cutoff]
        if running >= target:
            break

    order = sorted(range(len(items)), key=lambda i: (users[i], seconds[i]))
    rows, histograms, dropped = {"train": [], "test": []}, {}, Counter()
    empty_histogram = [0] * (TIMELINE_PERIODS + 1)
    for user, positions in groupby(order, key=lambda i: users[i]):
        side, sessions = split[user], []
        for day, day_positions in groupby(positions, key=lambda i: days[i]):
            if (day <= cutoff) != (side == "train"):
                continue
            # events in the same second have no order in the data; order them by item
            events = sorted(
                ((seconds[i], items[i]) for i in day_positions),
                key=lambda event: (event[0], crc32(event[1])),
            )
            kept = [
                item
                for k, (_, item) in enumerate(events)
                if k == 0 or item != events[k - 1][1]
            ]
            if side == "train":
                for item in kept:
                    histogram = histograms.setdefault(item, empty_histogram.copy())
                    histogram[timeline_slot(day - first_seen[item])] += 1
            sessions.append(session_text(day, events[0][0] % DAY // 3600, kept))
        if not sessions:
            dropped[side] += 1
            continue
        rows[side].append({"user_id": user, "text": f"{user}, " + " ".join(sessions)})
    return rows, available_from, histograms, cutoff, dropped


def build_catalog(
    items_csv: Path, available_from: dict[str, int], histograms: dict[str, list[int]]
) -> pa.Table:
    """One row per item in the items file, with first-seen day and sales timeline."""
    table = read_strings(
        items_csv, ["item_id", "title", "tag", "description", *COUNTERS.values()]
    )
    ids = table.column("item_id").to_pylist()
    counters = {
        label: [
            None if v is None or not v.strip() else float(v)
            for v in table.column(column).to_pylist()
        ]
        for label, column in COUNTERS.items()
    }

    def cumulative(histogram: list[int] | None) -> list[int] | None:
        if histogram is None:
            return None
        out, running = [], 0
        for count in histogram[:TIMELINE_PERIODS]:
            running += count
            out.append(running)
        return out + [running + histogram[TIMELINE_PERIODS]]

    def midnight(item: str) -> int | None:
        day = available_from.get(item)
        return None if day is None else day * DAY

    def category(tag: str | None) -> str | None:
        cleaned = " ".join((tag or "").replace(",", " ").split())
        return cleaned or None

    nulls = [None] * len(ids)
    columns = {
        "id": ids,
        "name": table.column("title").to_pylist(),
        "market": nulls,
        "image_url": nulls,
        "brand": nulls,
        # PixelRec has no prices; the coin count (coins viewers give a video) stands in
        "price": [None if c is None else f"{c:.8f}" for c in counters["coins"]],
        "currency": nulls,
        "search_keywords": nulls,
        "categories": [category(t) for t in table.column("tag").to_pylist()],
        "keywords": nulls,
        "event_type": ["product"] * len(ids),
        "group": ["product"] * len(ids),
        "sales_since": [cumulative(histograms.get(i)) for i in ids],
        "timestamp": [midnight(i) for i in ids],
        "frequency": [sum(histograms[i]) if i in histograms else None for i in ids],
        **{
            f"count_{label}": [None if v is None else int(v) for v in values]
            for label, values in counters.items()
        },
        "description": table.column("description").to_pylist(),
    }
    return pa.table(columns, schema=CATALOG_SCHEMA)


def write(
    out: Path, rows: dict[str, list[dict]], catalog: pa.Table, meta: dict
) -> None:
    """Write sequences, catalog and meta.json."""
    for side, side_rows in rows.items():
        (out / "sequences" / side).mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(side_rows),
            out / "sequences" / side / "shard_0_1.parquet",
        )
    (out / "catalog").mkdir(parents=True, exist_ok=True)
    with pa.ipc.new_file(out / "catalog" / "shard_0_1.arrow", catalog.schema) as writer:
        writer.write_table(catalog)
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def api_catalog(catalog: pa.Table, image_url_prefix: str | None = None) -> pa.Table:
    """The catalog's API columns; image URLs are `<prefix><id>.jpg`, or left empty."""
    if image_url_prefix:
        urls = pc.binary_join_element_wise(
            image_url_prefix, catalog.column("id"), ".jpg", ""
        )
    else:
        urls = pa.nulls(catalog.num_rows, pa.string())
    index = catalog.schema.get_field_index("image_url")
    return catalog.set_column(index, "image_url", urls).select(API_COLUMNS)


def sequence_items(rows: dict[str, list[dict]]) -> set[str]:
    """Every item id that occurs in the written sequences."""
    return {
        token
        for side_rows in rows.values()
        for row in side_rows
        for token in row["text"].split(", ", 1)[1].split()
        if token[0] not in "[<"
    }


def write_api_catalog(
    out: Path, catalog: pa.Table, items: set[str], image_url_prefix: str | None
) -> None:
    """Write the API catalog: every item that occurs in the sequences."""
    used = catalog.filter(
        pa.array([i in items for i in catalog.column("id").to_pylist()])
    )
    pq.write_table(api_catalog(used, image_url_prefix), out / "catalog_api.parquet")
    print(f"catalog_api: {used.num_rows:,} items occur in the sequences")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image-url-prefix")
    args = parser.parse_args()

    interactions = read_strings(
        args.raw / "interactions" / f"{VARIANT}.csv",
        ["item_id", "user_id", "timestamp"],
    )
    print(f"{interactions.num_rows:,} interactions")
    rows, available_from, histograms, cutoff, dropped = build_sequences(interactions)
    catalog = build_catalog(
        args.raw / "items" / f"{VARIANT}.csv", available_from, histograms
    )
    meta = {
        "variant": VARIANT,
        "test_frac": TEST_FRAC,
        "cutoff_share": CUTOFF_SHARE,
        "cutoff": to_date(cutoff).isoformat(),
        "dropped_users": dict(dropped),
        "counts": {f"{side}_users": len(r) for side, r in rows.items()}
        | {"items": catalog.num_rows, "items_in_train": len(histograms)},
    }
    write(args.out, rows, catalog, meta)
    print(json.dumps(meta, indent=2))
    write_api_catalog(args.out, catalog, sequence_items(rows), args.image_url_prefix)


if __name__ == "__main__":
    main()
