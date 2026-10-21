# Managing catalogs

How catalogs are named and limited, how to update, list and delete them, and how to add the keywords that `sections` needs. For what the uploaded file must contain, see [catalog-format.md](catalog-format.md).

## Names and limits

- A catalog can hold at most 100,000 items. Items added later with `add_items` count too.
- Each API key can hold 2 catalogs. To raise the limit, contact alexander@unboxai.com.
- The file name is the catalog's name. Uploading a file with the same name again refreshes that catalog and keeps its `catalog_id`; a new name takes one of the 2 slots. A failed upload does not use up a slot. Once both slots are in use, an upload under a new name is refused until you delete one of them.
- One `add_items` or `add_keywords` job runs per catalog at a time, and the catalog cannot be refreshed or deleted while it runs.

With `wait=True`, a progress bar shows each step of a job and the time left; where the output is not a terminal or a notebook, each new state is printed once instead. Pass `on_progress=lambda _: None` for silence.

## Adding or updating items

To add items to an embedded catalog, or change some of its items, upload just those items in the same format. Only they are embedded, so the job takes about as long as the file is big, and the catalog keeps answering queries while it runs:

```python
client.add_items("new_items.parquet", catalog_id=job.catalog_id, wait=True)
```

- An item whose `id` is already in the catalog is replaced with the new row; any other item is added. Nothing is removed.
- Uploading the catalog's file again with `embed` replaces the whole catalog, added items included. Put them in that file too to keep them.
- A catalog embedded before `add_items` existed needs one more full upload before items can be added to it.

## Listing and deleting catalogs

```python
for c in client.list_catalogs():
    print(c.catalog_id, c.catalog_name, c.status, c.owned)

client.delete_catalog("cat_...")
```

- `list_catalogs()` returns your own catalogs, in any status, and the shared ones that are ready. `owned` marks your own.
- `delete_catalog()` removes one of your catalogs with its items and frees its slot. It cannot be undone.
- Deleting another key's catalog fails with status 404, the same as an unknown id. While a job runs for the catalog, deleting fails with status 409; try again once the job is ready or failed.

## Keywords and sections

Sections group the items the model ranks under titles such as "black coats", best section first. They are built from item keywords, which `embed` does not generate by default because they take time. Generate them for an embedded catalog with `add_keywords`:

```python
client.add_keywords(job.catalog_id, wait=True)
response = client.sections([Search("coat")], catalog_id=job.catalog_id)
for section in response.sections:
    print(section.title, [item.data["name"] for item in section.items])
```

Or generate them during the upload, so the catalog goes live with them:

```python
job = client.embed(path, keywords=True, wait=True, timeout=4 * 3600.0)
```

- Keyword generation runs at roughly 150 items a minute, so a 20,000-item catalog takes about 2 hours 20 minutes. `add_keywords` waits up to 4 hours by default, enough for about 35,000 items; for a larger catalog, or with `embed(..., keywords=True)`, pass a longer `timeout`.
- `add_keywords` re-embeds nothing, and the catalog keeps answering queries while it runs. The keywords appear when the job is ready. If it fails, the catalog stays as it was.
- With `embed(..., keywords=True)`, if keyword generation fails, the catalog still goes live, without keywords. The job's status then reads `keywords_skipped`, with the reason in `keywords_error`, and `sections` returns an empty list for that catalog.
- Uploading the catalog again with `embed` replaces its keywords. Pass `keywords=True` there, or call `add_keywords` again, to keep them.
- Items added or replaced with `add_items` have no keywords until you call `add_keywords` again or upload the catalog with `embed(..., keywords=True)`; until then they are left out of sections.
- A history that ends in a search groups that search's results; any other history groups recommendations.
- `max_sections` can be at most 15 and `max_items_per_section` at most 100.
- Each item in a section carries its keywords in `item.data`: `top_terms`, the search terms generated for it, each with a score, best first; and `top_terms_mapped`, the section titles it fits. Items from `complete` carry `top_terms_mapped` only.
