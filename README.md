# BehaviorGPT

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Unbox-AI/behaviorgpt/blob/main/notebooks/showcase.ipynb)

**BehaviorGPT** is UnboxAI's Large Behavioral Model. It is trained on long sequences of things people actually did (viewed this, added that, bought the other) and predicts what comes next. Search, recommendations and personalization are the same call with different histories.

This repository holds the Python client for the UnboxAI API and a notebook that walks through the model step by step.

## Requirements

- An UnboxAI API key. Sign up at [unboxai.com/behaviorgpt](https://unboxai.com/behaviorgpt); the key arrives by email.
- Python 3.11 or newer, if you run the notebook locally.

## The notebook

[`notebooks/showcase.ipynb`](notebooks/showcase.ipynb) covers the model in eight steps: setup, a first query, one call for many features, how context changes the answer, then embedding your own catalog, inspecting the embedding space, querying it, and trying it in the demo.

### Run it in the browser

Click the **Open in Colab** badge at the top of this page. Nothing to install: the first cell installs the client and Step 1 asks for your API key. To avoid retyping the key, store it in Colab's Secrets panel (key icon in the left sidebar) as `UNBOXAI_EMBED_API_KEY`. For Step 5, upload your parquet file through the Files panel before running the embed cell.

### Run it locally

Clone the repo and sync the environment with [uv](https://github.com/astral-sh/uv):

```sh
git clone https://github.com/Unbox-AI/behaviorgpt.git
cd behaviorgpt
uv sync
```

Copy `.env.example` to `.env` and paste your key:

```sh
cp .env.example .env
```

```
UNBOXAI_EMBED_API_KEY=your-key-here
```

Register the project environment as a Jupyter kernel once:

```sh
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=behaviorgpt-env
```

Then start JupyterLab. It opens in your browser at `http://localhost:8888`; open the notebook and pick the `behaviorgpt-env` kernel:

```sh
uv run jupyter lab
```

## Use the client in your own code

```sh
pip install behaviorgpt
```

The client reads `UNBOXAI_EMBED_API_KEY` from the environment on startup. You can also pass `api_key=` directly to `UnboxAIClient`.

## Bring your own catalog

The model can only return products it knows about. To use it on your store, upload your catalog as a parquet file:

```python
job = client.embed("my_catalog.parquet", wait=True, timeout=1800.0)
```

- The required columns and types are described in [`docs/catalog-format.md`](docs/catalog-format.md).
- A catalog can hold at most 20 000 products.
- [`examples/amazon_example_100.csv`](examples/amazon_example_100.csv) shows what a correctly shaped catalog looks like.

If mapping your data to the format is hard, open a coding agent in this directory, point it at `docs/catalog-format.md`, and ask it to convert your file into a matching parquet at the repo root.

## Try it in the demo

Once your catalog is embedded, open [behaviorgpt.unboxai.com](https://behaviorgpt.unboxai.com/), select the **BehaviorGPT V4.0-13B** model, choose **Bring your own catalog** and enter your API key. Product images must be publicly reachable for the demo to display them.
