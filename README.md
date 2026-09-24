# BehaviorGPT

**BehaviorGPT** is UnboxAI's Large Behavioral Model. It is trained on long sequences of things people actually did (viewed this, added that, bought the other) and predicts what comes next. Search, recommendations and personalization are the same call with different histories.

This repository holds the Python client for the UnboxAI API and a notebook that walks through the model step by step.

## Requirements

- An UnboxAI API key. Sign up at [unboxai.com/behaviorgpt](https://unboxai.com/behaviorgpt); the key arrives by email.
- Python 3.11 or newer.

## The notebook

[`notebooks/showcase.ipynb`](notebooks/showcase.ipynb) covers the model in eight steps: setup, a first query, one call for many features, how context changes the answer, then embedding your own catalog, inspecting the embedding space, querying it, and trying it in the demo.

Needs [uv](https://github.com/astral-sh/uv). Clone, install, add your key, register the kernel, start JupyterLab:

```sh
git clone https://github.com/Unbox-AI/behaviorgpt.git
cd behaviorgpt
uv sync
cp .env.example .env   # paste your key as UNBOXAI_EMBED_API_KEY
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=behaviorgpt-env
uv run jupyter lab
```

JupyterLab opens in your browser. Open the notebook and pick the `behaviorgpt-env` kernel.

## Use the client in your own code

```sh
pip install behaviorgpt
```

The client reads `UNBOXAI_EMBED_API_KEY` from the environment on startup. You can also pass `api_key=` directly to `UnboxAIClient`.

## Usage

The model can only return products it knows about. To use it on your store, upload your catalog as a parquet file.

- The required columns and types are described in [`docs/catalog-format.md`](docs/catalog-format.md).
- A catalog can hold at most 20 000 products.
- [`examples/amazon_example_100.csv`](examples/amazon_example_100.csv) shows what a correctly shaped catalog looks like.

If mapping your data to the format is hard, open a coding agent in this directory, point it at `docs/catalog-format.md`, and ask it to convert your file into a matching parquet at the repo root.

## Try it in the demo

Once your catalog is embedded, open [behaviorgpt.unboxai.com](https://behaviorgpt.unboxai.com/), select the **BehaviorGPT V4.0-13B** model, choose **Bring your own catalog** and enter your API key. Product images must be publicly reachable for the demo to display them.
