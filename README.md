# BehaviorGPT

**BehaviorGPT** is our flagship Large Behavioral Model (LBM) which achieves State of The Art on numeorus benchmarks.

This repository contains the Python core client for interacting with our API as well as some example Notebooks for following along and experimenting with our LBM.

It provides access to the UnboxAI REST API from any Python 3.11+ application. The library includes type definitions for all request params and response fields, and offers both synchronous and asynchronous clients powered by `httpx`.

## Installation

Install from PyPI:

```sh
pip install behaviorgpt
```

Or,

Clone this repo:

```sh
git clone https://github.com/Unbox-AI/behaviorgpt.git
```

And then, install [uv](https://github.com/astral-sh/uv) and sync the dependencies:

```sh
uv sync
source .venv/bin/activate
```

## Usage

You will need an UnboxAI API Key to use this repository, get yours at: [UnboxAI](https://behaviorgpt-staging.unboxai.com/).

The maximum amount of products that can be present in your catalog is currently: 20 000 

If you find mapping the data of your product catalog difficult. Do the following: 

- Invoke your coding agent in this dir.
- Tell the agent to read the catalog-format.md file in /docs.
- Give instructions to structure your catalog, regardless of file, into what the .md file suggests, to the best of its abaility.
- Add the final parquet file to the repo root.

### Examples

The `notebooks/*` dir holds a plethora of examples on how you can use **BehaviorGPT** with this UnboxAI SDK:

- How to embed your own catalog, sync it with the LBM and visualize the embedding space;
- How to run personalized search and recommendation on the selected assortment;

Set the `uv` .venv as a valid Jupyter kernel:

```sh
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=behaviorgpt-env
```

To launch the Jupyter environment:

```sh
uv run jupyter lab
```
