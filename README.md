# NES-LTER API Dashboard

A simple Streamlit + Polars dashboard for data from <https://nes-lter-api.whoi.edu/>.

## Features

- Select cruises directly, or select all cruises overlapping a date range.
- Cruise track / geospatial station plot using cast latitude and longitude.
- Section plots of variables along depth versus latitude, longitude, or cast.
- Single-cast/station profile plots.
- Supports CTD bottle, nutrient, and chlorophyll endpoints.

## Run locally

```bash
pip install -r requirements.txt
streamlit run main.py
```

If using `uv`:

```bash
uv sync
uv run streamlit run main.py
```

## Deploy on Streamlit Community Cloud

Use these settings at <https://share.streamlit.io/>:

- Repository: `blongworth/lter-api-dashboard`
- Branch: `main`
- Main file path: `main.py`

No secrets are required.
