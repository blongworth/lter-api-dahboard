# NES-LTER API Dashboard

A simple Streamlit + Polars dashboard for data from <https://nes-lter-api.whoi.edu/>.

## Features

- Select cruises directly, or select all cruises overlapping a date range.
- Cruise track / geospatial station plot using cast latitude and longitude.
- Section plots of variables along depth versus latitude, longitude, or cast.
- Single-cast/station profile plots.
- Supports CTD bottle, nutrient, and chlorophyll endpoints.

## Run

```bash
pip install -e .
streamlit run main.py
```

If using `uv`:

```bash
uv sync
uv run streamlit run main.py
```
