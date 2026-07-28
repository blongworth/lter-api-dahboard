from __future__ import annotations

import io
from datetime import date
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

API_BASE = "https://nes-lter-api.whoi.edu/api"
DATASETS = {
    "CTD bottles": "ctd/bottles/{cruise}.csv",
    "Nutrients": "nut/{cruise}.csv",
    "Chlorophyll": "chl/{cruise}.csv",
}
DEPTH_SOURCE_NOTES = {
    "CTD bottles": "Depth is added from the API's CTD bottle summary metadata (`ctd/bottle_summary/{cruise}.csv`).",
    "Nutrients": "Depth is provided directly by the nutrients endpoint.",
    "Chlorophyll": "Depth is provided directly by the chlorophyll endpoint.",
}
IDENTIFIER_COLUMNS = {
    "name",
    "vessel",
    "type",
    "cruise_name",
    "cruise",
    "number",
    "cast",
    "replicate",
    "filter_size",
    "sample_id",
    "alternate_sample_id",
    "project_id",
    "nearest_station",
}
DATETIME_COLUMNS = {"start_time", "end_time", "date", "dateTime8601"}
PLOT_EXCLUDE_COLUMNS = {"latitude", "longitude", "depth", "cast", "niskin"}
BATHYMETRY_POINT_LIMIT = 8_000

st.set_page_config(page_title="NES-LTER API dashboard", layout="wide")


@st.cache_data(ttl="1h", max_entries=100, show_spinner=False)
def fetch_csv(path: str) -> pl.DataFrame:
    """Fetch one CSV endpoint from the NES-LTER API."""
    url = f"{API_BASE}/{path.lstrip('/')}"
    request = Request(url, headers={"User-Agent": "nes-lter-streamlit-dashboard/0.1"})
    try:
        with urlopen(request, timeout=45) as response:
            raw = response.read()
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not fetch {url}: {exc}") from exc

    if not raw.strip():
        return pl.DataFrame()

    return pl.read_csv(
        io.BytesIO(raw),
        null_values=["", "NaN", "nan", "NULL"],
        infer_schema_length=None,
    )


def normalize_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Coerce API CSV columns that should be numeric or datetime.

    Some endpoints mix numeric and station labels, for example cast `13b`.
    Reading with a full schema scan preserves those mixed identifiers as strings;
    this helper restores numeric types for plotting without losing identifiers.
    """
    exprs = []
    for name, dtype in df.schema.items():
        if name in DATETIME_COLUMNS and dtype == pl.String:
            exprs.append(pl.col(name).str.to_datetime(strict=False, time_zone="UTC").alias(name))
        elif name not in IDENTIFIER_COLUMNS and dtype == pl.String:
            exprs.append(pl.col(name).cast(pl.Float64, strict=False).alias(name))
    return df.with_columns(exprs) if exprs else df


def numeric_columns(df: pl.DataFrame, exclude: set[str] | None = None) -> list[str]:
    exclude = exclude or set()
    return [name for name, dtype in df.schema.items() if name not in exclude and dtype.is_numeric()]


@st.cache_data(ttl="1h", max_entries=5, show_spinner=False)
def load_cruises() -> pl.DataFrame:
    return normalize_dataframe(fetch_csv("ctd/cruises/all.csv")).sort("start_time")


@st.cache_data(ttl="1h", max_entries=30, show_spinner=False)
def load_casts(cruise_names: tuple[str, ...]) -> pl.DataFrame:
    frames = []
    for cruise in cruise_names:
        try:
            frames.append(normalize_dataframe(fetch_csv(f"ctd/casts/{cruise}.csv")))
        except RuntimeError:
            continue
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def join_bottle_summary_depth(cruise: str, bottles: pl.DataFrame) -> pl.DataFrame:
    """Add the API's canonical bottle-summary depth to the CTD bottle table.

    The full CTD bottle endpoint includes Sea-Bird columns such as `depsm`, but it
    does not expose the API's user-facing `depth` column. The metadata endpoint
    `ctd/bottle_summary/{cruise}.csv` defines that depth per cruise/cast/niskin,
    so we join it in and only fall back to `depsm` if the summary is unavailable.
    """
    if bottles.is_empty() or "depth" in bottles.columns:
        return bottles

    try:
        summary = normalize_dataframe(fetch_csv(f"ctd/bottle_summary/{cruise}.csv"))
    except RuntimeError:
        return bottles.with_columns(pl.col("depsm").alias("depth")) if "depsm" in bottles.columns else bottles

    join_keys = [key for key in ["cruise", "cast", "niskin"] if key in bottles.columns and key in summary.columns]
    if len(join_keys) != 3 or "depth" not in summary.columns:
        return bottles.with_columns(pl.col("depsm").alias("depth")) if "depsm" in bottles.columns else bottles

    left = bottles.with_columns([pl.col("cast").cast(pl.Utf8), pl.col("niskin").cast(pl.Int64, strict=False)])
    right = summary.select(join_keys + ["depth"]).with_columns(
        [pl.col("cast").cast(pl.Utf8), pl.col("niskin").cast(pl.Int64, strict=False)]
    )
    joined = left.join(right, on=join_keys, how="left")
    if "depsm" in joined.columns:
        joined = joined.with_columns(pl.coalesce([pl.col("depth"), pl.col("depsm")]).alias("depth"))
    return joined


@st.cache_data(ttl="1h", max_entries=5, show_spinner=False)
def load_bathymetry() -> pl.DataFrame:
    try:
        return normalize_dataframe(fetch_csv("ctd/bathymetry_file.csv"))
    except RuntimeError:
        # The API advertises this endpoint, but it can return 404 if the server-side
        # bathymetry file is not mounted. Keep the dashboard usable and surface a
        # notice in the map instead of failing the whole app.
        return pl.DataFrame()


@st.cache_data(ttl="1h", max_entries=30, show_spinner=False)
def load_dataset(cruise_names: tuple[str, ...], dataset: str) -> pl.DataFrame:
    endpoint = DATASETS[dataset]
    frames = []
    for cruise in cruise_names:
        try:
            df = normalize_dataframe(fetch_csv(endpoint.format(cruise=cruise)))
            if dataset == "CTD bottles":
                df = join_bottle_summary_depth(cruise, df)
            frames.append(df)
        except RuntimeError:
            continue
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def find_geospatial_columns(df: pl.DataFrame) -> tuple[str | None, str | None, str | None]:
    lower_to_name = {name.lower(): name for name in df.columns}
    latitude = next((lower_to_name[name] for name in ["latitude", "lat", "y"] if name in lower_to_name), None)
    longitude = next((lower_to_name[name] for name in ["longitude", "lon", "long", "x"] if name in lower_to_name), None)
    depth = next(
        (lower_to_name[name] for name in ["depth", "bathymetry", "elevation", "z"] if name in lower_to_name),
        None,
    )
    if depth is None:
        numeric = [name for name in numeric_columns(df) if name not in {latitude, longitude}]
        depth = numeric[0] if numeric else None
    return latitude, longitude, depth


def surface_observations(data_df: pl.DataFrame, variable: str) -> pl.DataFrame:
    required = {"cruise", "cast", "depth", variable}
    if data_df.is_empty() or not required.issubset(data_df.columns):
        return pl.DataFrame()

    return (
        data_df.select(["cruise", "cast", "depth", variable])
        .drop_nulls(["cruise", "cast", "depth", variable])
        .with_columns(pl.col("cast").cast(pl.Utf8))
        .sort("depth")
        .group_by(["cruise", "cast"], maintain_order=True)
        .first()
        .rename({"depth": "sample_depth"})
    )


def add_surface_variable(casts_df: pl.DataFrame, data_df: pl.DataFrame, variable: str) -> pl.DataFrame:
    surface_df = surface_observations(data_df, variable)
    if surface_df.is_empty() or not {"cruise_name", "number"}.issubset(casts_df.columns):
        return casts_df

    left = casts_df.with_columns(
        [
            pl.col("cruise_name").alias("cruise"),
            pl.col("number").cast(pl.Utf8).alias("cast"),
        ]
    )
    return left.join(surface_df, on=["cruise", "cast"], how="left").drop(["cruise", "cast"])


def cruises_in_date_range(cruise_df: pl.DataFrame, start: date, end: date) -> list[str]:
    filtered = cruise_df.filter(
        (pl.col("start_time").dt.date() <= end) & (pl.col("end_time").dt.date() >= start)
    )
    return filtered.get_column("name").to_list()


def dataframe_card(title: str, df: pl.DataFrame, *, key: str, height: int = 320) -> None:
    with st.container(border=True):
        st.subheader(title)
        st.dataframe(df, hide_index=True, height=height, width="stretch", key=key)


def render_metrics(summary: pl.DataFrame, casts_df: pl.DataFrame, data_df: pl.DataFrame | None, dataset: str) -> None:
    n_cruises = summary.height
    n_casts = casts_df.height
    n_rows = data_df.height if data_df is not None else 0
    start = summary.get_column("start_time").drop_nulls().min()
    end = summary.get_column("end_time").drop_nulls().max()
    date_label = "—"
    if start and end:
        date_label = f"{start.date()} to {end.date()}"

    with st.container(horizontal=True):
        st.metric("Cruises", f"{n_cruises:,}", border=True)
        st.metric("Casts", f"{n_casts:,}", border=True)
        st.metric(f"{dataset} rows", f"{n_rows:,}", border=True)
        st.metric("Date span", date_label, border=True)


def render_track(casts_df: pl.DataFrame, bathymetry_df: pl.DataFrame, data_df: pl.DataFrame, dataset: str) -> None:
    with st.container(border=True):
        st.subheader("Cruise track, stations, and bathymetry")
        if casts_df.is_empty() or not {"latitude", "longitude"}.issubset(casts_df.columns):
            st.warning("No cast location data were available for the selected cruise(s).")
            return

        surface_variables = numeric_columns(data_df, exclude=PLOT_EXCLUDE_COLUMNS) if not data_df.is_empty() else []
        with st.container(horizontal=True, vertical_alignment="bottom"):
            color_choice = st.selectbox(
                "Color stations by",
                ["Cruise"] + surface_variables,
                key="map_surface_variable",
                help="Surface values use the shallowest available sample for each cast/station.",
            )
            show_bathymetry = st.toggle("Show bathymetry", value=True, key="show_bathymetry")

        casts_plot = casts_df.drop_nulls(["latitude", "longitude"])
        color_column = "cruise_name"
        if color_choice != "Cruise":
            casts_plot = add_surface_variable(casts_plot, data_df, color_choice)
            if color_choice in casts_plot.columns and casts_plot.get_column(color_choice).drop_nulls().len() > 0:
                color_column = color_choice
            else:
                st.info(f"No surface `{color_choice}` values matched these cast locations; coloring by cruise instead.")

        hover_cols = [
            c
            for c in ["cruise_name", "number", "start_time", "depth", "sample_depth", color_choice]
            if c in casts_plot.columns
        ]
        fig = px.scatter_map(
            casts_plot.to_pandas(),
            lat="latitude",
            lon="longitude",
            color=color_column if color_column in casts_plot.columns else None,
            hover_data=hover_cols,
            color_continuous_scale="Viridis" if color_column == color_choice else None,
            zoom=6,
            height=700,
            map_style="open-street-map",
        )
        fig.update_traces(marker={"size": 11, "opacity": 0.9}, selector={"mode": "markers"})

        for cruise_name, group in casts_plot.sort(["cruise_name", "start_time"]).to_pandas().groupby("cruise_name"):
            fig.add_trace(
                go.Scattermap(
                    lat=group["latitude"],
                    lon=group["longitude"],
                    mode="lines",
                    line={"color": "rgba(35, 35, 35, 0.35)", "width": 2},
                    name=f"{cruise_name} track",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        bathy_lat, bathy_lon, bathy_depth = find_geospatial_columns(bathymetry_df)
        if show_bathymetry and bathy_lat and bathy_lon and bathy_depth:
            bathy_plot = bathymetry_df.drop_nulls([bathy_lat, bathy_lon, bathy_depth])
            if bathy_plot.height > BATHYMETRY_POINT_LIMIT:
                step = max(1, bathy_plot.height // BATHYMETRY_POINT_LIMIT)
                bathy_plot = bathy_plot[::step]
            bathy_pdf = bathy_plot.to_pandas()
            fig.add_trace(
                go.Scattermap(
                    lat=bathy_pdf[bathy_lat],
                    lon=bathy_pdf[bathy_lon],
                    mode="markers",
                    marker={
                        "size": 4,
                        "color": bathy_pdf[bathy_depth],
                        "colorscale": "Blues",
                        "opacity": 0.35,
                        "showscale": False,
                    },
                    name="Bathymetry",
                    text=bathy_pdf[bathy_depth],
                    hovertemplate="Bathymetry: %{text}<br>Lat: %{lat}<br>Lon: %{lon}<extra></extra>",
                )
            )
        elif show_bathymetry:
            st.info("Bathymetry data are unavailable or do not include latitude, longitude, and depth columns.")

        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend={"orientation": "h"})
        st.caption(
            f"Station colors use `{dataset}` surface data from the shallowest sample at each cast when a variable is selected."
        )
        st.plotly_chart(fig, width="stretch")


def render_section(data_df: pl.DataFrame, dataset: str, selected: list[str]) -> None:
    with st.container(border=True):
        st.subheader(f"Sections from {dataset}")
        st.caption(DEPTH_SOURCE_NOTES[dataset])
        if data_df.is_empty():
            st.warning(f"No {dataset} data were available for the selected cruise(s).")
            return
        if "depth" not in data_df.columns:
            st.warning("This dataset does not include a depth column.")
            return

        with st.container(horizontal=True, vertical_alignment="bottom"):
            x_options = [c for c in ["latitude", "longitude", "cast"] if c in data_df.columns]
            x_axis = st.selectbox("Section x-axis", x_options, key="section_x_axis")
            variables = numeric_columns(data_df, exclude=PLOT_EXCLUDE_COLUMNS)
            variable = st.selectbox("Variable", variables, key="section_variable")
            cruise_filter = st.multiselect("Cruises in section", selected, default=selected, key="section_cruises")

        section_df = data_df.filter(pl.col("cruise").is_in(cruise_filter)) if "cruise" in data_df.columns else data_df
        section_df = section_df.drop_nulls([x_axis, "depth", variable])
        if section_df.is_empty():
            st.info("No rows remain after applying the section filters.")
            return

        chart = (
            alt.Chart(section_df.to_pandas())
            .mark_circle(size=70, opacity=0.85)
            .encode(
                x=alt.X(f"{x_axis}:Q" if x_axis != "cast" else f"{x_axis}:N", title=x_axis.replace("_", " ").title()),
                y=alt.Y("depth:Q", title="Depth (m)", sort="descending"),
                color=alt.Color(f"{variable}:Q", title=variable.replace("_", " ").title(), scale=alt.Scale(scheme="viridis")),
                tooltip=[c for c in ["cruise", "cast", "niskin", "date", "latitude", "longitude", "depth", variable] if c in section_df.columns],
            )
            .interactive()
            .properties(height=620)
        )
        st.altair_chart(chart, width="stretch")


def render_profile(data_df: pl.DataFrame, dataset: str, selected: list[str]) -> None:
    with st.container(border=True):
        st.subheader(f"Single-station profiles from {dataset}")
        st.caption(DEPTH_SOURCE_NOTES[dataset])
        if data_df.is_empty() or "depth" not in data_df.columns:
            st.warning("Load a dataset with depth values to view profiles.")
            return

        profile_cruises = data_df.get_column("cruise").unique().sort().to_list() if "cruise" in data_df.columns else selected
        with st.container(horizontal=True, vertical_alignment="bottom"):
            profile_cruise = st.selectbox("Cruise", profile_cruises, key="profile_cruise")
            subset = data_df.filter(pl.col("cruise") == profile_cruise) if "cruise" in data_df.columns else data_df
            casts = subset.get_column("cast").cast(pl.Utf8).unique().sort().to_list() if "cast" in subset.columns else []
            profile_cast = st.selectbox("Cast/station", casts, key="profile_cast")
            variables = numeric_columns(subset, exclude=PLOT_EXCLUDE_COLUMNS)
            profile_var = st.selectbox("Profile variable", variables, key="profile_variable")

        profile_df = subset.filter(pl.col("cast").cast(pl.Utf8) == str(profile_cast)).drop_nulls(["depth", profile_var])
        if profile_df.is_empty():
            st.info("No profile rows remain after applying the filters.")
            return

        color_column = "replicate" if "replicate" in profile_df.columns else alt.value("#1f77b4")
        chart = (
            alt.Chart(profile_df.sort("depth").to_pandas())
            .mark_line(point=True)
            .encode(
                x=alt.X(f"{profile_var}:Q", title=profile_var.replace("_", " ").title()),
                y=alt.Y("depth:Q", title="Depth (m)", sort="descending"),
                color=color_column,
                tooltip=[c for c in ["cruise", "cast", "niskin", "replicate", "date", "latitude", "longitude", "depth", profile_var] if c in profile_df.columns],
            )
            .interactive()
            .properties(height=620)
        )
        st.altair_chart(chart, width="stretch")


def render_data(casts_df: pl.DataFrame, data_df: pl.DataFrame, dataset: str) -> None:
    with st.container(border=True):
        st.subheader("Loaded data")
        st.write(f"Casts: {casts_df.height:,} rows; {dataset}: {data_df.height:,} rows")
        st.caption(DEPTH_SOURCE_NOTES[dataset])
        st.dataframe(data_df, hide_index=True, height=650, width="stretch", key="loaded_data")


st.title("NES-LTER API dashboard")
st.caption("Explore cruise tracks, depth sections, and single-station profiles from the NES-LTER API.")

try:
    cruise_df = load_cruises()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.header("Filters")
    selection_mode = st.segmented_control("Select by", ["Cruises", "Date range"], default="Cruises")

    cruise_names = cruise_df.get_column("name").to_list()
    if selection_mode == "Cruises":
        default = [name for name in ["EN608"] if name in cruise_names] or cruise_names[-1:]
        selected = st.multiselect("Cruises", cruise_names, default=default, key="selected_cruises")
    else:
        min_dt = cruise_df.get_column("start_time").drop_nulls().min()
        max_dt = cruise_df.get_column("end_time").drop_nulls().max()
        date_range = st.date_input(
            "Date range",
            value=(min_dt.date() if min_dt else date(2017, 1, 1), max_dt.date() if max_dt else date.today()),
            key="selected_date_range",
        )
        if len(date_range) != 2:
            st.stop()
        selected = cruises_in_date_range(cruise_df, *date_range)
        st.caption(f"{len(selected)} cruise(s) overlap this range.")

    dataset = st.selectbox("Section/profile dataset", list(DATASETS), key="dataset")
    view = st.segmented_control(
        "View",
        ["Cruise track", "Sections", "Profiles", "Data"],
        default="Cruise track",
        key="view",
    )

if not selected:
    st.info("Select at least one cruise to begin.")
    st.stop()

selected_tuple = tuple(selected)
summary = cruise_df.filter(pl.col("name").is_in(selected))

with st.skeleton(height=220):
    casts_df = load_casts(selected_tuple)

needs_dataset = view in {"Cruise track", "Sections", "Profiles", "Data"}
data_df = None
if needs_dataset:
    with st.skeleton(height=220):
        data_df = load_dataset(selected_tuple, dataset)

bathymetry_df = pl.DataFrame()
if view == "Cruise track":
    with st.skeleton(height=220):
        bathymetry_df = load_bathymetry()

render_metrics(summary, casts_df, data_df, dataset)

if view == "Cruise track":
    render_track(casts_df, bathymetry_df, data_df if data_df is not None else pl.DataFrame(), dataset)
elif view == "Sections":
    render_section(data_df if data_df is not None else pl.DataFrame(), dataset, selected)
elif view == "Profiles":
    render_profile(data_df if data_df is not None else pl.DataFrame(), dataset, selected)
elif view == "Data":
    render_data(casts_df, data_df if data_df is not None else pl.DataFrame(), dataset)

with st.expander("Selected cruises"):
    st.dataframe(summary, hide_index=True, width="stretch", key="selected_cruises_table")
