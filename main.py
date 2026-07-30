from __future__ import annotations

import io
from datetime import date, datetime
from math import asin, cos, radians, sin, sqrt
from urllib.parse import quote
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
UNDERWAY_ENDPOINT = "underway/{cruise}.csv"
GEBCO_OPENDAP_URL = (
    "https://dap.ceda.ac.uk/thredds/dodsC/bodc/gebco/global/gebco_2026/"
    "ice_surface_elevation/netcdf/GEBCO_2026.nc"
)
DEPTH_SOURCE_NOTES = {
    "CTD bottles": "Depth uses the bottle endpoint's `depsm` field.",
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
    "station",
    "stationfullname",
    "comment",
}
DATETIME_COLUMNS = {"start_time", "end_time", "date", "dateTime8601"}
PLOT_EXCLUDE_COLUMNS = {
    "latitude",
    "longitude",
    "dec_lat",
    "dec_lon",
    "latitude_deg",
    "longitude_deg",
    "gps_furuno_latitude",
    "gps_furuno_longitude",
    "depth",
    "cast",
    "niskin",
    "nearest_station_distance_km",
}
OCEAN_BASEMAP_TILE_URL = "https://services.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}"
OCEAN_BASEMAP_ATTRIBUTION = "Esri, GEBCO, NOAA, National Geographic, Garmin, HERE, Geonames.org, and other contributors"

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
            exprs.append(
                pl.col(name).str.to_datetime(strict=False, time_zone="UTC").alias(name)
            )
        elif name not in IDENTIFIER_COLUMNS and dtype == pl.String:
            exprs.append(pl.col(name).cast(pl.Float64, strict=False).alias(name))
    normalized = df.with_columns(exprs) if exprs else df
    for name in ["cast", "number", "niskin"]:
        if name not in normalized.columns or normalized.schema[name] != pl.String:
            continue
        numeric = normalized.get_column(name).cast(pl.Int64, strict=False)
        if numeric.null_count() == normalized.get_column(name).null_count():
            normalized = normalized.with_columns(numeric.alias(name))
    return normalized


def numeric_columns(df: pl.DataFrame, exclude: set[str] | None = None) -> list[str]:
    exclude = exclude or set()
    return [
        name
        for name, dtype in df.schema.items()
        if name not in exclude and dtype.is_numeric()
    ]


def normalize_underway_coordinates(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize vessel-specific underway navigation columns for the map."""
    latitude_aliases = [
        "latitude",
        "dec_lat",
        "latitude_deg",
        "gps_furuno_latitude",
        "gps_garmin741_latitude",
        "gps_nstarwaas_latitude",
        "gnss_adu2_latitude",
        "gnss_adu5_latitude",
    ]
    longitude_aliases = [
        "longitude",
        "dec_lon",
        "longitude_deg",
        "gps_furuno_longitude",
        "gps_garmin741_longitude",
        "gps_nstarwaas_longitude",
        "gnss_adu2_longitude",
        "gnss_adu5_longitude",
    ]
    expressions = []
    for canonical, aliases in [
        ("latitude", latitude_aliases),
        ("longitude", longitude_aliases),
    ]:
        available = [alias for alias in aliases if alias in df.columns]
        if available:
            expressions.append(
                pl.coalesce(
                    [pl.col(alias).cast(pl.Float64, strict=False) for alias in available]
                ).alias(canonical)
            )
    return df.with_columns(expressions) if expressions else df


def add_temperature_property(df: pl.DataFrame, *, bottle: bool = False) -> pl.DataFrame:
    """Expose a consistent temperature property across vessel/data schemas."""
    candidates = (
        ["temperature", "t090c", "potemp090c", "temp"]
        if bottle
        else [
            "tsg_sst",
            "tsg1_sst",
            "tsg2_sst",
            "tst_temperature",
            "tsg_temperature",
            "tsg1_temperature",
            "tsg2_temperature",
            "sbe48t",
            "aml_sst",
            "water_temperature_degree_c",
        ]
    )
    available = [candidate for candidate in candidates if candidate in df.columns]
    if not available:
        return df
    value = pl.coalesce(
        [pl.col(column).cast(pl.Float64, strict=False) for column in available]
    )
    expressions = [value.alias("temperature")]
    if not bottle:
        expressions.append(value.alias("sst"))
    return df.with_columns(expressions)


def property_options(
    df: pl.DataFrame, *, include_cruise: bool = False
) -> list[str]:
    options = numeric_columns(df, exclude=PLOT_EXCLUDE_COLUMNS) if not df.is_empty() else []
    return (["Cruise"] if include_cruise else []) + options


def temperature_property_index(options: list[str]) -> int:
    """Select the first common temperature field, falling back to the first option."""
    temperature_fields = [
        "temperature",
        "tsg_sst",
        "tsg1_sst",
        "tsg2_sst",
        "tst_temperature",
        "tsg_temperature",
        "tsg1_temperature",
        "tsg2_temperature",
        "sbe48t",
        "aml_sst",
        "water_temperature_degree_c",
        "t090c",
        "potemp090c",
        "temperature",
        "temp",
    ]
    for field in temperature_fields:
        if field in options:
            return options.index(field)
    return 0


def sst_property_index(options: list[str]) -> int:
    return options.index("sst") if "sst" in options else temperature_property_index(options)


@st.cache_data(ttl="1h", max_entries=5, show_spinner=False)
def load_cruises() -> pl.DataFrame:
    return normalize_dataframe(fetch_csv("ctd/cruises/all.csv")).sort("start_time")


@st.cache_data(ttl="1h", max_entries=30, show_spinner=False)
def load_casts(cruise_names: tuple[str, ...]) -> pl.DataFrame:
    frames = []
    for cruise in cruise_names:
        try:
            cast_index = normalize_dataframe(fetch_csv(f"ctd/casts/{cruise}.csv"))
            if "number" not in cast_index.columns:
                continue

            metadata_columns = [
                column
                for column in [
                    "cruise_name",
                    "number",
                    "latitude",
                    "longitude",
                    "depth",
                    "start_time",
                    "end_time",
                ]
                if column in cast_index.columns
            ]
            metadata = cast_index.select(metadata_columns).with_columns(
                pl.col("number").cast(pl.Utf8)
            )
            for cast in (
                cast_index.get_column("number")
                .drop_nulls()
                .cast(pl.Utf8)
                .unique()
                .to_list()
            ):
                try:
                    detail = normalize_dataframe(
                        fetch_csv(
                            f"ctd/cast/{quote(cruise, safe='')}/"
                            f"{quote(cast, safe='')}.csv"
                        )
                    )
                except RuntimeError:
                    continue
                if detail.is_empty():
                    continue
                if "cruise" not in detail.columns:
                    detail = detail.with_columns(pl.lit(cruise).alias("cruise"))
                if "cast" not in detail.columns:
                    detail = detail.with_columns(pl.lit(cast).alias("cast"))
                detail = detail.with_columns(
                    [
                        pl.col("cruise").alias("cruise_name"),
                        pl.col("cast").cast(pl.Utf8).alias("number"),
                    ]
                )
                frames.append(
                    detail.join(metadata, on=["cruise_name", "number"], how="left")
                )
        except RuntimeError:
            continue
    return (
        normalize_dataframe(pl.concat(frames, how="diagonal_relaxed"))
        if frames
        else pl.DataFrame()
    )


@st.cache_data(ttl="1h", max_entries=30, show_spinner=False)
def load_underway(cruise_names: tuple[str, ...]) -> pl.DataFrame:
    """Load underway navigation and surface observations for each cruise."""
    frames = []
    for cruise in cruise_names:
        try:
            df = normalize_underway_coordinates(
                normalize_dataframe(fetch_csv(UNDERWAY_ENDPOINT.format(cruise=cruise)))
            )
            df = add_temperature_property(df)
            if "cruise" not in df.columns:
                df = df.with_columns(pl.lit(cruise).alias("cruise"))
            frames.append(df)
        except RuntimeError:
            continue
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def use_bottle_endpoint_depth(bottles: pl.DataFrame) -> pl.DataFrame:
    """Use the CTD bottle endpoint's own depth field for plotting."""
    if bottles.is_empty() or "depth" in bottles.columns:
        return bottles
    return (
        bottles.with_columns(pl.col("depsm").alias("depth"))
        if "depsm" in bottles.columns
        else bottles
    )


@st.cache_data(ttl="24h", max_entries=1, show_spinner=False)
def load_stations() -> pl.DataFrame:
    """Load the NES-LTER station reference file from the API."""
    return fetch_csv("stations/file.csv").with_columns(
        [
            pl.col("startDate").str.to_date(strict=False).alias("start_date"),
            pl.when(pl.col("endDate") == "current")
            .then(None)
            .otherwise(pl.col("endDate"))
            .str.to_date(strict=False)
            .alias("end_date"),
            pl.col("decimalLatitude").alias("station_latitude"),
            pl.col("decimalLongitude").alias("station_longitude"),
        ]
    )


def observation_date(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        except ValueError:
            return None
    return None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    )
    return 2 * radius_km * asin(sqrt(a))


def add_nearest_station(data_df: pl.DataFrame) -> pl.DataFrame:
    """Add nearest station labels from the API station reference endpoint."""
    if data_df.is_empty() or "nearest_station" in data_df.columns:
        return data_df
    if not {"latitude", "longitude"}.issubset(data_df.columns):
        return data_df

    stations_df = load_stations().drop_nulls(
        ["station", "station_latitude", "station_longitude"]
    )
    if stations_df.is_empty():
        return data_df

    station_records = stations_df.select(
        ["station", "station_latitude", "station_longitude", "start_date", "end_date"]
    ).to_dicts()
    nearest_stations: list[str | None] = []
    nearest_distances: list[float | None] = []
    cache: dict[tuple[float, float, date | None], tuple[str | None, float | None]] = {}
    date_column = "date" if "date" in data_df.columns else None

    for row in data_df.select(
        ["latitude", "longitude"] + ([date_column] if date_column else [])
    ).iter_rows(named=True):
        lat = row["latitude"]
        lon = row["longitude"]
        obs_date = observation_date(row[date_column]) if date_column else None
        if lat is None or lon is None:
            nearest_stations.append(None)
            nearest_distances.append(None)
            continue

        key = (round(float(lat), 5), round(float(lon), 5), obs_date)
        if key not in cache:
            candidates = []
            for station in station_records:
                start_date = station["start_date"]
                end_date = station["end_date"]
                if obs_date and start_date and obs_date < start_date:
                    continue
                if obs_date and end_date and obs_date > end_date:
                    continue
                distance = haversine_km(
                    float(lat),
                    float(lon),
                    station["station_latitude"],
                    station["station_longitude"],
                )
                candidates.append((station["station"], distance))
            cache[key] = (
                min(candidates, key=lambda item: item[1])
                if candidates
                else (None, None)
            )

        station_name, distance_km = cache[key]
        nearest_stations.append(station_name)
        nearest_distances.append(distance_km)

    return data_df.with_columns(
        [
            pl.Series("nearest_station", nearest_stations),
            pl.Series("nearest_station_distance_km", nearest_distances),
        ]
    )


@st.cache_data(ttl="1h", max_entries=30, show_spinner=False)
def load_dataset(cruise_names: tuple[str, ...], dataset: str) -> pl.DataFrame:
    endpoint = DATASETS[dataset]
    frames = []
    for cruise in cruise_names:
        try:
            df = normalize_dataframe(fetch_csv(endpoint.format(cruise=cruise)))
            if dataset == "CTD bottles":
                df = use_bottle_endpoint_depth(df)
                df = add_temperature_property(df, bottle=True)
            frames.append(add_nearest_station(df))
        except RuntimeError:
            continue
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def cast_plot_data(casts_df: pl.DataFrame) -> pl.DataFrame:
    """Normalize cast-endpoint fields for section and profile plots."""
    if casts_df.is_empty():
        return pl.DataFrame()

    expressions = []
    if "cruise_name" in casts_df.columns:
        expressions.append(pl.col("cruise_name").alias("cruise"))
    if "number" in casts_df.columns:
        expressions.append(pl.col("number").alias("cast"))
    if "start_time" in casts_df.columns:
        expressions.append(pl.col("start_time").alias("date"))
    if "depsm" in casts_df.columns:
        expressions.append(pl.col("depsm").alias("depth"))
    elif "prdm" in casts_df.columns:
        expressions.append(pl.col("prdm").alias("depth"))
    return casts_df.with_columns(expressions) if expressions else casts_df


def endpoint_depth_data(data_df: pl.DataFrame) -> pl.DataFrame:
    """Use the selected endpoint's depth field instead of metadata depth."""
    if data_df.is_empty() or "depsm" not in data_df.columns:
        return data_df
    return data_df.with_columns(pl.col("depsm").alias("depth"))


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


def add_surface_variable(
    casts_df: pl.DataFrame, data_df: pl.DataFrame, variable: str
) -> pl.DataFrame:
    surface_df = surface_observations(data_df, variable)
    if surface_df.is_empty() or not {"cruise_name", "number"}.issubset(
        casts_df.columns
    ):
        return casts_df

    left = casts_df.with_columns(
        [
            pl.col("cruise_name").alias("cruise"),
            pl.col("number").cast(pl.Utf8).alias("cast"),
        ]
    )
    return left.join(surface_df, on=["cruise", "cast"], how="left").drop(
        ["cruise", "cast"]
    )


def shallowest_bottles(
    data_df: pl.DataFrame, casts_df: pl.DataFrame
) -> pl.DataFrame:
    """Return one shallowest bottle per cruise/cast with cast coordinates."""
    required = {"cruise", "cast", "depth"}
    if data_df.is_empty() or not required.issubset(data_df.columns):
        return pl.DataFrame()

    bottles = (
        data_df.drop_nulls(["cruise", "cast", "depth"])
        .with_columns(pl.col("cast").cast(pl.Utf8))
        .sort("depth")
        .group_by(["cruise", "cast"], maintain_order=True)
        .first()
    )
    cast_keys = {"cruise_name", "number", "latitude", "longitude"}
    if not cast_keys.issubset(casts_df.columns):
        return (
            bottles.drop_nulls(["latitude", "longitude"])
            if {"latitude", "longitude"}.issubset(bottles.columns)
            else pl.DataFrame()
        )
    cast_locations = casts_df.select(
        ["cruise_name", "number", "latitude", "longitude"]
    ).rename({"cruise_name": "cruise", "number": "cast"}).with_columns(
        pl.col("cast").cast(pl.Utf8)
    ).unique(["cruise", "cast"])
    if {"latitude", "longitude"}.issubset(bottles.columns):
        bottles = bottles.drop(["latitude", "longitude"])
    return bottles.join(cast_locations, on=["cruise", "cast"], how="left").drop_nulls(
        ["latitude", "longitude"]
    )


def cruises_in_date_range(cruise_df: pl.DataFrame, start: date, end: date) -> list[str]:
    filtered = cruise_df.filter(
        (pl.col("start_time").dt.date() <= end)
        & (pl.col("end_time").dt.date() >= start)
    )
    return filtered.get_column("name").to_list()


@st.cache_data(ttl="24h", max_entries=20, show_spinner=False)
def section_bathymetry(
    x_axis: str, x_min: float, x_max: float, other_center: float
) -> pl.DataFrame:
    """Fetch a narrow GEBCO OPeNDAP profile for a section."""
    if x_axis not in {"latitude", "longitude"}:
        return pl.DataFrame()

    try:
        import xarray as xr

        half_width = 0.25
        latitude_slice = (
            slice(x_min, x_max)
            if x_axis == "latitude"
            else slice(other_center - half_width, other_center + half_width)
        )
        longitude_slice = (
            slice(other_center - half_width, other_center + half_width)
            if x_axis == "latitude"
            else slice(x_min, x_max)
        )
        with xr.open_dataset(
            GEBCO_OPENDAP_URL, engine="pydap", decode_cf=False
        ) as dataset:
            elevation = dataset["elevation"].sel(
                lat=latitude_slice,
                lon=longitude_slice,
            )
            if x_axis == "latitude":
                elevation = elevation.median(dim="lon", skipna=True)
                values = elevation.to_dataframe(name="elevation").reset_index()
                values = values.rename(columns={"lat": "section_x"})
            else:
                elevation = elevation.median(dim="lat", skipna=True)
                values = elevation.to_dataframe(name="elevation").reset_index()
                values = values.rename(columns={"lon": "section_x"})
    except Exception:
        return pl.DataFrame()

    bathymetry = pl.from_pandas(values).select(
        [
            pl.col("section_x").cast(pl.Float64, strict=False).alias(x_axis),
            (-pl.col("elevation")).cast(pl.Float64, strict=False).alias("bathymetry"),
        ]
    )
    return (
        bathymetry.select([x_axis, "bathymetry"])
        .drop_nulls()
        .filter(pl.col("bathymetry") > 0)
        .group_by(x_axis)
        .agg(pl.col("bathymetry").median())
        .sort(x_axis)
        .with_columns(pl.lit(0.0).alias("surface"))
    )


@st.cache_data(ttl="1h", max_entries=30, show_spinner=False)
def interpolate_section(
    section_df: pl.DataFrame, x_axis: str, variable: str
) -> pl.DataFrame:
    """Locally grid section observations using an ODV-like weighted average."""
    if x_axis not in {"latitude", "longitude"}:
        return pl.DataFrame()

    try:
        import numpy as np
        from scipy.spatial import cKDTree

        profile_keys = [key for key in ["cruise", "cast"] if key in section_df.columns]
        if not profile_keys:
            return pl.DataFrame()
        columns = profile_keys + [x_axis, "depth", variable]
        sample = section_df.select(columns).drop_nulls().to_pandas()
        if sample.empty:
            return pl.DataFrame()

        profiles = []
        for _, profile in sample.groupby(profile_keys, dropna=True):
            x_value = float(profile[x_axis].median())
            profile = (
                profile.groupby("depth", as_index=False)[variable]
                .mean()
                .sort_values("depth")
            )
            if len(profile) < 2:
                continue
            profiles.append((x_value, profile["depth"].to_numpy(), profile[variable].to_numpy()))
        if len(profiles) < 2:
            return pl.DataFrame()

        profile_x = np.array([profile[0] for profile in profiles])
        depth_min = min(profile[1].min() for profile in profiles)
        depth_max = max(profile[1].max() for profile in profiles)
        x_min, x_max = profile_x.min(), profile_x.max()
        if x_min == x_max or depth_min == depth_max:
            return pl.DataFrame()

        x_grid = np.linspace(x_min, x_max, 120)
        depth_grid = np.linspace(depth_min, depth_max, 100)
        profile_values = np.full((len(profiles), len(depth_grid)), np.nan)
        for row, (_, depths, values) in enumerate(profiles):
            # No vertical extrapolation: values exist only between observations
            # within the same cast.
            valid_depths = (depth_grid >= depths.min()) & (depth_grid <= depths.max())
            profile_values[row, valid_depths] = np.interp(
                depth_grid[valid_depths], depths, values
            )

        # At each depth, use nearby cast profiles only. This avoids diagonal
        # blending between unrelated shallow and deep observations.
        radius = max((x_max - x_min) * 0.18, 1e-9)
        tree = cKDTree(profile_x[:, None])
        neighbors = min(8, len(profiles))
        distances, indices = tree.query(
            x_grid[:, None], k=neighbors, distance_upper_bound=radius, workers=-1
        )
        if neighbors == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        valid_neighbors = indices < len(profiles)
        safe_indices = np.minimum(indices, len(profiles) - 1)
        grid_values = np.full((len(depth_grid), len(x_grid)), np.nan)
        for depth_row in range(len(depth_grid)):
            available = valid_neighbors & np.isfinite(profile_values[:, depth_row])[safe_indices]
            weights = np.where(
                available, 1.0 / np.maximum(distances, 1e-9) ** 2, 0.0
            )
            values = np.where(
                available, profile_values[safe_indices, depth_row], 0.0
            )
            weight_sum = weights.sum(axis=1)
            grid_values[depth_row] = np.divide(
                (values * weights).sum(axis=1),
                weight_sum,
                out=np.full(len(x_grid), np.nan),
                where=weight_sum > 0,
            )

        grid_x, grid_depth = np.meshgrid(x_grid, depth_grid)
        valid = np.isfinite(grid_values)
        return pl.DataFrame(
            {
                x_axis: grid_x[valid],
                "depth": grid_depth[valid],
                variable: grid_values[valid],
            }
        )
    except Exception:
        return pl.DataFrame()


def mask_interpolation_by_bathymetry(
    interpolated_df: pl.DataFrame,
    bathymetry_df: pl.DataFrame,
    x_axis: str,
) -> pl.DataFrame:
    """Remove interpolated cells at or below the GEBCO seafloor."""
    if interpolated_df.is_empty() or bathymetry_df.is_empty():
        return pl.DataFrame()

    import numpy as np

    bathymetry = bathymetry_df.sort(x_axis)
    bathy_x = bathymetry.get_column(x_axis).to_numpy()
    bathy_depth = bathymetry.get_column("bathymetry").to_numpy()
    grid_x = interpolated_df.get_column(x_axis).to_numpy()
    bottom = np.interp(
        grid_x,
        bathy_x,
        bathy_depth,
        left=np.nan,
        right=np.nan,
    )
    valid = np.isfinite(bottom) & (
        interpolated_df.get_column("depth").to_numpy() < bottom
    )
    return interpolated_df.filter(pl.Series("valid", valid))


def dataframe_card(
    title: str, df: pl.DataFrame, *, key: str, height: int = 320
) -> None:
    with st.container(border=True):
        st.subheader(title)
        st.dataframe(df, hide_index=True, height=height, width="stretch", key=key)


def render_metrics(
    summary: pl.DataFrame,
    casts_df: pl.DataFrame,
    data_df: pl.DataFrame | None,
    dataset: str,
) -> None:
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


def render_track(
    casts_df: pl.DataFrame,
    data_df: pl.DataFrame,
    bottle_df: pl.DataFrame,
    underway_df: pl.DataFrame,
    dataset: str,
) -> None:
    with st.container(border=True):
        st.subheader("Underway track, stations, and bathymetry")
        if underway_df.is_empty() or not {"latitude", "longitude"}.issubset(
            underway_df.columns
        ):
            st.warning(
                "No underway location data were available for the selected cruise(s)."
            )
            if casts_df.is_empty() or not {"latitude", "longitude"}.issubset(
                casts_df.columns
            ):
                return
            st.info("Showing cast locations as a fallback.")

        underway_options = property_options(underway_df, include_cruise=True)
        bottle_options = property_options(bottle_df) or ["Depth"]
        with st.container(horizontal=True, vertical_alignment="bottom"):
            color_choice = st.selectbox(
                "Underway property",
                underway_options,
                index=sst_property_index(underway_options),
                key="map_surface_variable",
                help="Choose the property used to color underway observations.",
            )
            bottle_property = st.selectbox(
                "Shallowest bottle property",
                bottle_options,
                index=temperature_property_index(bottle_options),
                key="map_bottle_property",
                help="Choose the property used to color shallowest-bottle markers.",
            )
            show_bathymetry = st.toggle(
                "Use ocean bathymetry basemap",
                value=True,
                key="show_bathymetry",
                help="Uses Esri's public Ocean Basemap tiles, which include bathymetric relief.",
            )
            show_shallowest_bottles = st.toggle(
                "Show shallowest bottle per cast",
                value=False,
                key="show_shallowest_bottles",
                help="Overlay the shallowest CTD bottle from each cast at its cast location.",
            )

        using_underway = not underway_df.is_empty() and {"latitude", "longitude"}.issubset(
            underway_df.columns
        )
        track_df = (
            underway_df
            if using_underway
            else casts_df
        )
        track_plot = track_df.drop_nulls(["latitude", "longitude"])
        color_column = (
            "cruise_name" if "cruise_name" in track_plot.columns else "cruise"
        )
        if color_choice != "Cruise":
            if color_choice in track_plot.columns:
                color_column = color_choice
            else:
                color_column = (
                    "cruise_name" if "cruise_name" in track_plot.columns else "cruise"
                )
            if (
                color_choice in track_plot.columns
                and track_plot.get_column(color_choice).drop_nulls().len() > 0
            ):
                pass
            else:
                st.info(
                    f"No underway `{color_choice}` values were available; coloring by cruise instead."
                )

        hover_cols = [
            c
            for c in [
                "cruise",
                "cruise_name",
                "number",
                "cast",
                "start_time",
                "date",
                color_choice,
            ]
            if c in track_plot.columns
        ]
        fig = px.scatter_map(
            track_plot.to_pandas(),
            lat="latitude",
            lon="longitude",
            color=color_column if color_column in track_plot.columns else None,
            hover_data=hover_cols,
            color_continuous_scale="Viridis" if color_column == color_choice else None,
            zoom=6,
            height=700,
            map_style="white-bg" if show_bathymetry else "open-street-map",
        )
        fig.update_traces(
            marker={"size": 6 if using_underway else 11, "opacity": 0.75},
            selector={"mode": "markers"},
        )

        for cruise_name, group in (
            track_plot.sort(
                [c for c in ["cruise_name", "cruise", "start_time", "date"] if c in track_plot.columns]
            )
            .to_pandas()
            .groupby("cruise_name" if "cruise_name" in track_plot.columns else "cruise")
        ):
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

        if show_shallowest_bottles:
            bottle_plot = shallowest_bottles(bottle_df, casts_df)
            if bottle_plot.is_empty():
                st.info("No shallowest-bottle locations were available for this selection.")
            else:
                bottle_hover = [
                    c
                    for c in ["cruise", "cast", "niskin", "depth", bottle_property]
                    if c in bottle_plot.columns
                ]
                fig.add_trace(
                    go.Scattermap(
                        lat=bottle_plot["latitude"].to_list(),
                        lon=bottle_plot["longitude"].to_list(),
                        mode="markers",
                        marker={
                            "size": 13,
                            "color": (
                                bottle_plot[bottle_property].to_list()
                                if bottle_property in bottle_plot.columns
                                else "#ff7f0e"
                            ),
                            "colorscale": "Viridis",
                            "showscale": bottle_property in bottle_plot.columns,
                            "opacity": 0.95,
                            "colorbar": {"title": bottle_property},
                        },
                        name="Shallowest bottle",
                        text=[
                            "<br>".join(
                                f"{column}: {row[column]}" for column in bottle_hover
                            )
                            for row in bottle_plot.select(bottle_hover).to_dicts()
                        ],
                        hovertemplate="%{text}<extra></extra>",
                    )
                )

        map_layout = {
            "margin": dict(l=0, r=0, t=0, b=0),
            "legend": {"orientation": "h"},
        }
        if show_bathymetry:
            map_layout["map"] = {
                "layers": [
                    {
                        "sourcetype": "raster",
                        "source": [OCEAN_BASEMAP_TILE_URL],
                        "sourceattribution": OCEAN_BASEMAP_ATTRIBUTION,
                        "type": "raster",
                        "below": "traces",
                    }
                ]
            }
        fig.update_layout(**map_layout)
        st.caption(
            f"The map defaults to the underway navigation track. The orange overlay shows one shallowest `{dataset}` bottle per cast when enabled. "
            f"Bathymetry is shown with public Esri Ocean Basemap tiles when enabled."
        )
        st.plotly_chart(fig, width="stretch")


def render_section(
    data_df: pl.DataFrame,
    dataset: str,
    selected: list[str],
) -> None:
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
            x_options = [
                c for c in ["latitude", "longitude", "cast"] if c in data_df.columns
            ]
            x_axis = st.selectbox("Section x-axis", x_options, key="section_x_axis")
            variables = numeric_columns(data_df, exclude=PLOT_EXCLUDE_COLUMNS)
            if not variables:
                st.warning(
                    "The selected source has no numeric property to plot."
                )
                return
            variable = st.selectbox("Variable", variables, key="section_variable")
            cruise_filter = st.multiselect(
                "Cruises in section", selected, default=selected, key="section_cruises"
            )
            interpolate = st.toggle(
                "Interpolate between points",
                value=False,
                key="section_interpolate",
                help="Estimate values between observations, masked to the GEBCO seafloor.",
            )
            show_bathymetry = st.toggle(
                "Show filled bathymetry",
                value=True,
                key="section_bathymetry",
                help="Fill the seafloor below the section using underway bathymetry data.",
            )

        section_df = (
            data_df.filter(pl.col("cruise").is_in(cruise_filter))
            if "cruise" in data_df.columns
            else data_df
        )
        section_df = section_df.drop_nulls([x_axis, "depth", variable])
        if section_df.is_empty():
            st.info("No rows remain after applying the section filters.")
            return

        section_bathy_df = pl.DataFrame()
        if show_bathymetry or interpolate:
            other_axis = "longitude" if x_axis == "latitude" else "latitude"
            if x_axis in {"latitude", "longitude"} and other_axis in section_df.columns:
                bathymetry_extent = section_df.select([x_axis, other_axis]).drop_nulls()
                if not bathymetry_extent.is_empty():
                    section_bathy_df = section_bathymetry(
                        x_axis,
                        float(bathymetry_extent.get_column(x_axis).min()),
                        float(bathymetry_extent.get_column(x_axis).max()),
                        float(bathymetry_extent.get_column(other_axis).median()),
                    )

        section_plot_bottom = float(section_df.get_column("depth").max())
        if not section_bathy_df.is_empty():
            section_plot_bottom = max(
                section_plot_bottom,
                float(section_bathy_df.get_column("bathymetry").max()),
            )
        section_plot_bottom = max(section_plot_bottom * 1.02, 1.0)
        y_scale = alt.Scale(domain=[0, section_plot_bottom])

        points = (
            alt.Chart(section_df.to_pandas())
            .mark_circle(size=70, opacity=0.85)
            .encode(
                x=alt.X(
                    f"{x_axis}:Q" if x_axis != "cast" else f"{x_axis}:N",
                    title=x_axis.replace("_", " ").title(),
                    scale=(
                        alt.Scale(zero=False)
                        if x_axis in {"latitude", "longitude"}
                        else alt.Undefined
                    ),
                ),
                y=alt.Y(
                    "depth:Q",
                    title="Depth (m)",
                    sort="descending",
                    scale=y_scale,
                ),
                color=alt.Color(
                    f"{variable}:Q",
                    title=variable.replace("_", " ").title(),
                    scale=alt.Scale(scheme="viridis"),
                ),
                tooltip=[
                    c
                    for c in [
                        "cruise",
                        "cast",
                        "niskin",
                        "date",
                        "latitude",
                        "longitude",
                        "depth",
                        variable,
                    ]
                    if c in section_df.columns
                ],
            )
        )
        layers = []
        if interpolate:
            if section_bathy_df.is_empty():
                st.info("Interpolation requires bathymetry data and was not shown.")
            else:
                interpolated_df = mask_interpolation_by_bathymetry(
                    interpolate_section(section_df, x_axis, variable),
                    section_bathy_df,
                    x_axis,
                )
                if not interpolated_df.is_empty():
                    layers.append(
                        alt.Chart(interpolated_df.to_pandas())
                        .mark_rect()
                        .encode(
                            x=alt.X(
                                f"{x_axis}:Q",
                                title=x_axis.replace("_", " ").title(),
                            ),
                            y=alt.Y(
                                "depth:Q",
                                title="Depth (m)",
                                sort="descending",
                                scale=y_scale,
                            ),
                            color=alt.Color(
                                f"{variable}:Q",
                                title=variable.replace("_", " ").title(),
                                scale=alt.Scale(scheme="viridis"),
                            ),
                        )
                    )
                else:
                    st.info("No interpolated cells remained above the bathymetry.")

        layers.append(points)
        if show_bathymetry:
            if not section_bathy_df.is_empty():
                bathymetry = (
                    alt.Chart(
                        section_bathy_df.with_columns(
                            pl.lit(section_plot_bottom).alias("section_bottom")
                        ).to_pandas()
                    )
                    .mark_area(color="#6b7280", opacity=0.35)
                    .encode(
                        x=alt.X(
                            f"{x_axis}:Q",
                            title=x_axis.replace("_", " ").title(),
                            scale=alt.Scale(zero=False),
                        ),
                        y=alt.Y(
                            "bathymetry:Q",
                            title="Depth (m)",
                            sort="descending",
                            scale=y_scale,
                        ),
                        y2="section_bottom:Q",
                        tooltip=[
                            alt.Tooltip(f"{x_axis}:Q", title=x_axis.title()),
                            alt.Tooltip("bathymetry:Q", title="Bathymetry (m)"),
                        ],
                    )
                )
                layers.insert(0, bathymetry)
            else:
                st.info("No raster bathymetry data were available for this section.")

        chart = alt.layer(*layers).interactive().properties(height=620)
        st.altair_chart(chart, width="stretch")


def render_profile(
    data_df: pl.DataFrame,
    dataset: str,
    selected: list[str],
    casts_df: pl.DataFrame,
    bottle_df: pl.DataFrame,
) -> None:
    with st.container(border=True):
        st.subheader(f"Single-station profiles from {dataset}")
        st.caption(
            "Profile depth uses the selected CTD endpoint's `depsm` field."
            if dataset == "CTD bottles"
            else DEPTH_SOURCE_NOTES[dataset]
        )
        ctd_source = "Bottles"
        if dataset == "CTD bottles":
            ctd_source = st.segmented_control(
                "CTD source",
                ["Bottles", "Casts", "Both"],
                default="Bottles",
                key="profile_ctd_source",
                help=(
                    "Casts uses ctd/cast/{cruise}/{cast}.csv and its `depsm` depth. "
                    "Bottles uses ctd/bottles/{cruise}.csv and its `depsm` depth."
                ),
            )
            cast_data = cast_plot_data(casts_df)
            bottle_data = endpoint_depth_data(bottle_df)
            if ctd_source == "Casts":
                data_df = cast_data
            elif ctd_source == "Both":
                data_df = pl.concat(
                    [cast_data, bottle_data], how="diagonal_relaxed"
                )
            else:
                data_df = bottle_data

        if data_df.is_empty() or "depth" not in data_df.columns:
            st.warning("Load a dataset with depth values to view profiles.")
            return

        profile_cruises = (
            data_df.get_column("cruise").unique().sort().to_list()
            if "cruise" in data_df.columns
            else selected
        )
        with st.container(horizontal=True, vertical_alignment="bottom"):
            profile_cruise = st.selectbox(
                "Profile cruise", profile_cruises, key="profile_cruise"
            )
            subset = (
                data_df.filter(pl.col("cruise") == profile_cruise)
                if "cruise" in data_df.columns
                else data_df
            )
            station_column = next(
                (
                    column
                    for column in ["nearest_station", "station"]
                    if column in subset.columns
                ),
                None,
            )
            selector_options = ["Cast"] + (["Station"] if station_column else [])
            selector_type = (
                st.segmented_control(
                    "Select profile by",
                    selector_options,
                    default="Cast",
                    key="profile_selector_type",
                )
                if len(selector_options) > 1
                else "Cast"
            )
            selector_column = station_column if selector_type == "Station" else "cast"
            selector_label = f"Profile {selector_type.lower()}"
            selector_values = (
                subset.get_column(selector_column)
                .drop_nulls()
                .unique()
                .sort()
                .cast(pl.Utf8)
                .to_list()
                if selector_column in subset.columns
                else []
            )
            if not selector_values:
                st.warning(
                    f"No {selector_type.lower()} values are available for profiles."
                )
                return
            selected_profile = st.selectbox(
                selector_label, selector_values, key="profile_selector_value"
            )
            variables = numeric_columns(subset, exclude=PLOT_EXCLUDE_COLUMNS)
            if not variables:
                st.warning(
                    "The selected source has no numeric profile property."
                )
                return
            profile_var = st.selectbox(
                "Profile variable", variables, key="profile_variable"
            )

        profile_df = subset.filter(
            pl.col(selector_column).cast(pl.Utf8) == str(selected_profile)
        ).drop_nulls(["depth", profile_var])
        if profile_df.is_empty():
            st.info("No profile rows remain after applying the filters.")
            return

        profile_df = profile_df.sort("depth")

        if selector_type == "Station" and "cast" in profile_df.columns:
            color_column = "cast"
        elif "replicate" in profile_df.columns:
            color_column = "replicate"
        else:
            color_column = alt.value("#1f77b4")
        chart = (
            alt.Chart(profile_df.to_pandas())
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    f"{profile_var}:Q", title=profile_var.replace("_", " ").title()
                ),
                y=alt.Y("depth:Q", title="Depth (m)", sort="descending"),
                order=alt.Order("depth:Q", sort="ascending"),
                color=color_column,
                tooltip=[
                    c
                    for c in [
                        "cruise",
                        "cast",
                        "niskin",
                        "replicate",
                        "nearest_station",
                        "station",
                        "date",
                        "latitude",
                        "longitude",
                        "depth",
                        profile_var,
                    ]
                    if c in profile_df.columns
                ],
            )
            .interactive()
            .properties(height=620)
        )
        st.altair_chart(chart, width="stretch")

        endpoint_lines = []
        if dataset == "CTD bottles":
            if ctd_source in {"Bottles", "Both"}:
                endpoint_lines.append(
                    f"{API_BASE}/ctd/bottles/{profile_cruise}.csv"
                )
            if ctd_source in {"Casts", "Both"}:
                cast_id = (
                    str(selected_profile)
                    if selector_type == "Cast"
                    else "{cast}"
                )
                endpoint_lines.append(
                    f"{API_BASE}/ctd/cast/{profile_cruise}/{cast_id}.csv"
                )
        else:
            endpoint_lines.append(
                f"{API_BASE}/{DATASETS[dataset].format(cruise=profile_cruise)}"
            )

        st.subheader("Profile data source")
        st.caption("Endpoint(s) used for this plot:")
        for endpoint in endpoint_lines:
            st.code(endpoint)
        st.caption(f"Rows used for this plot: {profile_df.height:,}")
        st.dataframe(
            profile_df,
            hide_index=True,
            width="stretch",
            key="profile_plot_data_table",
        )


def render_data(
    casts_df: pl.DataFrame,
    data_df: pl.DataFrame,
    bottle_df: pl.DataFrame,
    underway_df: pl.DataFrame,
    dataset: str,
) -> None:
    with st.container(border=True):
        st.subheader("Loaded data")
        datasets = [
            (
                "Underway",
                underway_df,
                None,
                f"{API_BASE}/underway/{{cruise}}.csv",
            ),
            (
                "CTD casts",
                casts_df,
                None,
                f"{API_BASE}/ctd/cast/{{cruise}}/{{cast}}.csv",
            ),
            (
                "CTD bottles",
                bottle_df,
                DEPTH_SOURCE_NOTES["CTD bottles"],
                f"{API_BASE}/ctd/bottles/{{cruise}}.csv",
            ),
        ]
        if dataset != "CTD bottles":
            datasets.append(
                (
                    dataset,
                    data_df,
                    DEPTH_SOURCE_NOTES[dataset],
                    f"{API_BASE}/{DATASETS[dataset]}",
                )
            )

        tabs = st.tabs([name for name, _, _, _ in datasets])
        for tab, (name, table, note, endpoint) in zip(tabs, datasets):
            with tab:
                st.write(f"{name}: {table.height:,} rows")
                st.caption(f"API endpoint: {endpoint}")
                if note:
                    st.caption(note)
                if table.is_empty():
                    st.info(f"No {name.lower()} data were available for the selected cruise(s).")
                else:
                    st.dataframe(
                        table,
                        hide_index=True,
                        height=650,
                        width="stretch",
                        key=f"loaded_data_{name.lower().replace(' ', '_')}"
                    )


st.title("NES-LTER API dashboard")
st.caption(
    "Explore cruise tracks, depth sections, and single-station profiles from the NES-LTER API."
)

try:
    cruise_df = load_cruises()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.header("Filters")
    selection_mode = st.segmented_control(
        "Select by", ["Cruises", "Date range"], default="Cruises"
    )

    cruise_names = cruise_df.get_column("name").to_list()
    if selection_mode == "Cruises":
        default = [name for name in ["EN617"] if name in cruise_names] or cruise_names[
            -1:
        ]
        selected = st.multiselect(
            "Cruises", cruise_names, default=default, key="selected_cruises"
        )
    else:
        min_dt = cruise_df.get_column("start_time").drop_nulls().min()
        max_dt = cruise_df.get_column("end_time").drop_nulls().max()
        date_range = st.date_input(
            "Date range",
            value=(
                min_dt.date() if min_dt else date(2017, 1, 1),
                max_dt.date() if max_dt else date.today(),
            ),
            key="selected_date_range",
        )
        if len(date_range) != 2:
            st.stop()
        selected = cruises_in_date_range(cruise_df, *date_range)
        st.caption(f"{len(selected)} cruise(s) overlap this range.")

    dataset = st.selectbox("Section/profile dataset", list(DATASETS), key="dataset")

if not selected:
    st.info("Select at least one cruise to begin.")
    st.stop()

selected_tuple = tuple(selected)
summary = cruise_df.filter(pl.col("name").is_in(selected))

with st.skeleton(height=220):
    casts_df = load_casts(selected_tuple)

with st.skeleton(height=220):
    underway_df = load_underway(selected_tuple)

with st.skeleton(height=220):
    data_df = load_dataset(selected_tuple, dataset)

with st.skeleton(height=220):
    bottle_df = (
        data_df
        if dataset == "CTD bottles"
        else load_dataset(selected_tuple, "CTD bottles")
    )

render_metrics(summary, casts_df, data_df, dataset)

view = st.segmented_control(
    "View",
    ["Cruise track", "Sections", "Profiles", "Data"],
    default="Cruise track",
    key="view",
)

if view == "Cruise track":
    render_track(
        casts_df,
        data_df if data_df is not None else pl.DataFrame(),
        bottle_df if bottle_df is not None else pl.DataFrame(),
        underway_df if underway_df is not None else pl.DataFrame(),
        dataset,
    )
elif view == "Sections":
    render_section(
        data_df if data_df is not None else pl.DataFrame(),
        dataset,
        selected,
    )
elif view == "Profiles":
    render_profile(
        data_df if data_df is not None else pl.DataFrame(),
        dataset,
        selected,
        casts_df,
        bottle_df,
    )
elif view == "Data":
    render_data(
        casts_df,
        data_df if data_df is not None else pl.DataFrame(),
        bottle_df if bottle_df is not None else pl.DataFrame(),
        underway_df if underway_df is not None else pl.DataFrame(),
        dataset,
    )

with st.expander("Selected cruises"):
    st.dataframe(
        summary, hide_index=True, width="stretch", key="selected_cruises_table"
    )
