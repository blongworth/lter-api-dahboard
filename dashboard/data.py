from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from math import asin, cos, radians, sin, sqrt
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import polars as pl
import streamlit as st

from .config import (
    API_BASE,
    DATASET_SPECS,
    DATETIME_COLUMNS,
    IDENTIFIER_COLUMNS,
    PLOT_EXCLUDE_COLUMNS,
    UNDERWAY_ENDPOINT,
)

LOGGER = logging.getLogger(__name__)


@st.cache_data(ttl="1h", max_entries=150, show_spinner=False)
def fetch_csv(path: str) -> pl.DataFrame:
    """Fetch and parse one API CSV endpoint."""
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


@st.cache_data(ttl="24h", max_entries=100, show_spinner=False)
def fetch_text(path: str) -> str:
    """Fetch a text or Markdown API endpoint."""
    url = f"{API_BASE}/{path.lstrip('/')}"
    request = Request(url, headers={"User-Agent": "nes-lter-streamlit-dashboard/0.1"})
    try:
        with urlopen(request, timeout=45) as response:
            return response.read().decode("utf-8")
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not fetch {url}: {exc}") from exc


@st.cache_data(ttl="24h", max_entries=30, show_spinner=False)
def load_ctd_metadata(cruise_names: tuple[str, ...]) -> pl.DataFrame:
    frames = []
    for cruise in cruise_names:
        try:
            metadata = normalize_dataframe(
                fetch_csv(f"ctd/metadata/{quote(cruise, safe='')}.csv")
            )
            if not metadata.is_empty():
                frames.append(metadata.with_columns(pl.lit(cruise).alias("cruise")))
        except RuntimeError:
            LOGGER.warning("Unable to load CTD metadata for %s", cruise, exc_info=True)
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


@st.cache_data(ttl="24h", max_entries=30, show_spinner=False)
def load_underway_definitions(cruise_names: tuple[str, ...]) -> pl.DataFrame:
    frames = []
    for cruise in cruise_names:
        try:
            definitions = normalize_dataframe(
                fetch_csv(f"underway/column_definition/{quote(cruise, safe='')}.csv")
            )
            if not definitions.is_empty():
                frames.append(definitions.with_columns(pl.lit(cruise).alias("cruise")))
        except RuntimeError:
            LOGGER.warning(
                "Unable to load underway definitions for %s", cruise, exc_info=True
            )
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


@st.cache_data(ttl="1h", max_entries=30, show_spinner=False)
def load_raw_underway(cruise_names: tuple[str, ...]) -> pl.DataFrame:
    """Load the raw rows returned by each underway CSV endpoint."""
    frames = []
    for cruise in cruise_names:
        try:
            raw = fetch_csv(f"underway/{quote(cruise, safe='')}.csv")
            if not raw.is_empty():
                frames.append(raw.with_columns(pl.lit(cruise).alias("cruise")))
        except RuntimeError:
            LOGGER.warning(
                "Unable to load raw underway data for %s", cruise, exc_info=True
            )
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


@st.cache_data(ttl="24h", max_entries=30, show_spinner=False)
def load_cruise_readme(cruise: str) -> str:
    return fetch_text(f"ctd/cruises/readme/{quote(cruise, safe='')}")


@st.cache_data(ttl="24h", max_entries=10, show_spinner=False)
def load_dataset_readme(dataset: str) -> str:
    path = {
        "CTD bottles": "ctd/cruises/readme",
        "Nutrients": "nut/readme",
        "Chlorophyll": "chl/readme",
    }.get(dataset)
    if path is None:
        return ""
    return fetch_text(path)


def normalize_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Apply endpoint-independent datetime, numeric, and identifier types."""
    expressions = []
    for name, dtype in df.schema.items():
        if name in DATETIME_COLUMNS and dtype == pl.String:
            expressions.append(
                pl.col(name).str.to_datetime(strict=False, time_zone="UTC").alias(name)
            )
        elif name not in IDENTIFIER_COLUMNS and dtype == pl.String:
            original = df.get_column(name)
            numeric = original.cast(pl.Float64, strict=False)
            if numeric.null_count() == original.null_count():
                expressions.append(pl.col(name).cast(pl.Float64).alias(name))
    normalized = df.with_columns(expressions) if expressions else df
    for name in ["cast", "number", "niskin"]:
        if name not in normalized.columns or normalized.schema[name] != pl.String:
            continue
        original = normalized.get_column(name)
        numeric = original.cast(pl.Int64, strict=False)
        if numeric.null_count() == original.null_count():
            normalized = normalized.with_columns(numeric.alias(name))
    return normalized


def numeric_columns(df: pl.DataFrame, exclude: set[str] | None = None) -> list[str]:
    excluded = exclude or set()
    return [
        name
        for name, dtype in df.schema.items()
        if name not in excluded and dtype.is_numeric()
    ]


def normalize_underway_coordinates(df: pl.DataFrame) -> pl.DataFrame:
    aliases = {
        "latitude": [
            "latitude",
            "dec_lat",
            "latitude_deg",
            "gps_furuno_latitude",
            "gps_garmin741_latitude",
            "gps_nstarwaas_latitude",
            "gnss_adu2_latitude",
            "gnss_adu5_latitude",
        ],
        "longitude": [
            "longitude",
            "dec_lon",
            "longitude_deg",
            "gps_furuno_longitude",
            "gps_garmin741_longitude",
            "gps_nstarwaas_longitude",
            "gnss_adu2_longitude",
            "gnss_adu5_longitude",
        ],
    }
    expressions = []
    for canonical, candidates in aliases.items():
        available = [name for name in candidates if name in df.columns]
        if available:
            expressions.append(
                pl.coalesce(
                    [pl.col(name).cast(pl.Float64, strict=False) for name in available]
                ).alias(canonical)
            )
    return df.with_columns(expressions) if expressions else df


def add_temperature_property(df: pl.DataFrame, *, bottle: bool = False) -> pl.DataFrame:
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
    available = [name for name in candidates if name in df.columns]
    if not available:
        return df
    value = pl.coalesce(
        [pl.col(name).cast(pl.Float64, strict=False) for name in available]
    )
    expressions = [value.alias("temperature")]
    if not bottle:
        expressions.append(value.alias("sst"))
    return df.with_columns(expressions)


def property_options(df: pl.DataFrame, *, include_cruise: bool = False) -> list[str]:
    options = (
        numeric_columns(df, exclude=PLOT_EXCLUDE_COLUMNS) if not df.is_empty() else []
    )
    return (["Cruise"] if include_cruise else []) + options


def temperature_property_index(options: list[str]) -> int:
    candidates = [
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
        "temp",
    ]
    return next((options.index(name) for name in candidates if name in options), 0)


def sst_property_index(options: list[str]) -> int:
    return (
        options.index("sst")
        if "sst" in options
        else temperature_property_index(options)
    )


@st.cache_data(ttl="1h", max_entries=5, show_spinner=False)
def load_cruises() -> pl.DataFrame:
    return normalize_dataframe(fetch_csv("ctd/cruises/all.csv")).sort("start_time")


@st.cache_data(ttl="1h", max_entries=100, show_spinner=False)
def load_cast_detail(cruise: str, cast: str) -> pl.DataFrame:
    """Load one sensor profile from ctd/cast/{cruise}/{cast}.csv."""
    detail = normalize_dataframe(
        fetch_csv(f"ctd/cast/{quote(cruise, safe='')}/{quote(cast, safe='')}.csv")
    )
    if detail.is_empty():
        return detail
    if "cruise" not in detail.columns:
        detail = detail.with_columns(pl.lit(cruise).alias("cruise"))
    if "cast" not in detail.columns:
        detail = detail.with_columns(pl.lit(cast).alias("cast"))
    return detail.with_columns(
        [
            pl.col("cruise").alias("cruise_name"),
            pl.col("cast").cast(pl.Utf8).alias("number"),
        ]
    )


@st.cache_data(ttl="1h", max_entries=30, show_spinner=False)
def load_casts(cruise_names: tuple[str, ...]) -> pl.DataFrame:
    """Load per-cast sensor rows and attach cast-index location metadata."""
    frames: list[pl.DataFrame] = []
    for cruise in cruise_names:
        try:
            cast_index = normalize_dataframe(
                fetch_csv(f"ctd/casts/{quote(cruise, safe='')}.csv")
            )
        except RuntimeError:
            LOGGER.warning("Unable to load cast index for %s", cruise, exc_info=True)
            continue
        if "number" not in cast_index.columns:
            continue
        metadata_columns = [
            name
            for name in [
                "cruise_name",
                "number",
                "latitude",
                "longitude",
                "depth",
                "start_time",
                "end_time",
            ]
            if name in cast_index.columns
        ]
        metadata = (
            cast_index.select(metadata_columns)
            .with_columns(pl.col("number").cast(pl.Utf8))
            .unique(["number"])
        )
        cast_ids = (
            cast_index.get_column("number")
            .drop_nulls()
            .cast(pl.Utf8)
            .unique()
            .to_list()
        )
        with ThreadPoolExecutor(max_workers=min(8, max(1, len(cast_ids)))) as executor:
            details = list(
                executor.map(
                    lambda cast_id, cruise_name=cruise: _safe_cast_detail(
                        cruise_name, cast_id
                    ),
                    cast_ids,
                )
            )
        for detail in details:
            if detail is not None and not detail.is_empty():
                frames.append(
                    detail.join(metadata, on=["cruise_name", "number"], how="left")
                )
    return (
        normalize_dataframe(pl.concat(frames, how="diagonal_relaxed"))
        if frames
        else pl.DataFrame()
    )


def _safe_cast_detail(cruise: str, cast: str) -> pl.DataFrame | None:
    try:
        return load_cast_detail(cruise, cast)
    except RuntimeError:
        LOGGER.warning("Unable to load cast %s/%s", cruise, cast, exc_info=True)
        return None


@st.cache_data(ttl="1h", max_entries=30, show_spinner=False)
def load_underway(cruise_names: tuple[str, ...]) -> pl.DataFrame:
    frames = []
    for cruise in cruise_names:
        try:
            df = normalize_underway_coordinates(
                normalize_dataframe(
                    fetch_csv(UNDERWAY_ENDPOINT.format(cruise=quote(cruise, safe="")))
                )
            )
            df = add_temperature_property(df)
            if "cruise" not in df.columns:
                df = df.with_columns(pl.lit(cruise).alias("cruise"))
            frames.append(df)
        except RuntimeError:
            LOGGER.warning("Unable to load underway data for %s", cruise, exc_info=True)
    return (
        normalize_dataframe(pl.concat(frames, how="diagonal_relaxed"))
        if frames
        else pl.DataFrame()
    )


def use_endpoint_depth(df: pl.DataFrame, depth_field: str | None) -> pl.DataFrame:
    if df.is_empty() or not depth_field or depth_field not in df.columns:
        return df
    return df.with_columns(pl.col(depth_field).alias("depth"))


@st.cache_data(ttl="24h", max_entries=1, show_spinner=False)
def load_stations() -> pl.DataFrame:
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
    if isinstance(value, (datetime, date)):
        return value.date() if isinstance(value, datetime) else value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        except ValueError:
            return None
    return None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    )
    return 2 * 6371.0 * asin(sqrt(a))


def add_nearest_station(df: pl.DataFrame) -> pl.DataFrame:
    if (
        df.is_empty()
        or "nearest_station" in df.columns
        or not {"latitude", "longitude"}.issubset(df.columns)
    ):
        return df
    stations = load_stations().drop_nulls(
        ["station", "station_latitude", "station_longitude"]
    )
    if stations.is_empty():
        return df
    records = stations.select(
        ["station", "station_latitude", "station_longitude", "start_date", "end_date"]
    ).to_dicts()
    date_column = "date" if "date" in df.columns else None
    cache: dict[tuple[float, float, date | None], tuple[str | None, float | None]] = {}
    names, distances = [], []
    for row in df.select(
        ["latitude", "longitude"] + ([date_column] if date_column else [])
    ).iter_rows(named=True):
        if row["latitude"] is None or row["longitude"] is None:
            names.append(None)
            distances.append(None)
            continue
        obs_date = observation_date(row[date_column]) if date_column else None
        key = (
            round(float(row["latitude"]), 5),
            round(float(row["longitude"]), 5),
            obs_date,
        )
        if key not in cache:
            candidates = [
                (
                    s["station"],
                    haversine_km(
                        float(row["latitude"]),
                        float(row["longitude"]),
                        s["station_latitude"],
                        s["station_longitude"],
                    ),
                )
                for s in records
                if not obs_date
                or (not s["start_date"] or obs_date >= s["start_date"])
                and (not s["end_date"] or obs_date <= s["end_date"])
            ]
            cache[key] = (
                min(candidates, key=lambda item: item[1])
                if candidates
                else (None, None)
            )
        name, distance = cache[key]
        names.append(name)
        distances.append(distance)
    return df.with_columns(
        [
            pl.Series("nearest_station", names),
            pl.Series("nearest_station_distance_km", distances),
        ]
    )


@st.cache_data(ttl="1h", max_entries=30, show_spinner=False)
def load_dataset(cruise_names: tuple[str, ...], dataset: str) -> pl.DataFrame:
    spec = DATASET_SPECS[dataset]
    frames = []
    for cruise in cruise_names:
        try:
            df = normalize_dataframe(
                fetch_csv(spec.endpoint.format(cruise=quote(cruise, safe="")))
            )
            df = use_endpoint_depth(df, spec.depth_field)
            if dataset == "CTD bottles":
                df = add_temperature_property(df, bottle=True)
            frames.append(add_nearest_station(df))
        except RuntimeError:
            LOGGER.warning("Unable to load %s for %s", dataset, cruise, exc_info=True)
    return (
        normalize_dataframe(pl.concat(frames, how="diagonal_relaxed"))
        if frames
        else pl.DataFrame()
    )


def load_datasets(
    cruise_names: tuple[str, ...], datasets: tuple[str, ...]
) -> pl.DataFrame:
    """Load and combine multiple compatible depth-oriented datasets."""
    frames = []
    for dataset in datasets:
        data = load_dataset(cruise_names, dataset)
        if not data.is_empty():
            frames.append(data.with_columns(pl.lit(dataset).alias("data_source")))
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def cast_plot_data(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    expressions = []
    if "cruise_name" in df.columns:
        expressions.append(pl.col("cruise_name").alias("cruise"))
    if "number" in df.columns:
        expressions.append(pl.col("number").alias("cast"))
    if "start_time" in df.columns:
        expressions.append(pl.col("start_time").alias("date"))
    if "depsm" in df.columns:
        expressions.append(pl.col("depsm").alias("depth"))
    elif "prdm" in df.columns:
        expressions.append(pl.col("prdm").alias("depth"))
    return df.with_columns(expressions) if expressions else df


def shallowest_bottles(data_df: pl.DataFrame, casts_df: pl.DataFrame) -> pl.DataFrame:
    required = {"cruise", "cast", "depth"}
    if data_df.is_empty() or not required.issubset(data_df.columns):
        return pl.DataFrame()
    bottles = (
        data_df.drop_nulls(list(required))
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
    locations = (
        casts_df.select(["cruise_name", "number", "latitude", "longitude"])
        .rename({"cruise_name": "cruise", "number": "cast"})
        .with_columns(pl.col("cast").cast(pl.Utf8))
        .unique(["cruise", "cast"])
    )
    if {"latitude", "longitude"}.issubset(bottles.columns):
        bottles = bottles.drop(["latitude", "longitude"])
    return bottles.join(locations, on=["cruise", "cast"], how="left").drop_nulls(
        ["latitude", "longitude"]
    )


def cruises_in_date_range(cruises: pl.DataFrame, start: date, end: date) -> list[str]:
    return (
        cruises.filter(
            (pl.col("start_time").dt.date() <= end)
            & (pl.col("end_time").dt.date() >= start)
        )
        .get_column("name")
        .to_list()
    )
