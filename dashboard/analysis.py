from __future__ import annotations

import logging

import numpy as np
import polars as pl
import streamlit as st
from scipy.spatial import cKDTree

from .config import GEBCO_OPENDAP_URL

LOGGER = logging.getLogger(__name__)


@st.cache_data(ttl="24h", max_entries=20, show_spinner=False)
def section_bathymetry(
    x_axis: str, x_min: float, x_max: float, other_center: float
) -> pl.DataFrame:
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
                lat=latitude_slice, lon=longitude_slice
            )
            if x_axis == "latitude":
                profile = elevation.median(dim="lon", skipna=True)
                coordinate = profile["lat"].values
            else:
                profile = elevation.median(dim="lat", skipna=True)
                coordinate = profile["lon"].values
            values = pl.DataFrame(
                {
                    "section_x": np.asarray(coordinate).reshape(-1),
                    "elevation": np.asarray(profile.values).reshape(-1),
                }
            )
        return (
            values.select(
                [
                    pl.col("section_x").cast(pl.Float64).alias(x_axis),
                    (-pl.col("elevation")).cast(pl.Float64).alias("bathymetry"),
                ]
            )
            .drop_nulls()
            .filter(pl.col("bathymetry") > 0)
            .group_by(x_axis)
            .agg(pl.col("bathymetry").median())
            .sort(x_axis)
            .with_columns(pl.lit(0.0).alias("surface"))
        )
    except (OSError, ValueError, KeyError, ImportError) as exc:
        LOGGER.warning("Unable to load section bathymetry: %s", exc)
        return pl.DataFrame()


@st.cache_data(ttl="1h", max_entries=30, show_spinner=False)
def interpolate_section(
    section_df: pl.DataFrame, x_axis: str, variable: str
) -> pl.DataFrame:
    """Interpolate vertically within casts, then locally between casts."""
    if x_axis not in {"latitude", "longitude"}:
        return pl.DataFrame()
    try:
        keys = [key for key in ["cruise", "cast"] if key in section_df.columns]
        if not keys:
            return pl.DataFrame()
        sample = section_df.select(keys + [x_axis, "depth", variable]).drop_nulls()
        if sample.is_empty():
            return pl.DataFrame()
        profiles = []
        for profile in sample.partition_by(keys, maintain_order=True):
            x_value = float(profile[x_axis].median())
            profile = (
                profile.group_by("depth").agg(pl.col(variable).mean()).sort("depth")
            )
            if len(profile) >= 2:
                profiles.append(
                    (x_value, profile["depth"].to_numpy(), profile[variable].to_numpy())
                )
        if len(profiles) < 2:
            return pl.DataFrame()
        profile_x = np.array([item[0] for item in profiles])
        depth_min = min(item[1].min() for item in profiles)
        depth_max = max(item[1].max() for item in profiles)
        x_min, x_max = profile_x.min(), profile_x.max()
        if x_min == x_max or depth_min == depth_max:
            return pl.DataFrame()
        x_grid, depth_grid = (
            np.linspace(x_min, x_max, 120),
            np.linspace(depth_min, depth_max, 100),
        )
        values = np.full((len(profiles), len(depth_grid)), np.nan)
        for row, (_, depths, profile_values) in enumerate(profiles):
            valid = (depth_grid >= depths.min()) & (depth_grid <= depths.max())
            values[row, valid] = np.interp(depth_grid[valid], depths, profile_values)
        tree = cKDTree(profile_x[:, None])
        neighbors = min(8, len(profiles))
        distances, indices = tree.query(
            x_grid[:, None],
            k=neighbors,
            distance_upper_bound=max((x_max - x_min) * 0.18, 1e-9),
            workers=-1,
        )
        if neighbors == 1:
            distances, indices = distances[:, None], indices[:, None]
        valid_neighbors = indices < len(profiles)
        safe_indices = np.minimum(indices, len(profiles) - 1)
        grid = np.full((len(depth_grid), len(x_grid)), np.nan)
        for depth_row in range(len(depth_grid)):
            available = valid_neighbors & np.isfinite(
                values[:, depth_row][safe_indices]
            )
            weights = np.where(available, 1.0 / np.maximum(distances, 1e-9) ** 2, 0.0)
            weight_sum = weights.sum(axis=1)
            grid[depth_row] = np.divide(
                (
                    np.where(available, values[safe_indices, depth_row], 0.0) * weights
                ).sum(axis=1),
                weight_sum,
                out=np.full(len(x_grid), np.nan),
                where=weight_sum > 0,
            )
        grid_x, grid_depth = np.meshgrid(x_grid, depth_grid)
        valid = np.isfinite(grid)
        return pl.DataFrame(
            {x_axis: grid_x[valid], "depth": grid_depth[valid], variable: grid[valid]}
        )
    except (ValueError, KeyError, IndexError) as exc:
        LOGGER.warning("Unable to interpolate section: %s", exc)
        return pl.DataFrame()


def mask_interpolation_by_bathymetry(
    interpolated_df: pl.DataFrame, bathymetry_df: pl.DataFrame, x_axis: str
) -> pl.DataFrame:
    if interpolated_df.is_empty() or bathymetry_df.is_empty():
        return pl.DataFrame()
    bathy = bathymetry_df.sort(x_axis)
    bottom = np.interp(
        interpolated_df.get_column(x_axis).to_numpy(),
        bathy.get_column(x_axis).to_numpy(),
        bathy.get_column("bathymetry").to_numpy(),
        left=np.nan,
        right=np.nan,
    )
    valid = np.isfinite(bottom) & (
        interpolated_df.get_column("depth").to_numpy() < bottom
    )
    return interpolated_df.filter(pl.Series("valid", valid))
