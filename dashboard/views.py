from __future__ import annotations

import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from .analysis import (
    interpolate_section,
    mask_interpolation_by_bathymetry,
    section_bathymetry,
)
from .config import (
    API_BASE,
    DATASET_SPECS,
    OCEAN_BASEMAP_ATTRIBUTION,
    OCEAN_BASEMAP_TILE_URL,
    PLOT_EXCLUDE_COLUMNS,
)
from .data import (
    cast_plot_data,
    numeric_columns,
    property_options,
    shallowest_bottles,
    sst_property_index,
    temperature_property_index,
    use_endpoint_depth,
)


def render_metrics(
    summary: pl.DataFrame, casts_df: pl.DataFrame, data_df: pl.DataFrame, dataset: str
) -> None:
    date_values = (
        summary.select(["start_time", "end_time"]).drop_nulls().to_dicts()
        if not summary.is_empty()
        else []
    )
    date_label = "Unavailable"
    if date_values:
        start = min(row["start_time"] for row in date_values)
        end = max(row["end_time"] for row in date_values)
        date_label = f"{start:%Y-%m-%d} to {end:%Y-%m-%d}"
    with st.container(horizontal=True):
        st.metric("Cruises", f"{summary.height:,}", border=True)
        st.metric("Casts", f"{casts_df.height:,}", border=True)
        st.metric(f"{dataset} rows", f"{data_df.height:,}", border=True)
        st.metric("Date span", date_label, border=True)


def render_track(
    casts_df: pl.DataFrame,
    bottle_df: pl.DataFrame,
    underway_df: pl.DataFrame,
    dataset: str,
) -> None:
    with st.container(border=True):
        st.subheader("Underway track, stations, and bathymetry")
        valid_underway = not underway_df.is_empty() and {
            "latitude",
            "longitude",
        }.issubset(underway_df.columns)
        if not valid_underway:
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
            )
            bottle_property = st.selectbox(
                "Shallowest bottle property",
                bottle_options,
                index=temperature_property_index(bottle_options),
                key="map_bottle_property",
            )
            show_bathymetry = st.toggle(
                "Use ocean bathymetry basemap", True, key="show_bathymetry"
            )
            show_bottles = st.toggle(
                "Show shallowest bottle per cast", False, key="show_shallowest_bottles"
            )
        track = underway_df if valid_underway else casts_df
        track = track.drop_nulls(["latitude", "longitude"])
        color_column = "cruise_name" if "cruise_name" in track.columns else "cruise"
        if color_choice != "Cruise" and color_choice in track.columns:
            color_column = color_choice
        fig = px.scatter_map(
            track,
            lat="latitude",
            lon="longitude",
            color=color_column if color_column in track.columns else None,
            hover_data=[
                c
                for c in [
                    "cruise",
                    "cruise_name",
                    "number",
                    "cast",
                    "date",
                    color_choice,
                ]
                if c in track.columns
            ],
            color_continuous_scale="Viridis" if color_column == color_choice else None,
            zoom=6,
            height=700,
            map_style="white-bg" if show_bathymetry else "open-street-map",
        )
        fig.update_traces(
            marker={"size": 6 if valid_underway else 11, "opacity": 0.75},
            selector={"mode": "markers"},
        )
        group_column = "cruise_name" if "cruise_name" in track.columns else "cruise"
        if group_column in track.columns:
            for group in track.partition_by(group_column, maintain_order=True):
                name = group[group_column][0]
                fig.add_trace(
                    go.Scattermap(
                        lat=group["latitude"].to_list(),
                        lon=group["longitude"].to_list(),
                        mode="lines",
                        line={"color": "rgba(35,35,35,0.35)", "width": 2},
                        name=f"{name} track",
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
        if show_bottles:
            bottle_plot = shallowest_bottles(bottle_df, casts_df)
            if not bottle_plot.is_empty():
                hover_columns = [
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
                            "color": bottle_plot[bottle_property].to_list()
                            if bottle_property in bottle_plot.columns
                            else "#ff7f0e",
                            "colorscale": "Viridis",
                            "showscale": bottle_property in bottle_plot.columns,
                            "opacity": 0.95,
                            "colorbar": {"title": bottle_property},
                        },
                        name="Shallowest bottle",
                        text=[
                            "<br>".join(f"{c}: {row[c]}" for c in hover_columns)
                            for row in bottle_plot.select(hover_columns).to_dicts()
                        ],
                        hovertemplate="%{text}<extra></extra>",
                    )
                )
        if show_bathymetry:
            fig.update_layout(
                map={
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
            )
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0}, legend={"orientation": "h"}
        )
        st.plotly_chart(fig, width="stretch")


def render_section(data_df: pl.DataFrame, dataset: str, selected: list[str]) -> None:
    with st.container(border=True):
        st.subheader(f"Sections from {dataset}")
        st.caption(DATASET_SPECS[dataset].depth_note)
        if data_df.is_empty() or "depth" not in data_df.columns:
            st.warning(f"No {dataset} data with depth were available.")
            return
        with st.container(horizontal=True, vertical_alignment="bottom"):
            x_options = [
                column
                for column in ["latitude", "longitude", "cast"]
                if column in data_df.columns
            ]
            x_axis = st.selectbox("Section x-axis", x_options, key="section_x_axis")
            variables = numeric_columns(data_df, exclude=PLOT_EXCLUDE_COLUMNS)
            if not variables:
                st.warning("The selected source has no numeric property to plot.")
                return
            variable = st.selectbox("Variable", variables, key="section_variable")
            cruises = st.multiselect(
                "Cruises in section", selected, default=selected, key="section_cruises"
            )
            interpolate = st.toggle(
                "Interpolate between points",
                False,
                key="section_interpolate",
                help="Interpolate within casts and locally between casts, without extrapolation.",
            )
            show_bathymetry = st.toggle(
                "Show filled bathymetry", True, key="section_bathymetry"
            )
        section_df = (
            data_df.filter(pl.col("cruise").is_in(cruises))
            if "cruise" in data_df.columns
            else data_df
        )
        section_df = section_df.drop_nulls([x_axis, "depth", variable])
        if section_df.is_empty():
            st.info("No rows remain after applying the section filters.")
            return
        bathy = pl.DataFrame()
        if show_bathymetry or interpolate:
            other_axis = "longitude" if x_axis == "latitude" else "latitude"
            if x_axis in {"latitude", "longitude"} and other_axis in section_df.columns:
                extent = section_df.select([x_axis, other_axis]).drop_nulls()
                if not extent.is_empty():
                    bathy = section_bathymetry(
                        x_axis,
                        float(extent[x_axis].min()),
                        float(extent[x_axis].max()),
                        float(extent[other_axis].median()),
                    )
        bottom = float(section_df["depth"].max())
        if not bathy.is_empty():
            bottom = max(bottom, float(bathy["bathymetry"].max()))
        y_scale = alt.Scale(domain=[0, max(bottom * 1.02, 1.0)])
        points = (
            alt.Chart(section_df)
            .mark_circle(size=70, opacity=0.85)
            .encode(
                x=alt.X(
                    f"{x_axis}:{'N' if x_axis == 'cast' else 'Q'}", title=x_axis.title()
                ),
                y=alt.Y("depth:Q", title="Depth (m)", sort="descending", scale=y_scale),
                color=alt.Color(f"{variable}:Q", scale=alt.Scale(scheme="viridis")),
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
        if interpolate and not bathy.is_empty():
            interpolated = mask_interpolation_by_bathymetry(
                interpolate_section(section_df, x_axis, variable), bathy, x_axis
            )
            if not interpolated.is_empty():
                layers.append(
                    alt.Chart(interpolated)
                    .mark_rect()
                    .encode(
                        x=alt.X(f"{x_axis}:Q"),
                        y=alt.Y("depth:Q", sort="descending", scale=y_scale),
                        color=alt.Color(
                            f"{variable}:Q", scale=alt.Scale(scheme="viridis")
                        ),
                    )
                )
        layers.append(points)
        if show_bathymetry and not bathy.is_empty():
            bathy_plot = bathy.with_columns(
                pl.lit(max(bottom * 1.02, 1.0)).alias("section_bottom")
            )
            layers.insert(
                0,
                alt.Chart(bathy_plot)
                .mark_area(color="#6b7280", opacity=0.35)
                .encode(
                    x=alt.X(f"{x_axis}:Q"),
                    y=alt.Y("bathymetry:Q", sort="descending", scale=y_scale),
                    y2="section_bottom:Q",
                ),
            )
        st.altair_chart(
            alt.layer(*layers).interactive().properties(height=620), width="stretch"
        )


def render_profile(
    data_df: pl.DataFrame,
    dataset: str,
    selected: list[str],
    casts_df: pl.DataFrame,
    bottle_df: pl.DataFrame,
) -> None:
    with st.container(border=True):
        st.subheader(f"Single-station profiles from {dataset}")
        source = "Bottles"
        if dataset == "CTD bottles":
            source = st.segmented_control(
                "CTD source",
                ["Bottles", "Casts", "Both"],
                default="Bottles",
                key="profile_ctd_source",
                help="Casts uses ctd/cast/{cruise}/{cast}.csv and depsm; bottles uses ctd/bottles/{cruise}.csv and depsm.",
            )
            cast_data = cast_plot_data(casts_df)
            bottle_data = use_endpoint_depth(bottle_df, "depsm")
            data_df = (
                cast_data
                if source == "Casts"
                else pl.concat([cast_data, bottle_data], how="diagonal_relaxed")
                if source == "Both"
                else bottle_data
            )
        st.caption(
            "Profile depth uses the selected endpoint's depth field."
            if dataset == "CTD bottles"
            else DATASET_SPECS[dataset].depth_note
        )
        if data_df.is_empty() or "depth" not in data_df.columns:
            st.warning("Load a dataset with depth values to view profiles.")
            return
        cruises = (
            data_df["cruise"].unique().sort().to_list()
            if "cruise" in data_df.columns
            else selected
        )
        with st.container(horizontal=True, vertical_alignment="bottom"):
            cruise = st.selectbox("Profile cruise", cruises, key="profile_cruise")
            subset = (
                data_df.filter(pl.col("cruise") == cruise)
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
            values = (
                subset[selector_column]
                .drop_nulls()
                .unique()
                .sort()
                .cast(pl.Utf8)
                .to_list()
                if selector_column in subset.columns
                else []
            )
            if not values:
                st.warning(
                    f"No {selector_type.lower()} values are available for profiles."
                )
                return
            selected_profile = st.selectbox(
                f"Profile {selector_type.lower()}", values, key="profile_selector_value"
            )
            variables = numeric_columns(subset, exclude=PLOT_EXCLUDE_COLUMNS)
            if not variables:
                st.warning("The selected source has no numeric profile property.")
                return
            profile_var = st.selectbox(
                "Profile variable", variables, key="profile_variable"
            )
        profile_df = (
            subset.filter(
                pl.col(selector_column).cast(pl.Utf8) == str(selected_profile)
            )
            .drop_nulls(["depth", profile_var])
            .sort("depth")
        )
        if profile_df.is_empty():
            st.info("No profile rows remain after applying the filters.")
            return
        color = (
            "cast"
            if selector_type == "Station" and "cast" in profile_df.columns
            else "replicate"
            if "replicate" in profile_df.columns
            else alt.value("#1f77b4")
        )
        chart = (
            alt.Chart(profile_df)
            .mark_line(point=True)
            .encode(
                x=alt.X(f"{profile_var}:Q", title=profile_var.title()),
                y=alt.Y("depth:Q", title="Depth (m)", sort="descending"),
                order=alt.Order("depth:Q", sort="ascending"),
                color=color,
                tooltip=[
                    c
                    for c in [
                        "cruise",
                        "cast",
                        "niskin",
                        "replicate",
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
        st.subheader("Profile data source")
        endpoint_lines = []
        if dataset == "CTD bottles":
            if source in {"Bottles", "Both"}:
                endpoint_lines.append(f"{API_BASE}/ctd/bottles/{cruise}.csv")
            if source in {"Casts", "Both"}:
                endpoint_lines.append(
                    f"{API_BASE}/ctd/cast/{cruise}/{selected_profile if selector_type == 'Cast' else '{cast}'}.csv"
                )
        else:
            endpoint_lines.append(
                f"{API_BASE}/{DATASET_SPECS[dataset].endpoint.format(cruise=cruise)}"
            )
        for endpoint in endpoint_lines:
            st.code(endpoint)
        st.caption(f"Rows used for this plot: {profile_df.height:,}")
        st.dataframe(
            profile_df, hide_index=True, width="stretch", key="profile_plot_data_table"
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
        tables = [
            ("Underway", underway_df, f"{API_BASE}/underway/{{cruise}}.csv"),
            ("CTD casts", casts_df, f"{API_BASE}/ctd/cast/{{cruise}}/{{cast}}.csv"),
            ("CTD bottles", bottle_df, f"{API_BASE}/ctd/bottles/{{cruise}}.csv"),
        ]
        if dataset != "CTD bottles":
            tables.append(
                (dataset, data_df, f"{API_BASE}/{DATASET_SPECS[dataset].endpoint}")
            )
        tabs = st.tabs([name for name, _, _ in tables])
        for tab, (name, table, endpoint) in zip(tabs, tables):
            with tab:
                st.write(f"{name}: {table.height:,} rows")
                st.caption(f"API endpoint: {endpoint}")
                if table.is_empty():
                    st.info(f"No {name.lower()} data were available.")
                else:
                    st.dataframe(
                        table,
                        hide_index=True,
                        height=650,
                        width="stretch",
                        key=f"loaded_data_{name.lower().replace(' ', '_')}",
                    )
