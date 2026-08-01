from __future__ import annotations

import polars as pl
import streamlit as st

from dashboard.config import DATASET_SPECS
from dashboard.data import (
    cruises_in_date_range,
    load_casts,
    load_cruises,
    load_dataset,
    load_datasets,
    load_underway,
)
from dashboard.views import (
    render_data,
    render_metadata,
    render_metrics,
    render_profile,
    render_section,
    render_sidebar_data_summary,
    render_track,
)

st.set_page_config(page_title="NES-LTER API dashboard", layout="wide")
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
    st.header("Data selection")
    selection_mode = st.segmented_control(
        "Select by", ["Cruises", "Date range"], default="Cruises"
    )
    cruise_names = cruise_df["name"].to_list()
    if selection_mode == "Cruises":
        default = [name for name in ["EN617"] if name in cruise_names] or cruise_names[
            -1:
        ]
        selected = st.multiselect(
            "Cruises", cruise_names, default=default, key="selected_cruises"
        )
    else:
        min_dt = cruise_df["start_time"].drop_nulls().min()
        max_dt = cruise_df["end_time"].drop_nulls().max()
        date_range = st.date_input(
            "Date range",
            value=(min_dt.date(), max_dt.date()),
            key="selected_date_range",
        )
        if len(date_range) != 2:
            st.stop()
        selected = cruises_in_date_range(cruise_df, *date_range)
        st.caption(f"{len(selected)} cruise(s) overlap this range.")

workspace = st.segmented_control(
    "Workspace", ["Explore", "Inspect"], default="Explore", key="workspace"
)
view = st.segmented_control(
    "View",
    ["Cruise track", "Sections", "Profiles"]
    if workspace == "Explore"
    else ["Data", "Metadata"],
    default="Cruise track" if workspace == "Explore" else "Data",
    key="view",
)

with st.sidebar:
    if view == "Cruise track":
        map_source = st.selectbox(
            "Map data source",
            ["Underway", "CTD", "Bottles"],
            key="map_source",
            help="Choose the endpoint used for map locations and the track.",
        )
    elif view == "Sections":
        section_datasets = tuple(
            st.multiselect(
                "Section data sources",
                list(DATASET_SPECS),
                default=["CTD bottles"],
                key="section_datasets",
            )
        )
    elif view == "Profiles":
        profile_dataset = st.selectbox(
            "Profile dataset",
            list(DATASET_SPECS),
            index=list(DATASET_SPECS).index("CTD bottles"),
            key="profile_dataset",
        )
        profile_source = st.segmented_control(
            "Profile source",
            ["CTD", "Bottles", "Both"],
            default="CTD",
            key="profile_source",
        )
    elif view == "Data":
        data_source = st.selectbox(
            "Data source",
            ["Underway", "CTD casts", *DATASET_SPECS],
            key="data_source",
        )
    elif view == "Metadata":
        metadata_dataset = st.selectbox(
            "Dataset documentation",
            list(DATASET_SPECS),
            key="metadata_dataset",
        )

    st.caption(f"{len(selected)} cruise(s) selected")
    with st.expander("Active selection"):
        st.write(f"**Workspace:** {workspace}")
        st.write(f"**View:** {view}")
        if view == "Cruise track":
            st.write(f"**Source:** {map_source}")
        elif view == "Sections":
            st.write(f"**Sources:** {', '.join(section_datasets) or 'None'}")
        elif view == "Profiles":
            st.write(f"**Dataset:** {profile_dataset}")
            st.write(f"**Source:** {profile_source}")
        elif view == "Data":
            st.write(f"**Source:** {data_source}")

if not selected:
    st.info("Select at least one cruise to begin.")
    st.stop()

selected_tuple = tuple(selected)
summary = cruise_df.filter(pl.col("name").is_in(selected))

context = " · ".join(selected_tuple)
if view == "Cruise track":
    context += f" · {map_source}"
elif view == "Sections":
    context += f" · {', '.join(section_datasets)}"
elif view == "Profiles":
    context += f" · {profile_dataset} · {profile_source}"
elif view == "Data":
    context += f" · {data_source}"
st.caption(context)

if view == "Metadata":
    render_metadata(selected_tuple, metadata_dataset)
elif view == "Cruise track":
    casts_df = load_casts(selected_tuple)
    underway_df = load_underway(selected_tuple)
    bottle_df = load_dataset(selected_tuple, "CTD bottles")
    render_metrics(summary, casts_df, pl.DataFrame(), "Selected data")
    render_sidebar_data_summary(
        [("Underway", underway_df.height), ("CTD casts", casts_df.height)]
    )
    render_track(casts_df, bottle_df, underway_df, "", map_source, selected_tuple)
elif view == "Sections":
    if not section_datasets:
        st.info("Select at least one section data source in the sidebar.")
        st.stop()
    data_df = load_datasets(selected_tuple, section_datasets)
    casts_df = load_casts(selected_tuple)
    section_label = ", ".join(section_datasets)
    render_sidebar_data_summary([(section_label, data_df.height)])
    render_metrics(summary, casts_df, data_df, section_label)
    render_section(data_df, section_label, selected)
elif view == "Profiles":
    data_df = load_dataset(selected_tuple, profile_dataset)
    casts_df = (
        load_casts(selected_tuple)
        if profile_dataset == "CTD bottles" and profile_source in {"CTD", "Both"}
        else pl.DataFrame()
    )
    bottle_df = data_df if profile_dataset == "CTD bottles" else pl.DataFrame()
    render_sidebar_data_summary(
        [(profile_dataset, data_df.height), ("CTD casts", casts_df.height)]
    )
    render_metrics(summary, casts_df, data_df, profile_dataset)
    render_profile(
        data_df,
        profile_dataset,
        selected,
        casts_df,
        bottle_df,
        profile_source,
    )
else:
    casts_df = load_casts(selected_tuple)
    underway_df = load_underway(selected_tuple)
    data_df = (
        load_underway(selected_tuple)
        if data_source == "Underway"
        else load_casts(selected_tuple)
        if data_source == "CTD casts"
        else load_dataset(selected_tuple, data_source)
    )
    bottle_df = (
        data_df
        if data_source == "CTD bottles"
        else load_dataset(selected_tuple, "CTD bottles")
    )
    render_sidebar_data_summary(
        [(data_source, data_df.height), ("CTD bottles", bottle_df.height)]
    )
    render_metrics(summary, casts_df, data_df, data_source)
    render_data(casts_df, data_df, bottle_df, underway_df, data_source, selected_tuple)

with st.expander("Selected cruises"):
    st.dataframe(
        summary, hide_index=True, width="stretch", key="selected_cruises_table"
    )
