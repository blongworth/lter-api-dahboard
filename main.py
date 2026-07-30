from __future__ import annotations

import polars as pl
import streamlit as st

from dashboard.config import DATASET_SPECS
from dashboard.data import (
    cruises_in_date_range,
    load_casts,
    load_cruises,
    load_dataset,
    load_underway,
)
from dashboard.views import (
    render_data,
    render_metadata,
    render_metrics,
    render_profile,
    render_section,
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
    st.header("Filters")
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
    dataset = st.selectbox(
        "Section/profile dataset", list(DATASET_SPECS), key="dataset"
    )

if not selected:
    st.info("Select at least one cruise to begin.")
    st.stop()

selected_tuple = tuple(selected)
summary = cruise_df.filter(pl.col("name").is_in(selected))
view = st.segmented_control(
    "View",
    ["Cruise track", "Sections", "Profiles", "Data", "Metadata"],
    default="Cruise track",
    key="view",
)

if view == "Metadata":
    render_metadata(selected_tuple, dataset)
elif view == "Cruise track":
    casts_df = load_casts(selected_tuple)
    underway_df = load_underway(selected_tuple)
    bottle_df = load_dataset(selected_tuple, "CTD bottles")
    render_metrics(summary, casts_df, pl.DataFrame(), "Selected data")
    render_track(casts_df, bottle_df, underway_df, dataset)
elif view == "Sections":
    data_df = load_dataset(selected_tuple, dataset)
    casts_df = load_casts(selected_tuple)
    render_metrics(summary, casts_df, data_df, dataset)
    render_section(data_df, dataset, selected)
elif view == "Profiles":
    data_df = load_dataset(selected_tuple, dataset)
    casts_df = (
        load_casts(selected_tuple) if dataset == "CTD bottles" else pl.DataFrame()
    )
    bottle_df = data_df if dataset == "CTD bottles" else pl.DataFrame()
    render_metrics(summary, casts_df, data_df, dataset)
    render_profile(data_df, dataset, selected, casts_df, bottle_df)
else:
    casts_df = load_casts(selected_tuple)
    underway_df = load_underway(selected_tuple)
    data_df = load_dataset(selected_tuple, dataset)
    bottle_df = (
        data_df
        if dataset == "CTD bottles"
        else load_dataset(selected_tuple, "CTD bottles")
    )
    render_metrics(summary, casts_df, data_df, dataset)
    render_data(casts_df, data_df, bottle_df, underway_df, dataset, selected_tuple)

with st.expander("Selected cruises"):
    st.dataframe(
        summary, hide_index=True, width="stretch", key="selected_cruises_table"
    )
