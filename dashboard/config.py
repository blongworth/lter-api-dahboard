from dataclasses import dataclass

API_BASE = "https://nes-lter-api.whoi.edu/api"
GEBCO_OPENDAP_URL = (
    "https://dap.ceda.ac.uk/thredds/dodsC/bodc/gebco/global/gebco_2026/"
    "ice_surface_elevation/netcdf/GEBCO_2026.nc"
)
OCEAN_BASEMAP_TILE_URL = (
    "https://services.arcgisonline.com/ArcGIS/rest/services/Ocean/"
    "World_Ocean_Base/MapServer/tile/{z}/{y}/{x}"
)
OCEAN_BASEMAP_ATTRIBUTION = (
    "Esri, GEBCO, NOAA, National Geographic, Garmin, HERE, Geonames.org, "
    "and other contributors"
)


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    endpoint: str
    depth_note: str
    depth_field: str | None = None


DATASET_SPECS = {
    "CTD bottles": DatasetSpec(
        "CTD bottles",
        "ctd/bottles/{cruise}.csv",
        "Depth uses the bottle endpoint's `depsm` field.",
        "depsm",
    ),
    "Nutrients": DatasetSpec(
        "Nutrients",
        "nut/{cruise}.csv",
        "Depth is provided directly by the nutrients endpoint.",
    ),
    "Chlorophyll": DatasetSpec(
        "Chlorophyll",
        "chl/{cruise}.csv",
        "Depth is provided directly by the chlorophyll endpoint.",
    ),
}
DATASETS = {name: spec.endpoint for name, spec in DATASET_SPECS.items()}
UNDERWAY_ENDPOINT = "underway/{cruise}.csv"

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
