import polars as pl

from dashboard.analysis import interpolate_section, mask_interpolation_by_bathymetry


def test_interpolation_uses_vertical_cast_profiles_without_extrapolation() -> None:
    data = pl.DataFrame(
        {
            "cruise": ["A", "A", "B", "B"],
            "cast": [1, 1, 2, 2],
            "latitude": [40.0, 40.0, 41.0, 41.0],
            "depth": [0.0, 100.0, 0.0, 100.0],
            "value": [0.0, 100.0, 10.0, 110.0],
        }
    )
    result = interpolate_section(data, "latitude", "value")
    assert not result.is_empty()
    assert result["depth"].min() >= 0.0
    assert result["depth"].max() <= 100.0
    shallow = result.filter(pl.col("depth") == result["depth"].min())["value"].mean()
    deep = result.filter(pl.col("depth") == result["depth"].max())["value"].mean()
    assert deep > shallow


def test_bathymetry_mask_removes_cells_below_seafloor() -> None:
    interpolated = pl.DataFrame(
        {
            "latitude": [40.0, 40.0, 40.0],
            "depth": [10.0, 50.0, 110.0],
            "value": [1.0, 2.0, 3.0],
        }
    )
    bathymetry = pl.DataFrame({"latitude": [40.0], "bathymetry": [100.0]})
    result = mask_interpolation_by_bathymetry(interpolated, bathymetry, "latitude")
    assert result["depth"].to_list() == [10.0, 50.0]
