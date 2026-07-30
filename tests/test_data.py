import polars as pl

from dashboard.data import normalize_dataframe, use_endpoint_depth


def test_numeric_cast_identifiers_are_integer() -> None:
    result = normalize_dataframe(
        pl.DataFrame({"cast": ["1", "10", None], "value": ["2.5", "3.5", "4.5"]})
    )
    assert result.schema["cast"] == pl.Int64
    assert result.schema["value"] == pl.Float64
    assert result["cast"].to_list() == [1, 10, None]


def test_mixed_cast_identifiers_remain_strings() -> None:
    result = normalize_dataframe(pl.DataFrame({"cast": ["1", "13b"]}))
    assert result.schema["cast"] == pl.String


def test_endpoint_depth_replaces_metadata_depth() -> None:
    result = use_endpoint_depth(
        pl.DataFrame({"depth": [10.0], "depsm": [12.5]}), "depsm"
    )
    assert result["depth"].to_list() == [12.5]
