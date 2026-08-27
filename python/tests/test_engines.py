"""Behavior pins for the dataframe read methods.

`to_pyarrow_table` and `to_pandas` read with the built-in DataFusion engine;
`filters` is a SQL predicate returning exactly the matching rows. These tests
pin the declared SQL semantics: three-valued NULL logic, IS NULL as the only
null spelling, and duplicate projections as errors.
"""

import pathlib

import pytest
from arro3.core import Array, DataType, Table
from arro3.core import Field as ArrowField

from deltalake import DeltaTable, write_deltalake
from deltalake.exceptions import DeltaError

pytestmark = pytest.mark.pyarrow

COLUMN_MAPPED_TABLE = "../crates/test/tests/data/table_with_column_mapping"
DV_TABLE = "../crates/test/tests/data/table-with-dv-small"


@pytest.fixture
def edge_table(tmp_path: pathlib.Path) -> DeltaTable:
    """s: ["", "a", "b", None], v: [1, 2, 3, 4]."""
    data = Table(
        {
            "s": Array(
                ["", "a", "b", None], ArrowField("s", DataType.string(), nullable=True)
            ),
            "v": Array([1, 2, 3, 4], ArrowField("v", DataType.int64(), nullable=True)),
        }
    )
    write_deltalake(tmp_path, data)
    return DeltaTable(tmp_path)


def read(dt: DeltaTable, **kwargs) -> dict:
    return dt.to_pyarrow_table(**kwargs).sort_by("v").to_pydict()


def test_full_read(edge_table):
    assert read(edge_table) == {
        "s": ["", "a", "b", None],
        "v": [1, 2, 3, 4],
    }


def test_typed_comparisons(edge_table):
    assert read(edge_table, filters="v >= 3")["v"] == [3, 4]
    assert read(edge_table, filters="v IN (1, 3)")["v"] == [1, 3]
    assert read(edge_table, filters="v = 1 OR s = 'b'")["v"] == [1, 3]


def test_empty_string_is_data(edge_table):
    # '' is an empty string, not a null
    assert read(edge_table, filters="s = ''") == {"s": [""], "v": [1]}


def test_columns_projection(edge_table):
    table = edge_table.to_pyarrow_table(columns=["v"])
    assert table.column_names == ["v"]
    assert sorted(table["v"].to_pylist()) == [1, 2, 3, 4]


def test_not_in_three_valued_logic(edge_table):
    # NULL NOT IN (...) is NULL, so the row is dropped
    assert read(edge_table, filters="s NOT IN ('a')")["s"] == ["", "b"]


def test_is_null_spelling(edge_table):
    # IS [NOT] NULL is the only null spelling; = NULL matches nothing
    assert read(edge_table, filters="s IS NULL")["s"] == [None]
    assert read(edge_table, filters="s IS NOT NULL")["s"] == ["", "a", "b"]


def test_duplicate_projection_errors(edge_table):
    with pytest.raises(DeltaError):
        edge_table.to_pyarrow_table(columns=["v", "v"])


def test_unknown_column_in_filters_errors(edge_table):
    with pytest.raises(DeltaError, match="nonexistent"):
        edge_table.to_pyarrow_table(filters="nonexistent = 1")


def test_reads_column_mapped_table():
    dt = DeltaTable(COLUMN_MAPPED_TABLE)
    table = dt.to_pyarrow_table(filters="\"Company Very Short\" = 'BME'")
    assert table["Super Name"].to_pylist() == ["Timothy Lamb"]


def test_reads_deletion_vector_table():
    dt = DeltaTable(DV_TABLE)
    table = dt.to_pyarrow_table()
    assert sorted(table["value"].to_pylist()) == [1, 2, 3, 4, 5, 6, 7, 8]


@pytest.mark.pandas
def test_to_pandas(edge_table):
    df = edge_table.to_pandas(filters="v < 3")
    assert sorted(df["v"].tolist()) == [1, 2]
