import pandas as pd
import pytest

from etl.common.quality_checks import assert_quality
from schemas.contracts import ColumnContract, TableContract

CONTRACT = TableContract(
    table_name="t",
    layer="silver",
    primary_keys=["id"],
    columns=[ColumnContract("id", "int64", False), ColumnContract("valor", "float64")],
)


def test_contract_accepts_conforming_frame_with_extra_columns():
    df = pd.DataFrame({"id": [1, 2], "valor": [1.0, 2.0], "extra": ["a", "b"]})
    assert CONTRACT.validate(df) is True


def test_contract_rejects_missing_column_and_names_it():
    with pytest.raises(ValueError, match="valor"):
        CONTRACT.validate(pd.DataFrame({"id": [1]}))


def test_quality_passes_clean_frame():
    df = pd.DataFrame({"id": [1, 2, 3], "fk": ["a", "b", "a"]})
    assert_quality(df, "t", "gold", ["id"], non_null_cols=["fk"], ref_checks={"fk": pd.Series(["a", "b"])})


def test_quality_rejects_duplicate_primary_key():
    with pytest.raises(ValueError, match="duplicadas"):
        assert_quality(pd.DataFrame({"id": [1, 1]}), "t", "gold", ["id"])


def test_quality_rejects_null_in_required_column():
    df = pd.DataFrame({"id": [1, 2], "fk": ["a", None]})
    with pytest.raises(ValueError, match="'fk'"):
        assert_quality(df, "t", "gold", ["id"], non_null_cols=["fk"])


def test_quality_rejects_null_primary_key_even_if_not_listed_as_required():
    with pytest.raises(ValueError, match="'id'"):
        assert_quality(pd.DataFrame({"id": [1, None]}), "t", "gold", ["id"])


def test_quality_rejects_orphan_foreign_key():
    df = pd.DataFrame({"id": [1, 2], "fk": ["a", "zzz"]})
    with pytest.raises(ValueError, match="integridad referencial"):
        assert_quality(df, "t", "gold", ["id"], ref_checks={"fk": pd.Series(["a"])})
