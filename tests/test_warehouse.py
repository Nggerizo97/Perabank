import pandas as pd
import pytest

from etl.common import warehouse
from models.ml_clustering_pipeline import escribir_clusters


def test_empty_warehouse_has_no_tables(gold_dir):
    assert warehouse.tables() == set()
    assert warehouse.size_bytes() == 0
    assert warehouse.query("SELECT 1 AS x")["x"].tolist() == [1]


def test_each_parquet_is_queryable_by_table_name(gold_dir):
    warehouse.write_table(pd.DataFrame({"sk": ["a", "b", "c"], "monto": [1.0, 2.0, 3.0]}), "fact_x")
    warehouse.write_table(pd.DataFrame({"sk": ["a", "b"], "nombre": ["uno", "dos"]}), "dim_x")

    assert warehouse.tables() == {"fact_x", "dim_x"}
    assert warehouse.size_bytes() > 0
    df = warehouse.query(
        "SELECT d.nombre, f.monto FROM fact_x f JOIN dim_x d USING (sk) WHERE f.monto >= ? ORDER BY f.monto",
        (2.0,),
    )
    assert df.to_dict("list") == {"nombre": ["dos"], "monto": [2.0]}


def test_paths_with_quotes_are_escaped(tmp_path, monkeypatch):
    monkeypatch.setattr(warehouse, "GOLD_DIR", tmp_path / "o'brien gold")
    warehouse.write_table(pd.DataFrame({"x": [7]}), "t")
    assert warehouse.query("SELECT x FROM t")["x"].tolist() == [7]


def test_unknown_table_raises(gold_dir):
    with pytest.raises(Exception, match="no_existe"):
        warehouse.query("SELECT * FROM no_existe")


def test_cluster_assignments_live_beside_gold_without_touching_it(gold_dir):
    dim = pd.DataFrame({"sk_cliente": ["a", "b", "c"]})
    dim_path = warehouse.write_table(dim, "dim_cliente")
    antes = dim_path.read_bytes()

    def resultado(clusters):
        return {
            "dominio": "retail",
            "clave": "sk_cliente",
            "columna_cluster": "sk_cluster_retail",
            "asignaciones": pd.DataFrame({"clave": ["a", "b"], "cluster": clusters}),
        }

    escribir_clusters(resultado([0, 1]))
    escribir_clusters(resultado([1, 1]))  # una nueva corrida reemplaza, no acumula

    assert dim_path.read_bytes() == antes
    df = warehouse.query("""
        SELECT c.sk_cliente, s.sk_cluster_retail
        FROM dim_cliente c LEFT JOIN cluster_retail s USING (sk_cliente)
        ORDER BY c.sk_cliente
    """)
    assert df["sk_cluster_retail"].tolist()[:2] == [1, 1]
    assert pd.isna(df["sk_cluster_retail"].iloc[2])
