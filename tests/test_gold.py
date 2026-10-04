import pandas as pd

from etl.common.quality_checks import assert_quality
from etl.gold.dimensions import build_dim_cliente, build_dim_fecha, build_dim_moneda
from etl.gold.facts import FX_FALLBACK_INR, _to_sk_fecha, build_fact_campana_marcado, build_fact_transaccion
from etl.gold.sources import SK_FECHA_DESCONOCIDA, generate_surrogate_key, map_surrogate_keys


def test_surrogate_key_is_deterministic_and_scoped_by_source():
    assert generate_surrogate_key("paysim", "C1") == generate_surrogate_key("paysim", " C1 ")
    assert generate_surrogate_key("paysim", "C1") != generate_surrogate_key("bank_transactions", "C1")


def test_map_surrogate_keys_matches_scalar_version_and_keeps_nulls():
    out = map_surrogate_keys(pd.Series(["a", None, "a"]), "src")
    assert out.iloc[0] == out.iloc[2] == generate_surrogate_key("src", "a")
    assert pd.isna(out.iloc[1])


def test_to_sk_fecha_routes_invalid_and_out_of_range_dates_to_unknown():
    out = _to_sk_fecha(pd.Series(["2023-01-15", "basura", None, "2133-01-01", "1900-01-01"]))
    assert out.tolist() == [20230115] + [SK_FECHA_DESCONOCIDA] * 4


def test_dim_fecha_has_unique_keys_and_unknown_member():
    dim = build_dim_fecha("2023-01-01", "2023-01-31")
    assert len(dim) == 32
    assert dim["sk_fecha"].is_unique
    assert SK_FECHA_DESCONOCIDA in dim["sk_fecha"].values


def test_dim_cliente_deduplicates_customers_across_sources(fake_silver):
    dim = build_dim_cliente()

    assert dim["sk_cliente"].is_unique
    counts = dim["source_system"].value_counts()
    assert counts["bank_transactions"] == fake_silver["bank_transactions"]["CustomerID"].nunique()
    assert counts["bank_marketing"] == len(fake_silver["bank_marketing"])
    paysim = fake_silver["paysim"]
    assert counts["paysim"] == pd.concat([paysim["nameOrig"], paysim["nameDest"]]).nunique()


def test_facts_keep_referential_integrity_with_dimensions(fake_silver):
    """El contrato central del copo de nieve: ningún hecho apunta a un miembro
    inexistente de dim_cliente, dim_fecha o dim_moneda."""
    dim_cliente = build_dim_cliente()
    dim_fecha = build_dim_fecha()
    dim_moneda = build_dim_moneda()

    fact_tx = build_fact_transaccion()
    assert len(fact_tx) == len(fake_silver["bank_transactions"]) + len(fake_silver["paysim"])
    assert_quality(
        fact_tx, "fact_transaccion", "gold", ["id_transaccion"],
        ref_checks={
            "sk_cliente": dim_cliente["sk_cliente"],
            "sk_fecha": dim_fecha["sk_fecha"],
            "sk_moneda": dim_moneda["sk_moneda"],
        },
    )

    fact_campana = build_fact_campana_marcado()
    assert len(fact_campana) == len(fake_silver["bank_marketing"])
    assert_quality(
        fact_campana, "fact_campana_marcado", "gold", ["id_campana_contacto"],
        ref_checks={"sk_cliente": dim_cliente["sk_cliente"], "sk_fecha": dim_fecha["sk_fecha"]},
    )


def test_fact_transaccion_falls_back_to_declared_fx_without_market_data(fake_silver):
    """Sin silver de mercado, la conversión INR->USD usa la tasa de respaldo declarada."""
    fact = build_fact_transaccion()
    inr = fact[fact["sk_moneda"] == "INR"]
    expected = (inr["monto_original"] / FX_FALLBACK_INR).round(2)
    pd.testing.assert_series_equal(inr["monto_usd"], expected, check_names=False)


def test_campaign_spread_follows_risk_rule(fake_silver):
    fact = build_fact_campana_marcado()
    expected = (
        1.0
        + (fact["tiene_hipoteca"] | fact["tiene_prestamo_personal"]) * 2.5
        + fact["tiene_mora"] * 1.5
    )
    pd.testing.assert_series_equal(fact["spread_tasa_credito"], expected, check_names=False)
