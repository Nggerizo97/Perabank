from etl.bronze.ingest_raw import main as run_bronze
from etl.enrichment.clean_datos_gov_co import main as run_clean_gov_co
from etl.enrichment.clean_ecb_rates import main as run_clean_ecb
from etl.enrichment.clean_market_data import main as run_clean_market
from etl.enrichment.fetch_datos_gov_co import main as run_fetch_gov_co
from etl.enrichment.fetch_ecb_rates import main as run_fetch_ecb
from etl.enrichment.fetch_market_data import main as run_fetch_market
from etl.gold.run_gold import main as run_gold
from etl.silver.run_silver import main as run_silver


def main():
    run_bronze()
    run_fetch_market()
    run_fetch_ecb()
    run_fetch_gov_co()
    run_silver()
    run_clean_market()
    run_clean_ecb()
    run_clean_gov_co()
    run_gold()
    run_clustering()


def run_clustering():
    """Segmentación no supervisada, después de gold y nunca antes: agrupa sobre las
    tablas gold recién construidas y deja sus asignaciones en cluster_<dominio>.parquet.
    """
    from models.ml_clustering_pipeline import main as run_segmentacion
    run_segmentacion()




if __name__ == "__main__":
    main()

