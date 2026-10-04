"""Una clase por fuente. Cada una solo declara su propia regla de limpieza,
todo lo demás (leer bronze, escribir silver, loggear) vive en SilverTransform."""
import pandas as pd

from etl.common.silver_base import SilverTransform

YES_NO_COLS_BANK_MARKETING = ["default", "housing", "loan", "deposit"]


class PaySimClean(SilverTransform):
    source_name = "paysim"

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates().astype({"isFraud": bool, "isFlaggedFraud": bool})
        return df.dropna(subset=["nameOrig", "nameDest", "amount"])


class BankTransactionsClean(SilverTransform):
    source_name = "bank_transactions"

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates(subset=["TransactionID"])
        df = df.assign(TransactionDate=pd.to_datetime(df["TransactionDate"], format="%d/%m/%y", errors="coerce"))
        return df.dropna(subset=["CustomerID", "TransactionDate"])


class CreditCardClean(SilverTransform):
    source_name = "creditcard"

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates().astype({"Class": bool})


class BankMarketingClean(SilverTransform):
    source_name = "bank_marketing"

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates()
        return df.assign(**{c: df[c].str.strip().str.lower().eq("yes") for c in YES_NO_COLS_BANK_MARKETING})


ALL_TRANSFORMS = [PaySimClean, BankTransactionsClean, CreditCardClean, BankMarketingClean]
