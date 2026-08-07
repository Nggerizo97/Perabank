"""Una clase por fuente. Cada una solo declara su propia regla de limpieza,
todo lo demás (leer bronze, escribir silver, loggear) vive en SilverTransform."""
import pandas as pd

from etl.common.silver_base import SilverTransform

YES_NO_COLS_BANK_MARKETING = ["default", "housing", "loan", "deposit"]


class PaySimClean(SilverTransform):
    source_name = "paysim"

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates()
        df["isFraud"] = df["isFraud"].astype(bool)
        df["isFlaggedFraud"] = df["isFlaggedFraud"].astype(bool)
        return df.dropna(subset=["nameOrig", "nameDest", "amount"])


class BankTransactionsClean(SilverTransform):
    source_name = "bank_transactions"

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates(subset=["TransactionID"])
        df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], format="%d/%m/%y", errors="coerce")
        return df.dropna(subset=["CustomerID", "TransactionDate"])


class CreditCardClean(SilverTransform):
    source_name = "creditcard"

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates()
        df["Class"] = df["Class"].astype(bool)
        return df


class BankMarketingClean(SilverTransform):
    source_name = "bank_marketing"

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates()
        for col in YES_NO_COLS_BANK_MARKETING:
            df[col] = df[col].str.strip().str.lower().eq("yes")
        return df


ALL_TRANSFORMS = [PaySimClean, BankTransactionsClean, CreditCardClean, BankMarketingClean]
