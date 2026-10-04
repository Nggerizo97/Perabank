import pandas as pd

from etl.silver.transforms import (
    BankMarketingClean,
    BankTransactionsClean,
    CreditCardClean,
    PaySimClean,
)


def test_paysim_drops_duplicates_and_rows_without_parties():
    row = {"nameOrig": "O1", "nameDest": "D1", "amount": 10.0, "isFraud": 1, "isFlaggedFraud": 0}
    df = pd.DataFrame([row, row, {**row, "nameOrig": None}])

    out = PaySimClean().clean(df)

    assert len(out) == 1
    assert out["isFraud"].dtype == bool and out["isFraud"].iloc[0]
    assert out["isFlaggedFraud"].dtype == bool


def test_bank_transactions_parses_day_first_dates_and_drops_unparseable():
    df = pd.DataFrame({
        "TransactionID": ["T1", "T1", "T2", "T3"],
        "CustomerID": ["C1", "C1", "C2", "C3"],
        "TransactionDate": ["2/8/16", "2/8/16", "not a date", "31/12/16"],
    })

    out = BankTransactionsClean().clean(df)

    assert out["TransactionID"].tolist() == ["T1", "T3"]
    # 2/8/16 es 2 de agosto (día primero), no 8 de febrero.
    assert out["TransactionDate"].tolist() == [pd.Timestamp("2016-08-02"), pd.Timestamp("2016-12-31")]


def test_creditcard_casts_class_to_bool():
    out = CreditCardClean().clean(pd.DataFrame({"Time": [0.0, 1.0], "Class": [0, 1]}))
    assert out["Class"].tolist() == [False, True]


def test_bank_marketing_normalizes_yes_no_columns():
    df = pd.DataFrame({
        "default": [" Yes", "no"],
        "housing": ["YES", "No "],
        "loan": ["no", "yes"],
        "deposit": ["yes", "maybe"],
    })

    out = BankMarketingClean().clean(df)

    assert out["default"].tolist() == [True, False]
    assert out["housing"].tolist() == [True, False]
    assert out["loan"].tolist() == [False, True]
    assert out["deposit"].tolist() == [True, False]
