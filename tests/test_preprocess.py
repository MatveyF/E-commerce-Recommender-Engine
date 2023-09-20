import pytest
import pandas as pd

from preprocess import remove_invalid, preprocess


@pytest.fixture
def sample_data():
    data = {
        "InvoiceDate": ["2009-12-12"] * 5 + ["2010-12-01"] + ["2011-11-11"] * 4,
        "StockCode": ["UwU", "F", "D", "FAILS", "C2", "DOT", "RSF", "M", "M&M", "POST"],
        "Customer ID": [0, None, 1, 2, 3, 4, 6, 7, 8, 9],
        "Invoice": ["55", "F", "1", "C2", "3", "4", "6", "7", "8", "9"],
    }
    return pd.DataFrame(data)


def test_remove_invalid(sample_data):
    df = sample_data

    result = remove_invalid(df)

    expected = pd.DataFrame(
        {
            "InvoiceDate": ["2009-12-12", "2011-11-11", "2011-11-11"],
            "StockCode": ["UwU", "RSF", "M&M"],
            "Customer ID": [0.0, 6.0, 8.0],
            "Invoice": ["55", "6", "8"],
        },
        index=[0, 6, 8],
    )

    pd.testing.assert_frame_equal(result, expected)


def test_preprocess(mocker, sample_data):
    df_2009_2010 = sample_data.iloc[:6]
    df_2010_2011 = sample_data.iloc[6:]

    train, test = preprocess(df_2009_2010, df_2010_2011, "2011-10-12")

    assert train.shape[0] == 1
    assert test.shape[0] == 2
    assert "C2" not in train["Invoice"].values
    assert "POST" not in test["StockCode"].values
