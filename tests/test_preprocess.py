import pandas as pd

from src.preprocess import remove_invalid


def test_remove_invalid():
    df = pd.DataFrame(
        {
            "StockCode": ["F", "D", "FAILS", "C2", "DOT", "RSF", "M", "M&M", "POST"],
            "Customer ID": [None, 1, 2, 3, 4, 6, 7, 8, 9],
            "Invoice": ["F", "1", "C2", "3", "4", "6", "7", "8", "9"],
        }
    )

    result = remove_invalid(df)

    expected = pd.DataFrame(
        {
            "StockCode": ["RSF", "M&M"],
            "Customer ID": [6.0, 8.0],
            "Invoice": ["6", "8"],
        },
        index=[5, 7],
    )

    pd.testing.assert_frame_equal(result, expected)
