"""This module contains functions I used for some of the data exploration. Please preprocess the data first."""
import pandas as pd

from loguru import logger


def find_optimal_data_split(df: pd.DataFrame) -> None:
    """Find an optimal date to split the dataset into training and test sets based on a scoring function.
    The scoring function rewards having users in both train and test sets and penalizes users only
    appearing in the test set.

    Args:
        df: The dataframe containing transaction data with columns 'InvoiceDate' and 'Customer ID'.
    """
    top_score = 0
    best_split_date, shared_customers, train_only_customers, test_only_customers = (
        None,
        None,
        None,
        None,
    )

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Prioritize customers present in both sets, penalize customers only in test set
    scoring_function = lambda x: 2 * x[0] + x[1] - x[2]

    for date in pd.date_range("2010-07-01", "2011-11-01", freq="D"):
        train = df[df["InvoiceDate"] < date]
        test = df[df["InvoiceDate"] >= date]

        train_customers = set(train["Customer ID"])
        test_customers = set(test["Customer ID"])

        score = scoring_function(
            [
                len(train_customers & test_customers),
                len(train_customers - test_customers),
                len(test_customers - train_customers),
            ]
        )

        if score > top_score:
            best_split_date = date
            top_score = score
            shared_customers = len(train_customers & test_customers)
            train_only_customers = len(train_customers - test_customers)
            test_only_customers = len(test_customers - train_customers)

    logger.info(f"Optimal split date: {best_split_date} with:")
    logger.info(f"Number of customers in both train and test sets: {shared_customers}")
    logger.info(f"Number of customers only in the train dataset: {train_only_customers}")
    logger.info(f"Number of customers only in the test dataset: {test_only_customers}")


def find_unsold_items(df: pd.DataFrame, split_date: str, months_before_split: int) -> None:
    """Identify and print the count of items that were not sold during specific periods.

    Args:
        df:
            The dataframe containing the transaction data with columns 'InvoiceDate' and 'StockCode'.
        split_date:
            The date at which the dataset is split into train and test sets.
        months_before_split:
            Number of months before the split_date to consider for identifying unsold items.
    """
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    all_items = set(df["StockCode"])

    gap_start = pd.Timestamp(split_date) - pd.DateOffset(months=months_before_split)

    items_sold_during_gap = set(
        df.loc[
            (df["InvoiceDate"] >= gap_start) & (df["InvoiceDate"] < split_date),
            "StockCode",
        ]
    )

    potentially_dead_items = all_items - items_sold_during_gap
    items_sold_in_test_period = set(df.loc[df["InvoiceDate"] >= split_date, "StockCode"])

    dead_items_sold_in_test = potentially_dead_items & items_sold_in_test_period
    truly_dead_items = potentially_dead_items - items_sold_in_test_period
    not_sold_during_gap = len(dead_items_sold_in_test) + len(truly_dead_items)

    logger.info(
        f"Count of items considered 'dead' but actually sold during the test period: {len(dead_items_sold_in_test)}"
    )
    logger.info(f"Count of items not sold after {gap_start} and not in the test period: {len(truly_dead_items)}")
    logger.info(f"Total count of items not sold during the gap period: {not_sold_during_gap}")
