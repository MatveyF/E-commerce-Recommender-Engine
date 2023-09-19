from pathlib import Path

import pandas as pd
import typer


app = typer.Typer()


def remove_invalid(df: pd.DataFrame) -> pd.DataFrame:
    # Drop transactions that are not associated with any product (Discounts, Postage, etc.)
    df = df[~df["StockCode"].isin(["D", "C2", "POST", "DOT", "BANK CHARGES", "M"])]

    # No Customer Id (maybe checked out as guest? cannot recommend products without this information)
    df = df[~df["Customer ID"].isna()]

    # Cancelled orders
    df = df[~df["Invoice"].astype(str).str.startswith("C")]

    return df


@app.command()
def preprocess(
    path_to_xlsx: Path, output_dir_path: Path, split_date: str = "2011-10-12"
) -> None:
    """Reads original "online_retail_II.xlsx" dataset, removes duplicate overlap, invalid rows, and saves to a csv.

    Args:
        path_to_xlsx: Path to where the data in .xlsx format is stored
        output_dir_path: Path to a directory where to save train.csv and test.csv
        split_date: Date at which we split the data into train and test, default is 2011-10-12
    """

    # Read data from original "online_retail_II.xlsx"
    df_2009_2010 = pd.read_excel(path_to_xlsx, "Year 2009-2010")
    df_2010_2011 = pd.read_excel(path_to_xlsx, "Year 2010-2011")

    # Remove duplicate overlap in December of 2010
    df_2009_2010 = df_2009_2010[df_2009_2010["InvoiceDate"] < "2010-12-01"]

    combined = pd.concat([df_2009_2010, df_2010_2011], axis=0)
    combined = remove_invalid(combined)

    train = combined[combined["InvoiceDate"] < split_date]
    test = combined[combined["InvoiceDate"] >= split_date]

    train.to_csv(output_dir_path / "train.csv", index=False)
    test.to_csv(output_dir_path / "test.csv", index=False)


if __name__ == "__main__":
    app()
