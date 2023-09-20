from pathlib import Path
from abc import ABC, abstractmethod

import pandas as pd
import psycopg2
from psycopg2 import sql


class DataLoader(ABC):
    """Signature for a data loader."""

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        pass


class CSVLoader(DataLoader):
    """Data loader for CSV files."""

    def __init__(self, path: Path):
        self.path = path

    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.path)


class PostgresLoader(DataLoader):
    """Data loader for Postgres database."""

    def __init__(self, dbname: str, user: str, password: str, host: str, port: int, table: str):
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.table = table

    def load_data(self) -> pd.DataFrame:
        conn = psycopg2.connect(
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
        )

        cur = conn.cursor()

        query = sql.SQL("SELECT * FROM {}").format(sql.Identifier(self.table))
        cur.execute(query)

        rows = cur.fetchall()

        # Get the column names from the cursor description
        colnames = [desc[0] for desc in cur.description]

        cur.close()
        conn.close()

        return pd.DataFrame(rows, columns=colnames)
