from typing import Protocol

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from data_loader import DataLoader


class NotFittedError(RuntimeError):
    pass


class ImplicitModel(Protocol):
    def fit(self, item_user_data: coo_matrix) -> None:
        ...

    def recommend(
        self,
        user_id: int,
        item_user_data: csr_matrix,
        N: int,
        filter_already_liked_items: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        ...


class CollaborativeRecommender:
    def __init__(self, loader: DataLoader, model: ImplicitModel):
        self.model = model

        df: pd.DataFrame = loader.load_data()

        # Create a numeric customer_id and stock_code column
        df["customer_id"] = df["Customer ID"].astype("category").cat.codes
        df["stock_code"] = df["StockCode"].astype("category").cat.codes

        # Create a lookup frame, so we can get the stock_codes later
        self.item_lookup = df[["stock_code", "Description"]].drop_duplicates()
        self.item_lookup["stock_code"] = self.item_lookup.stock_code.astype(str)
        self.item_lookup["score"] = 0

        # Create sparse matrices
        self.item_user_coo = coo_matrix((df["Quantity"].astype(float), (df["stock_code"], df["customer_id"])))
        self.user_item_csr = self.item_user_coo.T.tocsr()

        self._fitted = False

    def fit(self) -> None:
        self._fitted = True
        # Calculate the confidence by multiplying it by our alpha value
        alpha_val = 15
        data_conf = (self.item_user_coo * alpha_val).astype("double")

        self.model.fit(data_conf)

    def get_recommendations(self, user_id: int) -> pd.DataFrame:
        if not self._fitted:
            raise NotFittedError("Model has not been fitted yet.")

        ids, scores = self.model.recommend(user_id, self.user_item_csr[user_id], N=10, filter_already_liked_items=False)

        items = self.item_lookup[self.item_lookup.stock_code.isin(ids.astype(str))]

        for id_, score in zip(ids, scores):
            items.loc[items.stock_code == str(id_), "score"] = score

        return items
