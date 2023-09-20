from pathlib import Path

import pandas as pd
from scipy.sparse import coo_matrix
import joblib

from data_loader import DataLoader
from definitions import ImplicitModel, NotFittedError


class CollaborativeRecommender:
    def __init__(self, loader: DataLoader, model: ImplicitModel):
        self.model = model
        self._fitted = False

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

    def fit(self) -> None:
        # Calculate the confidence by multiplying it by our alpha value
        alpha_val = 15
        data_conf = (self.item_user_coo * alpha_val).astype("double")

        self.model.fit(data_conf)
        self._fitted = True

    def get_recommendations(self, user_id: int) -> pd.DataFrame:
        self._check_if_fitted()

        ids, scores = self.model.recommend(user_id, self.user_item_csr[user_id], N=10, filter_already_liked_items=False)

        items = self.item_lookup[self.item_lookup.stock_code.isin(ids.astype(str))]

        for id_, score in zip(ids, scores):
            items.loc[items.stock_code == str(id_), "score"] = score

        return items

    def save_model(self, directory_path: Path) -> None:
        self._check_if_fitted()

        with open(directory_path / "collaborative_model.joblib", "wb") as f:
            joblib.dump(self.model, f)

    def load_model(self, model_path: Path) -> None:
        with open(model_path, "rb") as f:
            self.model = joblib.load(f)
            self._fitted = True

    def _check_if_fitted(self) -> None:
        if not self._fitted:
            raise NotFittedError("Model has not been fitted yet.")
