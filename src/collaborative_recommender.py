from pathlib import Path

import pandas as pd
from scipy.sparse import coo_matrix
import joblib
from loguru import logger

from data_loader import DataLoader
from definitions import ImplicitModel, NotFittedError


class CollaborativeRecommender:
    """A collaborative filtering recommender system using implicit feedback.

    Args:
        loader:
            DataLoader instance to load the dataset.
        model:
            ImplicitModel instance to build the recommender system.
        alpha_val:
            A scaling factor used to weigh the confidence levels of user-item interactions.
            The confidence in a user's preference for an item is calculated as the number of
            interactions multiplied by this alpha value. Default is 15.
    """

    def __init__(self, loader: DataLoader, model: ImplicitModel, alpha_val: float = 15):
        self.model = model
        self._fitted = False
        self.alpha_val = alpha_val

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

        logger.info("Collaborative recommender initialized")

    def fit(self) -> None:
        """Fits the collaborative recommender model.

        Uses the item-user interactions matrix to fit the model and sets the model to "fitted" status.
        """
        # Calculate the confidence by multiplying it by the alpha value
        data_conf = (self.item_user_coo * self.alpha_val).astype("double")

        try:
            self.model.fit(data_conf)
            self._fitted = True
            logger.info("Collaborative recommender has been fitted")
        except Exception as e:
            logger.error(f"An error occurred during model fitting: {e}")
            raise e

    def get_recommendations(self, user_id: int) -> pd.DataFrame:
        """Returns the recommended items for a given user.

        Args:
            user_id: The ID of the user for whom to generate recommendations

        Returns:
            DataFrame containing recommended items and their respective scores
        """
        self._check_if_fitted()

        ids, scores = self.model.recommend(user_id, self.user_item_csr[user_id], N=10, filter_already_liked_items=False)

        items = self.item_lookup[self.item_lookup.stock_code.isin(ids.astype(str))]

        for id_, score in zip(ids, scores):
            items.loc[items.stock_code == str(id_), "score"] = score

        return items

    def save_model(self, directory_path: Path) -> None:
        self._check_if_fitted()
        logger.info(f"Saving a collaborative recommender model at {directory_path}")

        try:
            with open(directory_path / "collaborative_model.joblib", "wb") as f:
                joblib.dump(self.model, f)
                logger.info("Collaborative model saved successfully")
        except Exception as e:
            logger.error(f"An error occurred while saving the model: {e}")
            raise

    def load_model(self, model_path: Path) -> None:
        logger.info(f"Loading a collaborative recommender model from {model_path}")

        try:
            with open(model_path, "rb") as f:
                self.model = joblib.load(f)
                self._fitted = True
                logger.info("Collaborative model loaded successfully")
        except Exception as e:
            logger.error(f"An error occurred while loading the model: {e}")
            raise

    def _check_if_fitted(self) -> None:
        """Internal utility method to check if the model has been fitted.

        Raises:
            NotFittedError: If the model has not been fitted yet.
        """
        if not self._fitted:
            logger.error("Collaborative recommender has not been fitted yet. Please use `fit()` method first")
            raise NotFittedError("Model has not been fitted yet.")
