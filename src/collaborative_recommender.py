from pathlib import Path
from enum import Enum

import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from loguru import logger

from data_loader import DataLoader
from definitions import ImplicitModel, NotFittedError


class CollaborativeMethod(Enum):
    USER_BASED = "user-based"
    ITEM_BASED = "item-based"


class CollaborativeRecommender:
    """A collaborative filtering recommender system using implicit feedback.

    Args:
        loader:
            DataLoader instance to load the dataset.
        model:
            ImplicitModel instance to build the recommender system.
        method:
            The method to use for collaborative filtering. Default is user-based.
        alpha_val:
            A scaling factor used to weigh the confidence levels of user-item interactions.
            The confidence in a user's preference for an item is calculated as the number of
            interactions multiplied by this alpha value. Default is 15.
    """

    def __init__(
        self,
        loader: DataLoader,
        model: ImplicitModel,
        method: CollaborativeMethod = CollaborativeMethod.USER_BASED,
        alpha_val: float = 15,
    ):
        self.model = model
        self.method = method
        self.alpha_val = alpha_val
        self._fitted = False
        self.item_similarity = None

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

        logger.info("Collaborative recommender initialised")

    def fit(self) -> None:
        """Fits the collaborative recommender model and sets the model to "fitted" status.

        For user-based collaborative filtering, we fit a model from the implicit library.
        For item-based collaborative filtering, we compute the cosine similarity between items.

        Raises:
            ValueError: If the method is not supported.
        """

        if self.method == CollaborativeMethod.USER_BASED:
            # Calculate the confidence by multiplying it by the alpha value
            data_conf = (self.item_user_coo * self.alpha_val).astype("double")

            try:
                self.model.fit(data_conf)
                self._fitted = True
                logger.info("Collaborative recommender (user-based) has been fitted")
            except Exception as e:
                logger.error(f"An error occurred during model fitting: {e}")
                raise e

        elif self.method == CollaborativeMethod.ITEM_BASED:
            try:
                self.item_similarity = cosine_similarity(self.item_user_coo)
                self._fitted = True
                logger.info("Collaborative recommender (item-based) has been initialized with item similarities")
            except Exception as e:
                logger.error(f"An error occurred during item similarity computation: {e}")
                raise e

        else:
            logger.error(f"Unknown method: {self.method}. Expected 'user-based' or 'item-based'.")
            raise ValueError(f"Unknown method: {self.method}. Expected 'user-based' or 'item-based'.")

    def get_user_recommendations(self, user_id: int, n: int = 10) -> pd.DataFrame:
        """Returns the recommended items for a given user.

        Args:
            user_id: The ID of the user for whom to generate recommendations
            n: The number of recommendations to return

        Returns:
            DataFrame containing recommended items and their respective scores
        """
        self._check_if_fitted()

        ids, scores = self.model.recommend(user_id, self.user_item_csr[user_id], N=n, filter_already_liked_items=False)

        items = self.item_lookup[self.item_lookup.stock_code.isin(ids.astype(str))]

        for id_value, score in zip(ids, scores):
            items.loc[items.stock_code == str(id_value), "score"] = score

        return items

    def get_item_recommendations(self, item_id: int, n: int = 10) -> pd.DataFrame:
        """Returns the recommended items for a given item.

        Args:
            item_id: The ID of the item for which to generate recommendations
            n: The number of recommendations to return

        Returns:
            DataFrame containing recommended items and their respective scores

        Raises:
            ValueError: If the item similarity matrix is not initialized.
            ValueError: If the method is not supported.
        """
        self._check_if_fitted()

        if self.item_similarity is None:
            logger.error("Item similarity matrix is not initialized")
            raise ValueError("Item similarity matrix is not initialized")

        if self.method == CollaborativeMethod.USER_BASED:
            logger.warning("User-based collaborative recommender does not support item recommendations")
            raise ValueError("User-based collaborative recommender does not support item recommendations")

        similar_items = self.item_similarity[item_id]
        ids = similar_items.argsort()[-n:][::-1]
        scores = similar_items[ids]

        items = self.item_lookup[self.item_lookup.stock_code.isin(ids.astype(str))]

        for id_value, score in zip(ids, scores):
            items.loc[items.stock_code == str(id_value), "score"] = score

        return items

    def save_model(self, directory_path: Path) -> None:
        """Saves the collaborative recommender model."""

        self._check_if_fitted()

        if self.method == CollaborativeMethod.ITEM_BASED:
            logger.warning("Saving item-based collaborative recommender is not supported")
            return

        logger.info(f"Saving a collaborative recommender model at {directory_path}")

        try:
            with open(directory_path / "collaborative_model.joblib", "wb") as f:
                joblib.dump(self.model, f)
                logger.info("Collaborative model saved successfully")
        except Exception as e:
            logger.error(f"An error occurred while saving the model: {e}")
            raise

    def load_model(self, model_path: Path) -> None:
        """Loads the collaborative recommender model."""

        logger.info(f"Loading a collaborative recommender model from {model_path}")

        if self.method == CollaborativeMethod.ITEM_BASED:
            logger.warning("Loading item-based collaborative recommender is not supported")
            return

        try:
            with open(model_path, "rb") as f:
                self.model = joblib.load(f)
                self._fitted = True
                logger.info("Collaborative model loaded successfully")
        except Exception as e:
            logger.error(f"An error occurred while loading the model: {e}")
            raise

    def cache_item_similarity(self, directory_path: Path) -> None:
        """Caches the item similarity matrix."""

        self._check_if_fitted()

        if self.item_similarity is None:
            logger.error("Item similarity matrix is not initialized")
            raise ValueError("Item similarity matrix is not initialized")

        logger.info(f"Caching item similarity matrix at {directory_path}")

        try:
            with open(directory_path / "item_similarity_cache.pkl", "wb") as f:
                joblib.dump(self.item_similarity, f)
                logger.info("Item similarity matrix cached successfully")
        except Exception as e:
            logger.error(f"An error occurred while caching the item similarity matrix: {e}")
            raise

    def load_item_similarity_from_cache(self, item_similarity_path: Path) -> None:
        """Loads the item similarity matrix from cache."""

        logger.info(f"Loading item similarity matrix from {item_similarity_path}")

        try:
            with open(item_similarity_path, "rb") as f:
                self.item_similarity = joblib.load(f)
                self._fitted = True
                logger.info("Item similarity matrix loaded successfully")
        except Exception as e:
            logger.error(f"An error occurred while loading the item similarity matrix: {e}")
            raise

    def _check_if_fitted(self) -> None:
        """Internal utility method to check if the model has been fitted.

        Raises:
            NotFittedError: If the model has not been fitted yet.
        """
        if not self._fitted:
            logger.error("Collaborative recommender has not been fitted yet. Please use `fit()` method first")
            raise NotFittedError("Model has not been fitted yet.")
