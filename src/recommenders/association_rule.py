from __future__ import annotations
from pathlib import Path
from enum import Enum
from typing import List

from mlxtend.frequent_patterns import apriori, fpgrowth, fpmax, hmine, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import joblib
from loguru import logger

from data_loader import DataLoader
from definitions import NotFittedError, Recommendation


class MiningAlgorithm(Enum):
    APRIORI = "apriori", apriori
    FP_GROWTH = "fpgrowth", fpgrowth
    FPMAX = "fpmax", fpmax
    H_MINE = "hmine", hmine

    def __new__(cls, label, func):
        obj = object.__new__(cls)
        obj._value_ = label
        obj.func = func
        return obj

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class AssociationRuleRecommender:
    """Association rule recommender.

    Args:
        loader:
            DataLoader instance to load the dataset.
        mining_algorithm:
            The algorithm to use for frequent itemset mining. Default is apriori.
    """

    def __init__(self, loader: DataLoader, mining_algorithm: MiningAlgorithm = MiningAlgorithm.APRIORI):
        self.mining_algorithm = mining_algorithm
        self.df = loader.load_data()

        # Create a lookup frame, so we can get the descriptions later
        self._item_lookup = self.df[["StockCode", "Description"]].drop_duplicates()

        self.df = self.df[["StockCode", "Customer ID"]]
        transactional_encoder = TransactionEncoder()
        encoded_data = transactional_encoder.fit_transform(
            self.df.groupby("Customer ID")["StockCode"].apply(list).values
        )
        self.df = pd.DataFrame(encoded_data, columns=transactional_encoder.columns_)

        self.rules = None
        self.fitted = False

        logger.info("AssociationRuleRecommender initialised")

    def fit(self) -> None:
        """Calculates frequent itemsets and association rules."""

        frequent_itemsets = self._frequent_itemset_mining()

        self.rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
        self.fitted = True

    def get_recommendations(self, item_id: int, n: int = 10) -> List[Recommendation]:
        """Get recommendations for a given item.

        Args:
            item_id: The item for which to get recommendations.
            n: The number of recommendations to return. Defaults to 10.

        Returns:
            A list of Recommendation objects.
        """

        self._check_if_fitted()

        recommendations = self.rules[self.rules["antecedents"].apply(lambda x: str(item_id) in x)]

        recommendations = recommendations.sort_values(by="lift", ascending=False).iloc[:n]
        recommendations = recommendations[["consequents", "lift"]]

        unique_stock_codes = set()
        result = []

        # Fetching descriptions and combining into a list of Recommendation objects
        for _, row in recommendations.iterrows():
            stock_codes: frozenset = row["consequents"]

            for stock_code in stock_codes:
                if stock_code in unique_stock_codes:
                    continue

                description = self._item_lookup[self._item_lookup["StockCode"] == stock_code]["Description"].iloc[0]
                unique_stock_codes.add(stock_code)

                result.append(Recommendation(stock_code=stock_code, description=description, score=row["lift"]))

        return result

    def _frequent_itemset_mining(self, min_support: float = 0.01) -> pd.DataFrame:
        """Perform frequent itemset mining using the specified algorithm and minimum support.

        Args:
            min_support: The minimum support of the itemsets to be returned. Defaults to 0.01.

        Returns:
            A DataFrame of frequent itemsets.

        Raises:
            ValueError: If an invalid mining algorithm is specified.
        """

        if self.mining_algorithm not in MiningAlgorithm:
            logger.error(f"Invalid mining algorithm : {self.mining_algorithm}")
            raise ValueError(f"Invalid mining algorithm : {self.mining_algorithm}")

        return self.mining_algorithm(self.df, min_support=min_support, use_colnames=True)

    def save_model(self, directory_path: Path) -> None:
        """Saves the association rule model."""

        self._check_if_fitted()

        with open(directory_path / "association_rule_model.joblib", "wb") as f:
            joblib.dump(self, f)

    @staticmethod
    def load_model(model_path: Path) -> AssociationRuleRecommender:
        """Loads the association rule model.

        Args:
            model_path: The path to the association rule model.

        Returns:
            The loaded AssociationRuleRecommender instance.

        Raises:
            ValueError: If the model is not fitted.
        """

        with open(model_path, "rb") as f:
            recommender = joblib.load(f)

            if not recommender.fitted:
                logger.error("Association rule model is not fitted")
                raise ValueError("Association rule model is not fitted")

            return recommender

    def cache_rules(self, directory_path: Path) -> None:
        """Caches the association rules."""

        self._check_if_fitted()

        logger.info(f"Caching association rules at {directory_path}")

        try:
            with open(directory_path / "association_rules.csv", "w") as f:
                self.rules.to_csv(f, index=False)
        except Exception as e:
            logger.error(f"An error occurred while caching the rules: {e}")
            raise

    def load_rules_from_cache(self, rule_path: Path) -> None:
        """Loads the association rules from cache.

        Args:
            rule_path: The path to the cached association rules.
        """

        try:
            self.rules = pd.read_csv(rule_path)
            self.fitted = True
            logger.info("Rules loaded successfully")
        except Exception as e:
            logger.error(f"An error occurred while loading the rules: {e}")
            raise

    def _check_if_fitted(self) -> None:
        """Internal utility method to check if the model has been fitted.

        Raises:
            NotFittedError: If the model has not been fitted yet.
            ValueError: If no rules are found.
        """

        if not self.fitted:
            raise NotFittedError("Model has not been fitted yet.")

        elif self.rules is None:
            raise ValueError("No rules found. Please fit the model first to generate rules.")
