from __future__ import annotations
from pathlib import Path
from enum import Enum

from mlxtend.frequent_patterns import apriori, fpgrowth, fpmax, hmine, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import joblib
from loguru import logger

from data_loader import DataLoader
from definitions import NotFittedError


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
    def __init__(self, loader: DataLoader, mining_algorithm: MiningAlgorithm = MiningAlgorithm.APRIORI):
        self.mining_algorithm = mining_algorithm
        self.data = loader.load_data()

        self.data = self.data[["StockCode", "Customer ID"]]
        transactional_encoder = TransactionEncoder()
        encoded_data = transactional_encoder.fit_transform(
            self.data.groupby("Customer ID")["StockCode"].apply(list).values
        )
        self.data = pd.DataFrame(encoded_data, columns=transactional_encoder.columns_)

        self.rules = None
        self._fitted = False

        logger.info("AssociationRuleRecommender initialised")

    def fit(self):
        frequent_itemsets = self._frequent_itemset_mining()

        self.rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
        self._fitted = True

    def get_recommendations(self, item_id: str) -> pd.DataFrame:
        self._check_if_fitted()

        recommendations = self.rules[self.rules["antecedents"].apply(lambda x: item_id in x)]

        recommendations = recommendations.sort_values(by="lift", ascending=False)
        recommendations = recommendations[["antecedents", "consequents", "lift"]]

        return recommendations

    def _frequent_itemset_mining(self, min_support: float = 0.01) -> pd.DataFrame:
        """ Perform frequent itemset mining using the specified algorithm and minimum support.

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

        return self.mining_algorithm(self.data, min_support=min_support, use_colnames=True)

    def save_model(self, directory_path: Path) -> None:
        self._check_if_fitted()

        with open(directory_path / "association_rule_model.joblib", "wb") as f:
            joblib.dump(self, f)

    @staticmethod
    def load_model(model_path: Path) -> AssociationRuleRecommender:
        with open(model_path, "rb") as f:
            return joblib.load(f)

    def cache_rules(self, directory_path: Path) -> None:
        """Caches the association rules."""

        self._check_if_fitted()

        try:
            with open(directory_path / "association_rules.csv", "w") as f:
                self.rules.to_csv(f, index=False)
        except Exception as e:
            logger.error(f"An error occurred while caching the rules: {e}")
            raise

    def load_rules_from_cache(self, rule_path: Path) -> None:
        """Loads the association rules from cache."""

        try:
            self.rules = pd.read_csv(rule_path)
            self._fitted = True
            logger.info("Rules loaded successfully")
        except Exception as e:
            logger.error(f"An error occurred while loading the rules: {e}")
            raise

    def _check_if_fitted(self) -> None:
        if not self._fitted:
            raise NotFittedError("Model has not been fitted yet.")

        elif self.rules is None:
            raise ValueError("No rules found. Please fit the model first to generate rules.")
