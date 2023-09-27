from __future__ import annotations
from pathlib import Path
from enum import Enum

from mlxtend.frequent_patterns import apriori, fpgrowth, fpmax, hmine, association_rules
import pandas as pd
import joblib

from data_loader import DataLoader
from definitions import NotFittedError


class MiningAlgorithm(Enum):
    APRIORI = apriori
    FP_GROWTH = fpgrowth
    FPMAX = fpmax
    H_MINE = hmine

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)


class AssociationRuleRecommender:
    def __init__(self, loader: DataLoader, mining_algorithm: MiningAlgorithm, item_lookup):
        self.data = loader.load_data()
        self.mining_algorithm = mining_algorithm
        self.item_lookup = item_lookup

        self._fitted = False

    def fit(self):
        frequent_itemsets = self._frequent_itemset_mining()

        self.rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
        self._fitted = True

    def get_recommendations(self):
        self._check_if_fitted()
        pass

    def _frequent_itemset_mining(self, min_support: float = 0.1) -> pd.DataFrame:
        """ Perform frequent itemset mining using the specified algorithm and minimum support.

        Args:
            min_support: The minimum support of the itemsets to be returned. Defaults to 0.1.

        Returns:
            A DataFrame of frequent itemsets.

        Raises:
            ValueError: If an invalid mining algorithm is specified.
        """
        if self.mining_algorithm not in MiningAlgorithm:
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

    def _check_if_fitted(self) -> None:
        if not self._fitted:
            raise NotFittedError("Model has not been fitted yet.")
