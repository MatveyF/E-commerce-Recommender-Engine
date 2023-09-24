from __future__ import annotations
from pathlib import Path

from mlxtend.frequent_patterns import apriori, association_rules
import joblib

from data_loader import DataLoader
from definitions import NotFittedError


class AssociationRuleRecommender:
    def __init__(self, loader: DataLoader, item_lookup):
        self.data = loader.load_data()
        self.item_lookup = item_lookup

        self._fitted = False

    def fit(self):
        frequent_itemsets = apriori(self.data, min_support=0.1, use_colnames=True)
        self.rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

        self._fitted = True

    def get_recommendations(self):
        self._check_if_fitted()
        pass

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
