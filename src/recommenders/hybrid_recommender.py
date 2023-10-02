from pathlib import Path
from typing import List

from collaborative_recommender import UserBasedCollaborativeRecommender, ItemBasedCollaborativeRecommender
from association_rule import AssociationRuleRecommender
from data_loader import DataLoader
from definitions import Recommendation


class HybridRecommender:
    def __init__(
        self,
        loader: DataLoader,
        user_based_collaborative_recommender: UserBasedCollaborativeRecommender,
        item_based_collaborative_recommender: ItemBasedCollaborativeRecommender,
        association_rule_recommender: AssociationRuleRecommender,
    ):
        self.user_based_collaborative_recommender = user_based_collaborative_recommender
        self.item_based_collaborative_recommender = item_based_collaborative_recommender
        self.association_rule_recommender = association_rule_recommender

        self.data = loader.load_data()

    def fit(self) -> None:
        self.user_based_collaborative_recommender.fit()
        self.item_based_collaborative_recommender.fit()
        self.association_rule_recommender.fit()

    def get_home_page_recommendations(self, user_id: int, n: int = 10) -> List[Recommendation]:
        """Retrieve recommendations for home page"""

        return self.user_based_collaborative_recommender.get_recommendations(user_id, n)

    def get_item_page_recommendations(self, item_id: int, n: int = 10) -> List[Recommendation]:
        """Retrieve recommendations for item page"""

        if self.item_based_collaborative_recommender.fitted:
            item_based_recommendations = self.item_based_collaborative_recommender.get_recommendations(item_id, n)
        else:
            item_based_recommendations = None

        if self.association_rule_recommender.fitted:
            association_rules_recommendations = self.association_rule_recommender.get_recommendations(item_id, n)
        else:
            association_rules_recommendations = None

        # If only one was used to recommend items, return that
        if item_based_recommendations is None:
            return association_rules_recommendations
        elif association_rules_recommendations is None:
            return item_based_recommendations

        # Otherwise, combine the recommendations
        # TODO: decide on a method to combine recommendations and the format of the output for each recommender
        combined_recommendations = ...

        return combined_recommendations

    def load_model_for_user_collaborative(self, path: Path) -> None:
        self.user_based_collaborative_recommender.load_model(path)

    def load_item_similarity_for_item_collaborative(self, path: Path) -> None:
        self.item_based_collaborative_recommender.load_item_similarity_from_cache(path)

    def load_association_rules(self, path: Path) -> None:
        self.association_rule_recommender = AssociationRuleRecommender.load_model(path)

    def save_model_of_user_collaborative(self, path: Path) -> None:
        self.user_based_collaborative_recommender.save_model(path)

    def save_item_similarity_of_item_collaborative(self, path: Path) -> None:
        self.item_based_collaborative_recommender.cache_item_similarity(path)

    def save_association_rules(self, path: Path) -> None:
        self.association_rule_recommender.cache_rules(path)
