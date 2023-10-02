from pathlib import Path

import pytest
import pandas as pd
import numpy as np

from recommenders.collaborative_recommender import UserBasedCollaborativeRecommender, ItemBasedCollaborativeRecommender
from definitions import ImplicitModel, NotFittedError
from data_loader import DataLoader


@pytest.fixture
def mock_data_loader(mocker):
    mock = mocker.MagicMock(spec=DataLoader)
    mock.load_data.return_value = pd.DataFrame({
        "Customer ID": ["1", "1", "2", "2", "3"],
        "StockCode": ["1", "2", "1", "2", "1"],
        "Description": ["foo", "bar", "foo", "bar", "foo"],
        "Quantity": [5, 5, 5, 5, 5],
    })
    return mock


@pytest.fixture
def mock_model(mocker):
    return mocker.MagicMock(spec=ImplicitModel)


@pytest.fixture
def user_based_recommender(mock_data_loader, mock_model):
    return UserBasedCollaborativeRecommender(mock_data_loader, mock_model)


@pytest.fixture
def item_based_recommender(mock_data_loader):
    return ItemBasedCollaborativeRecommender(mock_data_loader)


class TestUserBasedCollaborativeRecommender:
    def test_init(self, user_based_recommender):
        assert user_based_recommender.fitted is False

    def test_fit_fitted(self, user_based_recommender):
        user_based_recommender.fit()
        assert user_based_recommender.fitted is True

    def test_get_recommendations_not_fitted(self, user_based_recommender):
        with pytest.raises(NotFittedError):
            user_based_recommender.get_recommendations(1)

    def test_get_recommendations(self, user_based_recommender):
        # Mock the return values
        user_based_recommender.model.recommend.return_value = (np.array([1, 2, 3]), np.array([0.8, 0.7, 0.6]))
        user_based_recommender.fit()
        result = user_based_recommender.get_recommendations(1)

        assert "2" in [r.stock_code for r in result]

    def test_save_model_not_fitted(self, user_based_recommender):
        with pytest.raises(NotFittedError):
            user_based_recommender.save_model(Path("/path/to/save"))


class TestItemBasedCollaborativeRecommender:
    def test_init(self, item_based_recommender):
        assert item_based_recommender.fitted is False

    def test_fit_fitted(self, item_based_recommender):
        item_based_recommender.fit()
        assert item_based_recommender.fitted is True

    def test_get_recommendations_not_fitted(self, item_based_recommender):
        with pytest.raises(NotFittedError):
            item_based_recommender.get_recommendations(1)

    def test_get_recommendations(self, item_based_recommender):
        item_based_recommender.fit()
        result = item_based_recommender.get_recommendations(1)

        assert "2" in [r.stock_code for r in result]

    def test_cache_item_similarity_not_fitted(self, item_based_recommender):
        with pytest.raises(NotFittedError):
            item_based_recommender.cache_item_similarity(Path("/path/to/save"))
