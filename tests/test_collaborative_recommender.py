from pathlib import Path
import pytest
from pandas import DataFrame
import numpy as np
from definitions import ImplicitModel, NotFittedError
from data_loader import DataLoader
from collaborative_recommender import CollaborativeRecommender


@pytest.fixture
def mock_data_loader(mocker):
    mock = mocker.MagicMock(spec=DataLoader)
    mock.load_data.return_value = DataFrame({
        "Customer ID": ["1", "2", "3"],
        "StockCode": ["01", "10", "11"],
        "Description": ["foo", "bar", "baz"],
        "Quantity": [5, 5, 5],
    })
    return mock


@pytest.fixture
def mock_model(mocker):
    return mocker.MagicMock(spec=ImplicitModel)


@pytest.fixture
def recommender(mock_data_loader, mock_model):
    return CollaborativeRecommender(mock_data_loader, mock_model)


class TestCollaborativeRecommender:
    def test_init(self, recommender):
        assert recommender._fitted is False

    def test_fit_fitted(self, recommender):
        recommender.fit()
        assert recommender._fitted is True

    def test_get_recommendations_not_fitted(self, recommender):
        with pytest.raises(NotFittedError):
            recommender.get_recommendations(1)

    def test_get_recommendations(self, recommender):
        # Mock the return values
        recommender.model.recommend.return_value = (np.array([1, 2, 3]), np.array([0.8, 0.7, 0.6]))
        recommender.fit()
        recommender.get_recommendations(1)

    def test_save_model_not_fitted(self, recommender):
        with pytest.raises(NotFittedError):
            recommender.save_model(Path('/path/to/save'))
