import pytest
import pandas as pd

from association_rule import MiningAlgorithm, AssociationRuleRecommender
from data_loader import DataLoader
from definitions import NotFittedError


@pytest.fixture
def mock_data_loader(mocker):
    mock = mocker.MagicMock(spec=DataLoader)
    mock.load_data.return_value = pd.DataFrame({
        "Customer ID": ["1", "1", "1", "2", "2"],
        "StockCode": ["1", "2", "3", "1", "2"],
        "Description": ["foo", "bar", "baz", "foo", "bar"],
    })
    return mock


@pytest.fixture
def recommender(mock_data_loader):
    return AssociationRuleRecommender(mock_data_loader, MiningAlgorithm.APRIORI)


class TestAssociationRuleRecommender:
    def test_initialisation(self, mock_data_loader):
        recommender = AssociationRuleRecommender(mock_data_loader, MiningAlgorithm.APRIORI)
        assert recommender.mining_algorithm == MiningAlgorithm.APRIORI
        assert recommender._fitted is False

    def test_fit_method(self, recommender):
        recommender.fit()
        assert recommender._fitted is True
        assert recommender.rules is not None

    def test_get_recommendations_without_fit(self, recommender):
        with pytest.raises(NotFittedError):
            recommender.get_recommendations(1234)

    def test_get_recommendations_with_fit(self, recommender):
        recommender.fit()
        result = recommender.get_recommendations(1)

        assert "2" in [r.stock_code for r in result]
        assert "3" in [r.stock_code for r in result]
        assert len(result) == 2

    def test_save_recommender_not_fitted(self, recommender):
        with pytest.raises(NotFittedError):
            recommender.save_model("test.pkl")
