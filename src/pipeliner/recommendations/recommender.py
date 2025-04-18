import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    mean_squared_error,
)


class ItemBasedRecommender(BaseEstimator):
    """Item-based collaborative filtering recommender.

    Args:
        n (int): Number of recommendations to generate for each item.
    """

    n: int
    similarity_matrix: pd.DataFrame
    user_item_matrix: pd.DataFrame

    def __init__(self, n=5):
        self.n = n

    def fit(self, X, y=None):
        """Fits the recommender to the given data.

        Args:
            X (pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]):
                Single DataFrame with similarity matrix
                or tuple of (similarity matrix, user/item matrix)

        Returns:
            self: Returns the instance itself.
        """
        if isinstance(X, pd.DataFrame):
            self.similarity_matrix = X
        elif isinstance(X, tuple):
            self.similarity_matrix = X[0]
            self.user_item_matrix = X[1]
        else:
            raise ValueError("Input should be DataFrame or (DataFrame, DataFrame)")

        return self

    def _get_exclusions(self, item_id: str, user_id: str):
        if user_id is None:
            return [item_id]
        single_user_matrix = self.user_item_matrix.loc[user_id]
        user_rated_items = single_user_matrix[single_user_matrix > 0]
        return [item_id] + user_rated_items.index.to_list()

    def _get_recommendations(self, item) -> np.array:
        if isinstance(item, str):
            item_id, user_id = item, None
        elif isinstance(item, tuple):
            item_id, user_id = item[0], item[1]
        else:
            raise ValueError("Input items should be str or (str, str)")

        exclusions = self._get_exclusions(item_id, user_id)

        item_recommendations = (
            self.similarity_matrix[item_id]
            .drop(exclusions, errors="ignore")
            .sort_values(ascending=False)
        )
        return np.array(item_recommendations.head(self.n).index)

    def predict(self, X) -> np.array:
        """Predicts n item recommendations for each item_id provided
        If tuples of (user_id, item_id) are provided, items previously
        rated by the user will be excluded from the recommendations.

        Args:
          X (Sequence): List of item_id or (item_id, user_id)

        Returns:
          np.array of shape (X.shape[0], n)
        """
        return np.array([self._get_recommendations(item) for item in X])

    # def predict_proba(self, X):
    #     raise NotImplementedError("predict_proba not implemented yet")


class UserBasedRecommender(BaseEstimator):
    """User-based collaborative filtering recommender.

    Args:
        n (int): Number of recommendations to generate for each user
        n_users (int): Number of similar users to consider for recommendations
    """

    n: int
    n_users: int
    similarity_matrix: pd.DataFrame
    user_item_matrix: pd.DataFrame

    def __init__(self, n=5, n_users=5):
        self.n = n
        self.n_users = n_users

    def fit(self, X, y=None):
        """Fits the recommender to the given data.

        Args:
            X (tuple[pd.DataFrame, pd.DataFrame]):
                tuple of (similarity matrix, user/item matrix)

        Returns:
            self: Returns the instance itself.

        Raises:
            ValueError: If input is not a tuple of DataFrames
        """
        if isinstance(X, tuple):
            self.similarity_matrix = X[0]
            self.user_item_matrix = X[1]
        else:
            raise ValueError("Input should be tuple of (DataFrame, DataFrame)")

        return self

    def _get_similar_users(self, user_id: str) -> pd.Series:
        return (
            self.similarity_matrix[user_id]
            .drop(user_id, errors="ignore")
            .sort_values(ascending=False)
            .head(self.n_users)
        )

    def _get_exclusions(self, user_id: str):
        single_user_matrix = self.user_item_matrix.loc[user_id]
        user_rated_items = single_user_matrix[single_user_matrix > 0]
        return user_rated_items.index.to_list()

    def _get_recommendations(self, user_id: str) -> np.array:
        if not isinstance(user_id, str):
            raise ValueError("Input items should be str")
        exclusions = self._get_exclusions(user_id)
        similar_users = self._get_similar_users(user_id)
        matrix = self.user_item_matrix.T[similar_users.index]

        user_recommendations = (
            matrix[~matrix.index.isin(exclusions) & (matrix > 0).any(axis="columns")]
            .max(axis=1)
            .sort_values(ascending=False)
        )

        return np.array(user_recommendations.head(self.n).index)

    def predict(self, X) -> np.array:
        """Predicts n item recommendations for each user_id provided.

        Args:
          X (Sequence): List of user_id

        Returns:
          np.array of shape (X.shape[0], n)
        """
        return np.array([self._get_recommendations(user_id) for user_id in X])

    def score(self, y_preds, y_test):
        """Calculates the accuracy score of the recommender.

        Args:
            y_preds: Predicted recommendations
            y_test: Ground truth recommendations

        Returns:
            float: Accuracy score between 0 and 1
        """
        scores = np.array([1.0 if t in p else 0.0 for t, p in zip(y_test, y_preds)])
        if len(scores) == 0:
            return np.nan
        accuracy = np.mean(scores)
        return accuracy

    # def predict_proba(self, X):
    #     raise NotImplementedError("predict_proba not implemented yet")


class SimilarityRecommender(BaseEstimator):
    """Similarity recommender.

    Args:
        n (int): Number of recommendations to generate.
    """

    n: int
    similarity_matrix: pd.DataFrame

    def __init__(self, n=5):
        self.n = n

    def fit(self, X, y=None):
        """Fits the recommender to the given data.

        Args:
            X pd.DataFrame:
                similarity matrix

        Returns:
            self: Returns the instance itself.
        """
        if isinstance(X, pd.DataFrame):
            self.similarity_matrix = X
        else:
            raise ValueError("Input should be DataFrame")

        return self

    def _get_recommendations(self, item_id) -> np.array:
        item_recommendations = (
            self.similarity_matrix[item_id]
            .drop([item_id], errors="ignore")
            .sort_values(ascending=False, kind="stable")
        )
        return item_recommendations[item_recommendations > 0].index[: self.n].to_numpy()

    def predict(self, X) -> np.array:
        """Predicts n item recommendations for each item_id provided

        Args:
          X (Sequence): List of item_id

        Returns:
          np.array of shape (X.shape[0], n)
        """
        return np.stack(np.vectorize(self._get_recommendations)(X))

    def _get_probabilities(self, item_id) -> np.array:
        return self.similarity_matrix[item_id].to_numpy().astype(np.float32).round(6)

    def predict_proba(self, X):
        return np.array([self._get_probabilities(item_id) for item_id in X])


class SimilarityRecommenderNP(BaseEstimator):
    """Similarity recommender.

    Args:
        n (int): Number of recommendations to generate.
    """

    n: int
    similarity_matrix: sp.spmatrix

    def __init__(self, n=5):
        self.n = n

    def fit(self, X, y=None):
        """Fits the recommender to the given data.

        Args:
            X sp.spmatrix:
                similarity matrix

        Returns:
            self: Returns the instance itself.
        """
        if isinstance(X, sp.spmatrix):
            self.similarity_matrix = X
        else:
            raise ValueError("Input should be scipy.sparse.spmatrix")

        return self

    def _get_recommendations(self, item_id) -> list:
        item_similarity = self.similarity_matrix[item_id].toarray()
        mask = (item_similarity > 0) * (np.arange(item_similarity.size) != item_id)
        sorter = np.argsort(1 - item_similarity, kind="stable")
        sorted_mask = mask[0, sorter]
        results = sorter[sorted_mask]
        return results[: self.n]

    def predict(self, X) -> np.array:
        """Predicts n item recommendations for each item_id provided

        Args:
          X (Sequence): List of item_id

        Returns:
          np.array of shape (X.shape[0], n)
        """
        return [self._get_recommendations(item_id) for item_id in X]

    def predict_proba(self, X):
        return self.similarity_matrix[X]
