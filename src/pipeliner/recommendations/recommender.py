import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator
from pipeliner.recommendations.transformer import (
    SimilarityTransformer,
)


class UserBasedRecommender(BaseEstimator):
    """User-based collaborative filtering recommender.

    Args:
        n (int): Number of recommendations to generate for each user
        n_users (int): Number of similar users to consider for recommendations
    """

    n: int
    n_users: int

    def __init__(self, n=5, n_users=5):
        self.n = n
        self.n_users = n_users
        self._user_transformer = SimilarityTransformer()

    def fit(self, X: sp.sparray, y=None):
        """Fits the recommender to the given data.

        Args:
            X sp.sparray:
                user/item matrix

        Returns:
            self: Returns the instance itself.

        Raises:
            ValueError: If input is not a scipy.sparse.sparray
        """
        if isinstance(X, sp.sparray):
            self._user_item_matrix = X
            self._user_indices = np.arange(X.shape[0])
            self._item_indices = np.arange(X.shape[1])
            self._user_similarity_matrix = self._user_transformer.transform(X)
        else:
            raise ValueError("Input should be scipy.sparse.sparray")

        return self

    def _get_similar_users(self, id: int) -> np.array:
        matrix = self._user_similarity_matrix[[id]]
        user_mask = matrix > 0
        user_mask[[0], [id]] = False
        user_sorter = np.argsort(1 - matrix.toarray()[0], kind="stable")
        sorted_mask = user_mask.toarray()[0][user_sorter]
        similar_users = user_sorter[sorted_mask][: self.n]

        return similar_users

    def _get_exclusions(self, id: int) -> np.array:
        single_user_ratings = self._user_item_matrix[[id]]
        rated = (single_user_ratings > 0).nonzero()[1]
        return rated

    def _get_recommendations(self, id: int) -> np.array:
        excluded_items = self._get_exclusions(id)
        similar_users = self._get_similar_users(id)

        matrix = self._user_item_matrix[similar_users]

        any_ratings = np.nonzero(matrix.sum(axis=0))[0]
        items_to_use = np.setdiff1d(any_ratings, excluded_items)

        filtered_matrix = matrix[:, items_to_use]

        mean_ratings = filtered_matrix.toarray().T.mean(axis=1)
        item_sorter = np.argsort(1 - mean_ratings, kind="stable")

        return items_to_use[item_sorter][: self.n]

    def predict(self, X) -> list[np.array]:
        """Predicts n recommendations for each id provided

        Args:
          X (Sequence): List of id

        Returns:
          list of np.array
        """
        return [self._get_recommendations(id) for id in X]


class SimilarityRecommender(BaseEstimator):
    """Similarity recommender.

    Args:
        n (int): Number of recommendations to generate.
    """

    n: int
    similarity_matrix: sp.sparray

    def __init__(self, n=5):
        self.n = n

    def fit(self, X, y=None):
        """Fits the recommender to the given data.

        Args:
            X sp.sparray:
                similarity matrix

        Returns:
            self: Returns the instance itself.

        Raises:
            ValueError: If input is not a scipy.sparse.sparray
        """
        if isinstance(X, sp.sparray):
            self.similarity_matrix = X
        else:
            raise ValueError("Input should be scipy.sparse.sparray")

        return self

    def _get_recommendations(self, id) -> np.array:
        item_similarity = self.similarity_matrix[[id], :].toarray()
        mask = (item_similarity > 0) * (np.arange(item_similarity.size) != id)
        sorter = np.argsort(1 - item_similarity, kind="stable")
        sorted_mask = mask[0, sorter]
        return sorter[sorted_mask][: self.n]

    def predict(self, X) -> list[np.array]:
        """Predicts n recommendations for each id provided

        Args:
          X (Sequence): List of id

        Returns:
          list of np.array
        """
        return [self._get_recommendations(id) for id in X]

    def predict_proba(self, X):
        return self.similarity_matrix[X]
