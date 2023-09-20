from typing import Protocol

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


class NotFittedError(RuntimeError):
    pass


class ImplicitModel(Protocol):
    def fit(self, item_user_data: coo_matrix) -> None:
        ...

    def recommend(
        self,
        user_id: int,
        item_user_data: csr_matrix,
        N: int,
        filter_already_liked_items: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        ...
