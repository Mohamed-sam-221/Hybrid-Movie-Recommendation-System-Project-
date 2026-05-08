import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

ratings = pd.read_csv("ratings.csv")

# user-item matrix
matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

svd = TruncatedSVD(n_components=50, random_state=42)

user_factors = svd.fit_transform(matrix)
item_factors = svd.components_

# reconstruct prediction
pred_matrix = np.dot(user_factors, item_factors)


def predict_rating(user_id, movie_id):
    if user_id not in matrix.index or movie_id not in matrix.columns:
        return matrix.values.mean()  # cold start fix

    u = matrix.index.get_loc(user_id)
    i = matrix.columns.get_loc(movie_id)

    return pred_matrix[u, i]