import pandas as pd
from content_based import content_recommend
from collaborative import predict_rating

movies = pd.read_csv("movies.csv")
movie_id_map = dict(zip(movies.title, movies.movieId))


def hybrid_recommendations(user_id, movie_title,
                           genre=None,
                           top_n=10,
                           w_content=0.5,
                           w_collab=0.5):

    content_df = content_recommend(movie_title, genre=genre, top_n=80)

    if content_df.empty:
        return pd.DataFrame()

    results = []

    for _, row in content_df.iterrows():

        movie_id = movie_id_map.get(row["title"])

        if movie_id is None:
            continue

        content_score = row["content_score"]
        collab_score = predict_rating(user_id, movie_id) / 5.0

        hybrid = (w_content * content_score) + (w_collab * collab_score)

        results.append({
            "title": row["title"],
            "genres": row["genres"],
            "content_score": content_score,
            "collab_score": collab_score,
            "hybrid_score": hybrid
        })

    df = pd.DataFrame(results)

    return df.sort_values("hybrid_score", ascending=False).head(top_n)