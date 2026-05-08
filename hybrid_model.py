import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset
from surprise import Reader
from surprise import SVD

# Load datasets
movies_path = r"C:\Users\Sam\AppData\Roaming\JetBrains\PyCharm2025.3\scratches\ptoject intelegan sec\movies.csv"
ratings_path = r"C:\Users\Sam\AppData\Roaming\JetBrains\PyCharm2025.3\scratches\ptoject intelegan sec\ratings.csv"

movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)
# -----------------------------
# CONTENT-BASED PART
# -----------------------------

movies['genres'] = movies['genres'].fillna('')

vectorizer = TfidfVectorizer(stop_words='english')
movies['content'] = movies['title'] + " " + movies['genres']
tfidf_matrix = vectorizer.fit_transform(movies['content'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

movie_indices = pd.Series(
    movies.index,
    index=movies['title']
).drop_duplicates()
# COLLABORATIVE PART

reader = Reader(rating_scale=(0.5, 5.0))

data = Dataset.load_from_df(
    ratings[['userId', 'movieId', 'rating']],
    reader
)

trainset = data.build_full_trainset()

svd_model = SVD()
svd_model.fit(trainset)

# HYBRID FUNCTION
def hybrid_recommendations(user_id, movie_title, top_n=10,
                           content_weight=0.4,
                           collaborative_weight=0.6):

    if movie_title not in movie_indices:
        return pd.DataFrame()

    idx = movie_indices[movie_title]

    similarity_scores = list(enumerate(cosine_sim[idx]))

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )[1:50]

    recommendations = []

    for movie_idx, content_score in similarity_scores:

        movie_id = movies.iloc[movie_idx]['movieId']
        title = movies.iloc[movie_idx]['title']

        # Collaborative score (0–1 scaling)
        collaborative_score = svd_model.predict(user_id, movie_id).est / 5.0
        #  NO min-max normalization needed
        # cosine similarity already normalized

        hybrid_score = (
            content_weight * content_score +
            collaborative_weight * collaborative_score
        )

        recommendations.append({
            'title': title,
            'content_score': content_score,
            'collaborative_score': collaborative_score,
            'hybrid_score': hybrid_score
        })
        print(content_score, collaborative_score, hybrid_score)

    recommendation_df = pd.DataFrame(recommendations)

    recommendation_df = recommendation_df.drop_duplicates(subset=['title'])

    recommendation_df = recommendation_df.sort_values(
        by='hybrid_score',
        ascending=False

    )

    return recommendation_df.head(top_n)
# Example
if __name__ == "__main__":

    result = hybrid_recommendations(
        user_id=1,
        movie_title="Toy Story (1995)",
        top_n=10
    )

    print(result)