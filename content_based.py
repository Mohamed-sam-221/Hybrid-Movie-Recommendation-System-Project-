import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv(r"C:\Users\Sam\AppData\Roaming\JetBrains\PyCharm2025.3\scratches\ptoject intelegan sec\movies.csv")

# Handle missing values
movies['genres'] = movies['genres'].fillna('')

#  أهم تحسين: دمج features
movies['content'] = movies['title'] + " " + movies['genres']

# TF-IDF (مرة واحدة فقط)
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1,2),
    max_features=5000
)

tfidf_matrix = vectorizer.fit_transform(movies['content'])

# Cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Mapping
movie_indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()


def get_content_recommendations(title, num_recommendations=10):

    if title not in movie_indices:
        return ["Movie not found"]

    idx = movie_indices[title]

    similarity_scores = list(enumerate(cosine_sim[idx]))

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similarity_scores = similarity_scores[1:num_recommendations + 1]

    movie_indices_list = [i[0] for i in similarity_scores]

    return movies['title'].iloc[movie_indices_list].tolist()


# Test
if __name__ == "__main__":
    recs = get_content_recommendations("Toy Story (1995)")

    print("Content-Based Recommendations:")
    for r in recs:
        print(r)