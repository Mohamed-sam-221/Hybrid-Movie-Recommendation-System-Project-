import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("movies.csv")

movies['genres'] = movies['genres'].fillna('')
movies['content'] = movies['title'] + " " + movies['genres']

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf = vectorizer.fit_transform(movies['content'])

cosine_sim = cosine_similarity(tfidf, tfidf)

movie_index = pd.Series(movies.index, index=movies['title']).drop_duplicates()


def content_recommend(title, genre=None, top_n=10):

    if title not in movie_index:
        return pd.DataFrame()

    idx = movie_index[title]

    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:80]

    results = []

    for i, score in scores:
        row = movies.iloc[i]

        #  genre filter
        if genre and genre not in row['genres']:
            continue

        results.append({
            "title": row["title"],
            "genres": row["genres"],
            "content_score": float(score)
        })

    df = pd.DataFrame(results)

    # ترتيب واضح
    return df.sort_values("content_score", ascending=False).head(top_n)