import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# 1. تحميل البيانات
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# 2. تنظيف البيانات
movies['genres'] = movies['genres'].fillna('Unknown')
movies['content'] = movies['title'] + " " + movies['genres']

# تقسيم البيانات قبل أي معالجة لضمان تقييم حقيقي (Train/Test Split)
train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)

# 3. Collaborative Filtering (Matrix Factorization - SVD)
# تمركز المتوسط (Mean Centering) لحل مشكلة الـ fillna(0)
user_ratings_mean = train_df.groupby('userId')['rating'].mean()
train_df_centered = train_df.copy()
train_df_centered['rating'] = train_df_centered.apply(lambda x: x['rating'] - user_ratings_mean[x['userId']], axis=1)

# إنشاء المصفوفة
matrix = train_df_centered.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

# تطبيق SVD
svd = TruncatedSVD(n_components=50, random_state=42)
user_factors = svd.fit_transform(matrix)
item_factors = svd.components_

# إعادة بناء المصفوفة المتوقعة
pred_matrix = np.dot(user_factors, item_factors)
pred_df = pd.DataFrame(pred_matrix, index=matrix.index, columns=matrix.columns)


def predict_rating(user_id, movie_id):
    # إذا كان المستخدم أو الفيلم غير موجود (Cold Start)
    if user_id not in user_ratings_mean.index:
        return 3.5  # متوسط عام

    user_avg = user_ratings_mean[user_id]

    if movie_id in pred_df.columns:
        # استرجاع القيمة المتوقعة وإضافة المتوسط الخاص بالمستخدم مرة أخرى
        pred_val = pred_df.loc[user_id, movie_id]
        return pred_val + user_avg
    else:
        return user_avg


# 4. Content-Based Filtering
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = vectorizer.fit_transform(movies['content'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
movie_index = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# 5. Hybrid Recommendation Function
movie_id_map = dict(zip(movies.title, movies.movieId))


def get_hybrid_recommendations(user_id, movie_title, top_n=10):
    if movie_title not in movie_index:
        return pd.DataFrame()

    idx = movie_index[movie_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:51]  # أفضل 50 بناءً على المحتوى

    results = []
    for i, score in sim_scores:
        m_row = movies.iloc[i]
        m_id = m_row['movieId']

        content_score = score
        # توقع التقييم وتطبيعه ليكون بين 0 و 1
        collab_score = predict_rating(user_id, m_id) / 5.0

        # دمج النتيجتين
        hybrid_score = (content_score * 0.5) + (collab_score * 0.5)

        results.append({
            "title": m_row["title"],
            "genres": m_row["genres"],
            "hybrid_score": hybrid_score
        })

    return pd.DataFrame(results).sort_values("hybrid_score", ascending=False).head(top_n)


# 6. التقييم (Evaluation)
test_predictions = []
test_actuals = []

for _, row in test_df.iterrows():
    test_actuals.append(row['rating'])
    test_predictions.append(predict_rating(row['userId'], row['movieId']))

rmse = np.sqrt(np.mean((np.array(test_actuals) - np.array(test_predictions)) ** 2))
print(f"Improved RMSE: {rmse}")

# Classification metrics
a_bin = [1 if x >= 3.5 else 0 for x in test_actuals]
p_bin = [1 if x >= 3.5 else 0 for x in test_predictions]
f1 = f1_score(a_bin, p_bin)

print(f"Precision: {precision_score(a_bin, p_bin)}")
print(f"Recall: {recall_score(a_bin, p_bin)}")
print(f"F1-Score: {f1}")