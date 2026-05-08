import streamlit as st
import pandas as pd
from hybrid_model import hybrid_recommendations

movies = pd.read_csv("movies.csv")

st.title(" Hybrid Movie Recommender (Final Version)")

# ----------------------
# INPUTS
# ----------------------
search = st.sidebar.text_input("Search movie")

filtered = movies['title']
if search:
    filtered = filtered[filtered.str.contains(search, case=False, na=False)]

movie = st.sidebar.selectbox("Select Movie", filtered.values)

user_id = st.sidebar.number_input("User ID", 1, 10000, 1)

genre = st.sidebar.selectbox(
    "Genre Filter",
    ["", "Action", "Comedy", "Drama", "Romance", "Sci-Fi", "Horror"]
)

top_n = st.sidebar.slider("Top N", 5, 20, 10)

w_content = st.sidebar.slider("Content Weight", 0.0, 1.0, 0.5)
w_collab = 1 - w_content

# ----------------------
# BUTTON
# ----------------------
if st.button("Recommend"):

    recs = hybrid_recommendations(
        user_id=user_id,
        movie_title=movie,
        genre=genre if genre != "" else None,
        top_n=top_n,
        w_content=w_content,
        w_collab=w_collab
    )

    if recs.empty:
        st.error("No recommendations found")
    else:
        st.success("Recommendations ready")

        # 🟢 Sorted Table
        st.dataframe(recs)

        # 🟢 Cards
        for _, row in recs.iterrows():
            st.markdown(
                f"**{row['title']}** | "
                f"{row['genres']} | "
                f"Score: {row['hybrid_score']:.3f}"
            )

# ----------------------
# EVALUATION SECTION
# ----------------------
st.divider()
st.subheader(" Model Evaluation (Summary)")

st.write("""
- Content-Based: cosine similarity on TF-IDF
- Collaborative: SVD (TruncatedSVD)
- Hybrid: weighted combination

Metrics used:
- RMSE
- MAE
- Precision / Recall / F1 (binary threshold = 3)
""")