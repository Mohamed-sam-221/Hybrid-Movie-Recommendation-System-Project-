import streamlit as st
import sys
sys.path.append(".")

from hybrid_model import hybrid_recommendations
import pandas as pd

movies = pd.read_csv(
    r"C:\Users\Sam\AppData\Roaming\JetBrains\PyCharm2025.3\scratches\ptoject intelegan sec\movies.csv"
)
st.set_page_config(page_title="Hybrid Recommender", layout="wide")

st.title(" Hybrid Movie Recommendation System")

# الأفضل: dropdown بدل text input
search = st.sidebar.text_input("Search movie")

filtered_movies = movies['title']

if search:
    filtered_movies = filtered_movies[filtered_movies.str.contains(search, case=False, na=False)]

movie_title = st.sidebar.selectbox(
    "Select Movie",
    filtered_movies.values
)
user_id = st.sidebar.number_input("User ID", 1, 10000, 1)

num_recommendations = st.sidebar.slider("num Recommendation", 5, 20, 10)

content_weight = st.sidebar.slider("Content Weight", 0.0, 1.0, 0.4)
collab_weight = 1 - content_weight

if st.button("Recommend"):

    recs = hybrid_recommendations(
        user_id=user_id,
        movie_title=movie_title,
        top_n=num_recommendations,
        content_weight=content_weight,
        collaborative_weight=collab_weight

    )

    if recs.empty:
        st.error("No recommendations found")
    else:
        st.success("Done")

        st.dataframe(recs)

        for _, row in recs.iterrows():
            st.markdown(f"**{row['title']}** - {row['hybrid_score']:.3f}")
