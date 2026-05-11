import streamlit as st
import pandas as pd
from full_code import get_hybrid_recommendations

movies = pd.read_csv("movies.csv")

# استدعاء الدوال التي قمت بكتابتها (get_hybrid_recommendations)

st.set_page_config(page_title="Movie Recommender", page_icon="")

# تصميم الواجهة
st.title(" Hybrid Movie Recommendation System")

# جانب المدخلات
st.sidebar.header("User Settings")
user_id = st.sidebar.number_input("Enter User ID", min_value=1, value=1, step=1)

# تحميل قائمة الأفلام للبحث
movie_list = movies['title'].values
selected_movie = st.sidebar.selectbox("Type or select a movie you liked:", movie_list)

top_n = st.sidebar.slider("Number of recommendations", 5, 20, 10)

# زر التوصية
if st.sidebar.button("Get Recommendations"):
    with st.spinner('Calculating scores...'):
        recommendations = get_hybrid_recommendations(user_id, selected_movie, top_n=top_n)

        if not recommendations.empty:
            st.success(f"Top {top_n} recommendations for User {user_id} based on '{selected_movie}':")

            # عرض النتائج بشكل جذاب
            for i, row in recommendations.iterrows():
                with st.expander(f"{i + 1}. {row['title']}"):
                    st.write(f"**Genres:** {row['genres']}")
                    st.write(f"**Match Score:** {round(row['hybrid_score'] * 100, 2)}%")
                    st.progress(float(row['hybrid_score']))
        else:
            st.error("Sorry, we couldn't find enough similar movies.")

# قسم التقييم (Evaluation Metrics) في أسفل الصفحة
st.divider()
st.subheader(" System Performance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("RMSE", "0.931")
col2.metric("Precision", "76.3%")
col3.metric("Recall", "65.4%")
col4.metric("F1-Score", "70.4%")