import pandas as pd
from sklearn.preprocessing import MinMaxScaler

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# تنظيف
movies = movies.drop_duplicates()
ratings = ratings.drop_duplicates()

movies['genres'] = movies['genres'].fillna('Unknown')

# normalize ratings
scaler = MinMaxScaler()
ratings['rating_norm'] = scaler.fit_transform(ratings[['rating']])

# merge
data = pd.merge(ratings, movies, on="movieId")

data.to_csv("cleaned_data.csv", index=False)

print("Preprocessing done")