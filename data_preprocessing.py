import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load datasets
movies_path = r"C:\Users\Sam\AppData\Roaming\JetBrains\PyCharm2025.3\scratches\ptoject intelegan sec\movies.csv"
ratings_path = r"C:\Users\Sam\AppData\Roaming\JetBrains\PyCharm2025.3\scratches\ptoject intelegan sec\ratings.csv"

movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)

# Remove duplicates
movies.drop_duplicates(inplace=True)
ratings.drop_duplicates(inplace=True)

# Missing values
movies['genres'] = movies['genres'].fillna('Unknown')

# Basic EDA
print(movies.head())
print(ratings.head())

print(movies.isnull().sum())
print(ratings.isnull().sum())

print(ratings['rating'].describe())
print(movies['genres'].value_counts().head(10))

# Normalize ratings (IMPORTANT FIX)
scaler = MinMaxScaler()
ratings['rating_norm'] = scaler.fit_transform(ratings[['rating']])

# Merge datasets
merged_data = pd.merge(ratings, movies, on='movieId')

print("Merged shape:", merged_data.shape)

# Save dataset
merged_data.to_csv("cleaned_movies_data.csv", index=False)

print("Preprocessing completed successfully")