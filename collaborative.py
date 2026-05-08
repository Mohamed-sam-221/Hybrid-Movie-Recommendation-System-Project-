import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse, mae

# Load data
ratings = pd.read_csv(r"C:\Users\Sam\AppData\Roaming\JetBrains\PyCharm2025.3\scratches\ptoject intelegan sec\ratings.csv")

# Reader
reader = Reader(rating_scale=(0.5, 5.0))

data = Dataset.load_from_df(
    ratings[['userId', 'movieId', 'rating']],
    reader
)

# Split
trainset, testset = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

#  ONE MODEL ONLY (correct)
svd_model = SVD(
    n_factors=100,
    n_epochs=20,
    lr_all=0.005,
    reg_all=0.02,
    random_state=42
)

# Train
svd_model.fit(trainset)

# Predict
predictions = svd_model.test(testset)

# Evaluation
print("RMSE:")
rmse(predictions)

print("MAE:")
mae(predictions)


def predict_rating(user_id, movie_id):
    return svd_model.predict(user_id, movie_id).est


# Test
if __name__ == "__main__":
    print(predict_rating(1, 10))