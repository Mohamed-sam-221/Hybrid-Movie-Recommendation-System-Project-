import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse, mae
from sklearn.metrics import precision_score, recall_score, f1_score

ratings = pd.read_csv(r"C:\Users\Sam\AppData\Roaming\JetBrains\PyCharm2025.3\scratches\ptoject intelegan sec\ratings.csv")

reader = Reader(rating_scale=(0.5, 5.0))

data = Dataset.load_from_df(
    ratings[['userId', 'movieId', 'rating']],
    reader
)

trainset, testset = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

model = SVD()
model.fit(trainset)

predictions = model.test(testset)

# RMSE / MAE
print("RMSE:", rmse(predictions))
print("MAE:", mae(predictions))

# Binary classification approximation

actual = []
predicted = []

for pred in predictions:
    actual.append(1 if pred.r_ui >= 3 else 0)
    predicted.append(1 if pred.est >= 3 else 0)

precision = precision_score(actual, predicted)
recall = recall_score(actual, predicted)
f1 = f1_score(actual, predicted)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)