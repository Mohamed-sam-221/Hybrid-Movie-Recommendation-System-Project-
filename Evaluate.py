import pandas as pd
import numpy as np
from collaborative import predict_rating

ratings = pd.read_csv(r"C:\Users\Sam\AppData\Roaming\JetBrains\PyCharm2025.3\scratches\ptoject intelegan sec\ratings.csv")

sample = ratings.sample(5000, random_state=42)

actual = []
predicted = []

for _, row in sample.iterrows():

    u = row['userId']
    m = row['movieId']

    actual.append(row['rating'])
    predicted.append(predict_rating(u, m))

# RMSE
rmse = np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2))

# MAE
mae = np.mean(np.abs(np.array(actual) - np.array(predicted)))

print("RMSE:", rmse)
print("MAE:", mae)

# classification metrics
from sklearn.metrics import precision_score, recall_score, f1_score

a_bin = [1 if x >= 3 else 0 for x in actual]
p_bin = [1 if x >= 3 else 0 for x in predicted]

print("Precision:", precision_score(a_bin, p_bin))
print("Recall:", recall_score(a_bin, p_bin))
print("F1:", f1_score(a_bin, p_bin))