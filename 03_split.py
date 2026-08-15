# 03_split.py
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("E:/Decode_Intern/fraud-detection/creditcard.csv")

# Turn Time (seconds since first transaction) into hour-of-day, which is
# actually meaningful. Fraud tends to cluster at unusual hours.
df["Hour"] = (df["Time"] / 3600) % 24
df = df.drop(columns=["Time"])

X = df.drop(columns=["Class"])   # features
y = df["Class"]                  # target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y          # <-- the critical argument
)

print("Train:", X_train.shape, "fraud in train:", y_train.sum())
print("Test :", X_test.shape,  "fraud in test :", y_test.sum())
print("Train fraud rate: {:.4f}%".format(100 * y_train.mean()))
print("Test  fraud rate: {:.4f}%".format(100 * y_test.mean()))