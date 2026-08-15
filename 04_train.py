import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline          # NOT sklearn.pipeline
import joblib, os


df = pd.read_csv("creditcard.csv")   
df["Hour"] = (df["Time"] / 3600) % 24
df = df.drop(columns=["Time"])
X, y = df.drop(columns=["Class"]), df["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

# ---------- Pipeline A: Logistic Regression ----------
lr_pipeline = Pipeline([
    ("scaler",     StandardScaler()),
    ("smote",      SMOTE(random_state=42, k_neighbors=5)),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

# ---------- Pipeline B: Random Forest ----------
rf_pipeline = Pipeline([
    ("scaler",     StandardScaler()),   
    ("smote",      SMOTE(random_state=42, k_neighbors=5)),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        n_jobs=-1,
        random_state=42))
])

print("Training Logistic Regression...")
lr_pipeline.fit(X_train, y_train)

print("Training Random Forest (this takes a few minutes)...")
rf_pipeline.fit(X_train, y_train)

print("Saving models now...")
os.makedirs("models", exist_ok=True)
joblib.dump(lr_pipeline, "models/lr_pipeline.pkl")
joblib.dump(rf_pipeline, "models/rf_pipeline.pkl")
print("Models saved to:", os.path.abspath("models"))
