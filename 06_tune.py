# 06_tune.py
import pandas as pd, joblib, time
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

df = pd.read_csv("creditcard.csv")
df["Hour"] = (df["Time"] / 3600) % 24
df = df.drop(columns=["Time"])
X, y = df.drop(columns=["Class"]), df["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---------- Logistic Regression grid ----------
lr_pipeline = Pipeline([
    ("scaler",     StandardScaler()),
    ("smote",      SMOTE(random_state=42)),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

lr_grid = {
    "smote__k_neighbors":  [3, 5, 7],
    "classifier__C":       [0.01, 0.1, 1.0],
}

lr_search = GridSearchCV(
    estimator=lr_pipeline,
    param_grid=lr_grid,
    scoring="average_precision",   # PR-AUC. NEVER "accuracy" here.
    cv=cv,
    n_jobs=-1,
    verbose=2,
    refit=True
)

start = time.time()
lr_search.fit(X_train, y_train)
print(f"LR done in {time.time()-start:.0f}s")
print("Best params:", lr_search.best_params_)
print("Best CV score:", round(lr_search.best_score_, 4))
joblib.dump(lr_search.best_estimator_, "models/lr_tuned.pkl")

# ---------- Random Forest (FAST tuning) ----------
from sklearn.model_selection import RandomizedSearchCV

rf_pipeline = Pipeline([
    ("scaler",     StandardScaler()),
    ("smote",      SMOTE(random_state=42)),
    ("classifier", RandomForestClassifier(n_jobs=-1, random_state=42))
])

rf_grid = {
    "smote__k_neighbors":          [3, 5],
    "classifier__n_estimators":    [100],          # only 100 trees
    "classifier__max_depth":       [10, None],     # shallow vs unlimited
    "classifier__min_samples_leaf": [1, 5]         # small vs larger leaves
}

rf_search = RandomizedSearchCV(
    rf_pipeline,
    rf_grid,
    n_iter=6,                      
    scoring="average_precision",
    cv=3,                          
    n_jobs=-1,
    random_state=42,
    verbose=2
)

start = time.time()
rf_search.fit(X_train, y_train)
print(f"RF done in {time.time()-start:.0f}s")
print("Best params:", rf_search.best_params_)
print("Best CV score:", round(rf_search.best_score_, 4))
joblib.dump(rf_search.best_estimator_, "models/rf_tuned.pkl")
