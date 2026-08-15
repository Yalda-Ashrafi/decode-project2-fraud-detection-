# 🛡️ FraudSense AI

An interactive fraud detection dashboard built on a leak-free supervised learning pipeline.

Detecting fraudulent card transactions hidden inside 284,807 records — where fraud accounts for just 0.17% of the data.

Show Image Show Image Show Image Show Image

## 📌 Project Context

This project was built as Project 2 — Supervised Learning (Fraud Detection Pipeline) of the Data Science Industrial Training Kit, powered by DecodeLabs (Batch 2026).

The challenge is not simply classification — it is classification under extreme class imbalance. A model that labels every transaction as legitimate achieves 99.83% accuracy while catching zero fraud. FraudSense AI therefore discards accuracy entirely and is evaluated on Precision, Recall, F1, ROC-AUC and PR-AUC.

The second challenge is data leakage. Applying SMOTE or scaling before the train/test split contaminates the test set with information derived from training data, producing scores that look excellent and mean nothing. This project uses imblearn.pipeline.Pipeline so that resampling and scaling are confined to training folds only.

# ✨ Features
#### Interactive dashboard:  six sections (Home, Dataset, Pipeline, Models, Prediction, About) with a sticky top navigation bar
#### Dataset upload: load the default Kaggle CSV or upload your own transaction file; every page updates automatically
#### Exploratory analysis: class distribution, transaction amount profiles, and per-feature correlation with the fraud label
#### Pipeline explainer: a visual walkthrough of the four-stage leak-free architecture and what goes wrong without it
#### Model comparison: Precision, Recall, F1, ROC-AUC and PR-AUC side by side, with an adjustable decision threshold
#### Interactive visualisations:confusion matrices, ROC curves and Precision–Recall curves rendered with Plotly
#### Live scoring: push a single transaction through the pipeline and see its fraud probability on a gauge
#### Batch scoring: upload a CSV, score every row, review the score distribution, and download annotated results
#### Light and dark themes: switchable from the navigation bar

# 🧰 Tech Stack
Layer	Tools
Language	Python 3.10+
Data	pandas, NumPy
Machine learning	scikit-learn, imbalanced-learn
Visualisation	Plotly, Matplotlib, Seaborn
Interface	Streamlit
Persistence	joblib
# 🧠 Models

Both models are wrapped in an imblearn pipeline so that scaling and SMOTE run inside each cross-validation fold rather than on the full dataset.

python
from imblearn.pipeline import Pipeline   # NOT sklearn.pipeline

Pipeline([
    ("scaler",     StandardScaler()),
    ("smote",      SMOTE(random_state=42)),
    ("classifier", LogisticRegression(max_iter=1000))   # or RandomForestClassifier()
])

#### Logistic Regression — linear decision boundary with transparent coefficients and fast training. Requires scaling: regularisation penalties are distorted by the large variance in transaction amounts. Tends toward high recall with low precision on this dataset.

#### Random Forest — non-linear ensemble, naturally invariant to feature scale because splits are ordinal. Slower to train, but holds a substantially better precision–recall balance.

### Pipeline order
Step	Stage	Why it lives here
1	Stratified split	train_test_split(..., stratify=y) runs first, preserving the exact fraud ratio in both partitions
2	StandardScaler	Fitted on training folds only; also keeps SMOTE's k-nearest-neighbour distances meaningful
3	SMOTE	Synthesises minority points by interpolation: x_new = x_i + λ(x_nn − x_i)
4	Classifier	Trained on the balanced fold; .predict() automatically skips resampling

Tuning — hyperparameters for both the resampler (smote__k_neighbors) and the classifier are searched jointly with GridSearchCV, scored on average precision rather than accuracy.

#  Dataset

Credit Card Fraud Detection — anonymised transactions made by European cardholders over two days in September 2013.

🔗 Download from Kaggle

Property	Value
Rows	284,807
Fraud cases	492 (0.172%)
Features	Time, V1–V28, Amount
Target	Class (0 = legitimate, 1 = fraud)
File size	~144 MB

V1–V28 are principal components released in place of the original features for confidentiality. Time is seconds elapsed since the first transaction; the app converts it to hour-of-day. Amount is the only feature on a materially different scale.

Download creditcard.csv and place it in the project root.

# How to Run
1. Clone and enter the project
bash
git clone <your-repo-url>
cd fraud-detection
2. Create a virtual environment
bash
python -m venv venv

Windows

powershell
venv\Scripts\activate

macOS / Linux

bash
source venv/bin/activate
3. Install dependencies
bash
pip install -r requirements.txt

Or directly:

bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn plotly streamlit joblib
4. Add the dataset

Place creditcard.csv in the project root.

5. Train the models
bash
python 04_train.py

This writes models/lr_pipeline.pkl and models/rf_pipeline.pkl. The Random Forest takes a few minutes.

6. Launch the dashboard
bash
streamlit run app.py

The app opens automatically at http://localhost:8501. If it does not, paste that address into your browser. Keep the terminal open while the app is running; press Ctrl+C to stop it.

# 📁 Project Structure
fraud-detection/
├── creditcard.csv          # dataset (downloaded separately)
├── 01_load.py              # data loading and inspection
├── 02_explore.py           # EDA and class imbalance charts
├── 03_split.py             # stratified train/test split
├── 04_train.py             # imblearn pipelines and model fitting
├── 05_evaluate.py          # metrics, confusion matrix, ROC, PR curves
├── 06_tune.py              # GridSearchCV hyperparameter tuning
├── app.py                  # Streamlit dashboard
├── requirements.txt
└── models/
    ├── lr_pipeline.pkl
    └── rf_pipeline.pkl
#  Deployment

This project can be deployed on Streamlit Cloud for public access.

1. Prepare the repository

Push the project to GitHub with a requirements.txt:

streamlit
pandas
numpy
scikit-learn
imbalanced-learn
plotly
joblib
2. Handle the dataset and models

creditcard.csv is roughly 144 MB and exceeds GitHub's 100 MB file limit. Two options:

Recommended — commit only the trained .pkl files and let users upload a CSV through the dashboard's built-in upload panel
Alternative — track the dataset with Git LFS, or download it at runtime from a hosted URL
3. Deploy
Sign in at share.streamlit.io with your GitHub account
Select New app, then choose your repository, branch, and app.py as the main file
Select Deploy

The app builds in a few minutes and receives a public *.streamlit.app URL. Pushing to the connected branch redeploys automatically.

Optional theme configuration

Add .streamlit/config.toml:

toml
[theme]
base = "dark"
primaryColor = "#8B5CF6"
font = "sans serif"
# Key Findings

Logistic Regression with SMOTE reaches high recall but floods the review queue with false positives. Random Forest holds a far better precision–recall balance at the default threshold.

More importantly: the decision threshold, not the algorithm, determines where an institution sits on that trade-off. Lowering it buys recall at the cost of precision. That choice belongs to the business, not the model.

ROC-AUC also flatters every model on this dataset — with 284,000 negatives, the false-positive rate barely moves. PR-AUC is the more honest metric at 0.17% prevalence.

<div align="center">

Created by Yalda Ashrafi © 2026 — All rights reserved.

Data Science Industrial Training Kit · Powered by DecodeLabs

</div>
