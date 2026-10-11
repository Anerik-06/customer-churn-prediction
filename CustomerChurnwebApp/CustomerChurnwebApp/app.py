
from flask import Flask, render_template, request
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

app = Flask(__name__)

# Load the customer churn dataset
df = pd.read_csv("customer_churn.csv")
df.columns = df.columns.str.strip()

# Convert TotalCharges to numeric values
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

df["Churn"] = df["Churn"].astype(str).str.strip()
df = df[df["Churn"].isin(["Yes", "No"])].copy()

# Prepare input features and target
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

# Preprocess numerical and categorical columns
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ]), numeric_cols),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_cols)
])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Baseline model
normal_model = Pipeline([
    ("prep", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

# Model with class-imbalance handling
balanced_model = Pipeline([
    ("prep", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000, class_weight="balanced"
    ))
])

normal_model.fit(X_train, y_train)
balanced_model.fit(X_train, y_train)

# Evaluate each model
def evaluate(model):
    predictions = model.predict(X_test)

    return {
        "accuracy": round(
            accuracy_score(y_test, predictions) * 100, 2
        ),
        "precision": round(
            precision_score(y_test, predictions, zero_division=0) * 100, 2
        ),
        "recall": round(
            recall_score(y_test, predictions, zero_division=0) * 100, 2
        ),
        "f1": round(
            f1_score(y_test, predictions, zero_division=0) * 100, 2
        ),
        "matrix": confusion_matrix(y_test, predictions).tolist()
    }

normal_results = evaluate(normal_model)
balanced_results = evaluate(balanced_model)


@app.route("/")
def home():
    total = len(df)
    churners = int((df["Churn"] == "Yes").sum())

    return render_template(
        "index.html",
        total=total,
        churners=churners,
        churn_rate=round(churners / total * 100, 2),
        normal=normal_results,
        balanced=balanced_results,
        prediction=None
    )


@app.route("/predict", methods=["POST"])
def predict():
    customer_id = request.form["customer_id"].strip()

    customer = df[
        df["customerID"].astype(str) == customer_id
    ]

    if customer.empty:
        message = "Customer ID not found. Please try a valid ID."
    else:
        row = customer.drop(
            columns=["customerID", "Churn"]
        ).iloc[[0]]

        result = balanced_model.predict(row)[0]
        probability = (
            balanced_model.predict_proba(row)[0][1] * 100
        )

        status = "Likely to Churn" if result == 1 else "Likely to Stay"

        message = (
            f"{status} | Estimated churn probability: "
            f"{probability:.2f}%"
        )

    return render_template(
        "index.html",
        total=len(df),
        churners=int((df["Churn"] == "Yes").sum()),
        churn_rate=round((df["Churn"] == "Yes").mean() * 100, 2),
        normal=normal_results,
        balanced=balanced_results,
        prediction=message
    )


if __name__ == "__main__":
    app.run(debug=True)
