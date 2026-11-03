import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
from sklearn.cluster import KMeans

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    log_loss,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================================
# DATASET PATH
# ==========================================================

FILE_PATH = r"C:\Users\User\Downloads\project_dataset.csv"

OUTPUT_FOLDER = os.path.join(
    os.path.dirname(FILE_PATH),
    "Mandi_ML_Results"
)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ==========================================================
# LOAD AND PREPARE DATASET
# ==========================================================

print("\n========== AGRICULTURAL MANDI DATASET ==========")

if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(
        "Dataset not found. Check the FILE_PATH at the beginning of the code."
    )

df = pd.read_csv(FILE_PATH)

# The actual dataset has price headers containing _x0020_.
df.columns = [
    str(col).strip().replace("_x0020_", "_").replace(" ", "_")
    for col in df.columns
]

required_columns = [
    "State", "District", "Market", "Commodity", "Variety",
    "Grade", "Arrival_Date", "Min_Price", "Max_Price", "Modal_Price"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Required columns not found in dataset: " + str(missing_columns)
    )

categorical_columns = [
    "State", "District", "Market", "Commodity", "Variety", "Grade"
]

for col in categorical_columns:
    df[col] = df[col].fillna("Unknown").astype(str).str.strip()

for col in ["Min_Price", "Max_Price", "Modal_Price"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.dropna(
    subset=["Min_Price", "Max_Price", "Modal_Price"],
    inplace=True
)
df = df[df["Modal_Price"] >= 0].copy()
df.reset_index(drop=True, inplace=True)

print("Dataset Loaded Successfully!")
print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])

print("\nFirst Five Records:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df[["Min_Price", "Max_Price", "Modal_Price"]].describe())

print("\nMissing Values:")
print(df.isnull().sum())

df.to_csv(
    os.path.join(OUTPUT_FOLDER, "cleaned_mandi_dataset.csv"),
    index=False
)


# ==========================================================
# PREPROCESSING FUNCTIONS
# ==========================================================

def create_onehot_encoder():
    # Compatible with both recent and older scikit-learn versions.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def create_preprocessor(categorical_features, numeric_features,
                        polynomial=False):
    transformers = []

    if categorical_features:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", create_onehot_encoder())
        ])
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    if numeric_features:
        numeric_steps = [
            ("imputer", SimpleImputer(strategy="median"))
        ]

        if polynomial:
            numeric_steps.append(
                ("polynomial", PolynomialFeatures(
                    degree=2, include_bias=False
                ))
            )

        numeric_steps.append(("scaler", StandardScaler()))

        numeric_pipeline = Pipeline(numeric_steps)
        transformers.append(
            ("numeric", numeric_pipeline, numeric_features)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


def evaluate_classifier(model, model_name, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    logloss = log_loss(y_test, probabilities, labels=[0, 1])
    auc = roc_auc_score(y_test, probabilities)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    print("\n", model_name, "Results")
    print("Accuracy:", round(accuracy, 4))
    print("Log Loss:", round(logloss, 4))
    print("ROC-AUC:", round(auc, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    print("\nConfusion Matrix:\n", cm)

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Low Price", "High Price"],
        zero_division=0
    ))

    classification_results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Log Loss": logloss,
        "ROC-AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Low Price", "High Price"],
        yticklabels=["Low Price", "High Price"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - " + model_name)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "Confusion_Matrix_" + model_name.replace(" ", "_") + ".png"
        )
    )
    plt.show()


# ==========================================================
# 1. LINEAR REGRESSION
# Predict Modal Price using mandi price information
# ==========================================================

print("\n========== 1. LINEAR REGRESSION ==========")

# Predict Modal_Price using the category fields and Min/Max prices.
# This is price-assisted estimation, not future-price forecasting.

regression_features = categorical_columns + ["Min_Price", "Max_Price"]

X = df[regression_features]
y = df["Modal_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

model = Pipeline([
    ("preprocessing", create_preprocessor(
        categorical_columns, ["Min_Price", "Max_Price"]
    )),
    ("model", LinearRegression())
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("Predicted Modal Prices (first 10):", predictions[:10])
print("MAE:", round(mae, 3))
print("MSE:", round(mse, 3))
print("RMSE:", round(rmse, 3))
print("R2 Score:", round(r2, 4))

# LinearRegression uses Ordinary Least Squares estimation.
ols_model = model.named_steps["model"]
print("OLS Intercept:", ols_model.intercept_)
print("Number of OLS Coefficients:", len(ols_model.coef_))

plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.6)
lower = min(y_test.min(), predictions.min())
upper = max(y_test.max(), predictions.max())
plt.plot(
    [lower, upper],
    [lower, upper],
    color="red",
    linestyle="--",
    label="Ideal Prediction"
)
plt.xlabel("Actual Modal Price")
plt.ylabel("Predicted Modal Price")
plt.title("Linear Regression - Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "01_Linear_Regression.png"))
plt.show()


# ==========================================================
# POLYNOMIAL REGRESSION
# Additional regression technique from the syllabus
# ==========================================================

print("\n========== POLYNOMIAL REGRESSION ==========")

polynomial_model = Pipeline([
    ("preprocessing", create_preprocessor(
        categorical_columns,
        ["Min_Price", "Max_Price"],
        polynomial=True
    )),
    ("model", LinearRegression())
])

polynomial_model.fit(X_train, y_train)
poly_predictions = polynomial_model.predict(X_test)

poly_mae = mean_absolute_error(y_test, poly_predictions)
poly_mse = mean_squared_error(y_test, poly_predictions)
poly_rmse = np.sqrt(poly_mse)
poly_r2 = r2_score(y_test, poly_predictions)

print("Polynomial Regression (Degree 2)")
print("MAE:", round(poly_mae, 3))
print("MSE:", round(poly_mse, 3))
print("RMSE:", round(poly_rmse, 3))
print("R2 Score:", round(poly_r2, 4))

regression_comparison = pd.DataFrame([
    {
        "Model": "Linear Regression (OLS)",
        "MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r2
    },
    {
        "Model": "Polynomial Regression",
        "MAE": poly_mae, "MSE": poly_mse,
        "RMSE": poly_rmse, "R2 Score": poly_r2
    }
])

print("\nRegression Model Comparison:")
print(regression_comparison.round(4))

regression_comparison.to_csv(
    os.path.join(OUTPUT_FOLDER, "Regression_Comparison.csv"),
    index=False
)

plt.figure(figsize=(8, 5))
plt.bar(
    ["Linear Regression", "Polynomial Regression"],
    [rmse, poly_rmse]
)
plt.ylabel("RMSE")
plt.title("Regression Model Comparison")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "02_Regression_Comparison.png"))
plt.show()


# ==========================================================
# CORRELATION ANALYSIS
# ==========================================================

print("\n========== CORRELATION ANALYSIS ==========")

correlation_matrix = df[
    ["Min_Price", "Max_Price", "Modal_Price"]
].corr()

print("Correlation Matrix:")
print(correlation_matrix.round(4))

correlation_matrix.to_csv(
    os.path.join(OUTPUT_FOLDER, "Correlation_Matrix.csv")
)

plt.figure(figsize=(7, 5))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    fmt=".2f"
)
plt.title("Correlation Heatmap of Mandi Prices")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "03_Correlation_Heatmap.png"))
plt.show()


# ==========================================================
# CREATE CLASSIFICATION TARGET
# High Price = Modal_Price above its commodity median
# Low Price = Modal_Price at or below its commodity median
# ==========================================================

print("\n========== PREPARING CLASSIFICATION DATA ==========")

commodity_counts = df.groupby("Commodity")["Modal_Price"].transform("count")
commodity_medians = df.groupby("Commodity")["Modal_Price"].transform("median")

classification_df = df[commodity_counts >= 2].copy()
classification_df["Commodity_Median"] = commodity_medians.loc[
    classification_df.index
]

classification_df["Price_Category"] = np.where(
    classification_df["Modal_Price"] >
    classification_df["Commodity_Median"],
    1,
    0
)

print("Classification Records:", len(classification_df))
print("0 = Low Price, 1 = High Price")
print(classification_df["Price_Category"].value_counts())

# Use categorical market information as predictors.
# Min_Price, Max_Price and Modal_Price are excluded from X
# to avoid directly giving the model the answer.
X = classification_df[categorical_columns]
y = classification_df["Price_Category"]

if y.nunique() != 2:
    raise ValueError("Classification requires both Low and High categories.")

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

classification_results = []


# ==========================================================
# 2. LOGISTIC REGRESSION
# Predict whether mandi price category is High or Low
# ==========================================================

print("\n========== 2. LOGISTIC REGRESSION ==========")

model = Pipeline([
    ("preprocessing", create_preprocessor(categorical_columns, [])),
    ("scaling", StandardScaler(with_mean=False)),
    ("classifier", LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ))
])

model.fit(X_train_c, y_train_c)

evaluate_classifier(
    model, "Logistic Regression", X_test_c, y_test_c
)


# ==========================================================
# 3. DECISION TREE
# Predict mandi price category using decision rules
# ==========================================================

print("\n========== 3. DECISION TREE ==========")

model = Pipeline([
    ("preprocessing", create_preprocessor(categorical_columns, [])),
    ("classifier", DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ))
])

model.fit(X_train_c, y_train_c)

evaluate_classifier(
    model, "Decision Tree", X_test_c, y_test_c
)

# Draw a readable decision tree.
tree_preprocessor = model.named_steps["preprocessing"]
tree_classifier = model.named_steps["classifier"]

try:
    feature_names = tree_preprocessor.get_feature_names_out()
except Exception:
    feature_names = None

plt.figure(figsize=(16, 8))
plot_tree(
    tree_classifier,
    feature_names=feature_names,
    class_names=["Low Price", "High Price"],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=7
)
plt.title("Decision Tree - Mandi Price Classification")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "04_Decision_Tree.png"))
plt.show()


# ==========================================================
# 4. RANDOM FOREST
# Predict mandi price category using multiple decision trees
# ==========================================================

print("\n========== 4. RANDOM FOREST ==========")

model = Pipeline([
    ("preprocessing", create_preprocessor(categorical_columns, [])),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

model.fit(X_train_c, y_train_c)

evaluate_classifier(
    model, "Random Forest", X_test_c, y_test_c
)


# ==========================================================
# 5. K-NEAREST NEIGHBORS (KNN)
# Predict price category using nearest mandi records
# ==========================================================

print("\n========== 5. K-NEAREST NEIGHBORS (KNN) ==========")

model = Pipeline([
    ("preprocessing", create_preprocessor(categorical_columns, [])),
    ("scaling", StandardScaler(with_mean=False)),
    ("classifier", KNeighborsClassifier(n_neighbors=5, n_jobs=-1))
])

model.fit(X_train_c, y_train_c)

evaluate_classifier(
    model, "KNN", X_test_c, y_test_c
)


# ==========================================================
# 6. SUPPORT VECTOR MACHINE (SVM)
# Classify mandi prices as High or Low
# ==========================================================

print("\n========== 6. SUPPORT VECTOR MACHINE (SVM) ==========")

# The original SVC(probability=True) can be very slow on this
# dataset because probability=True performs additional probability
# fitting. LinearSVC is a linear Support Vector Machine designed
# for large, high-dimensional datasets. CalibratedClassifierCV
# supplies probabilities so Log Loss and ROC-AUC remain available.

svm_classifier = CalibratedClassifierCV(
    estimator=LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=10000,
        random_state=RANDOM_STATE
    ),
    method="sigmoid",
    cv=3
)

model = Pipeline([
    ("preprocessing", create_preprocessor(categorical_columns, [])),
    ("scaling", StandardScaler(with_mean=False)),
    ("classifier", svm_classifier)
])

print("Training Linear SVM with probability calibration...")
model.fit(X_train_c, y_train_c)

evaluate_classifier(
    model, "Support Vector Machine", X_test_c, y_test_c
)


# ==========================================================
# 7. K-MEANS CLUSTERING
# Group mandi records based on market prices
# ==========================================================

print("\n========== 7. K-MEANS CLUSTERING ==========")

X = df[["Min_Price", "Max_Price", "Modal_Price"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["Cluster"] = model.fit_predict(X_scaled)

centroids = scaler.inverse_transform(model.cluster_centers_)

print("Mandi Price Clusters:")
print(
    df[
        ["Commodity", "Market", "Modal_Price", "Cluster"]
    ].head(15)
)

print("Cluster Centers:")
print(pd.DataFrame(
    centroids,
    columns=["Min_Price", "Max_Price", "Modal_Price"]
))

plt.figure(figsize=(8, 6))
plt.scatter(
    df["Min_Price"],
    df["Modal_Price"],
    c=df["Cluster"],
    cmap="viridis",
    s=35,
    alpha=0.7
)
plt.xlabel("Minimum Price")
plt.ylabel("Modal Price")
plt.title("K-Means Clustering of Mandi Prices")
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "05_KMeans_Clustering.png"))
plt.show()

df.to_csv(
    os.path.join(OUTPUT_FOLDER, "Mandi_Data_With_Clusters.csv"),
    index=False
)


# ==========================================================
# FINAL CLASSIFICATION MODEL COMPARISON
# ==========================================================

print("\n========== CLASSIFICATION MODEL COMPARISON ==========")

classification_comparison = pd.DataFrame(classification_results)

print(classification_comparison.round(4).to_string(index=False))

classification_comparison.to_csv(
    os.path.join(OUTPUT_FOLDER, "Classification_Comparison.csv"),
    index=False
)

plt.figure(figsize=(11, 6))
classification_comparison.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
].plot(kind="bar", ax=plt.gca())

plt.title("Classification Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_FOLDER, "06_Classification_Comparison.png")
)
plt.show()


# ==========================================================
# END OF ALL 7 MACHINE LEARNING MODELS
# ==========================================================

print("\n========================================")
print("ALL 7 MACHINE LEARNING MODELS EXECUTED")
print("========================================")

print("\nResults saved in:")
print(OUTPUT_FOLDER)

print("\nGenerated Files:")
for filename in sorted(os.listdir(OUTPUT_FOLDER)):
    print(filename)

print("\nPROJECT COMPLETED SUCCESSFULLY!")
