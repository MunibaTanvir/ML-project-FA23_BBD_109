# train.py
# Trains two models and saves them to the "models" folder:
#   1. Linear regression  -> predicts a house price (a number)
#   2. Logistic regression -> predicts an Iris flower species (a category)
# Run with:  python train.py

import os
import numpy as np
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

# Make sure the folder where we save models exists
os.makedirs("models", exist_ok=True)


# ---------------------------------------------------------------
# PART 1: REGRESSION (house price)
# ---------------------------------------------------------------
# We generate a small fake house dataset so no download is needed.
# A fixed seed (42) means we get the same "random" data every run.
rng = np.random.default_rng(42)
n = 500  # number of houses

size = rng.uniform(500, 3500, n)      # house size in square feet
bedrooms = rng.integers(1, 6, n)      # 1 to 5 bedrooms
age = rng.uniform(0, 50, n)           # age of the house in years

# The "true" price formula, plus some random noise so it looks realistic
price = 50000 + 150 * size + 10000 * bedrooms - 1000 * age + rng.normal(0, 20000, n)

# X = input features (one row per house), y = what we want to predict
X_reg = np.column_stack([size, bedrooms, age])
y_reg = price

# Keep 20% of the data aside to test the model on houses it has never seen
X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Create the model and fit (train) it on the training data
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# R² score: 1.0 is perfect, 0 means "no better than guessing the average"
r2 = r2_score(y_test, reg_model.predict(X_test))
print(f"Regression R² score: {r2:.3f}")

# Save the trained model to a file so the backend can load it later
joblib.dump(reg_model, "models/regression_model.joblib")


# ---------------------------------------------------------------
# PART 2: CLASSIFICATION (Iris species)
# ---------------------------------------------------------------
# The Iris dataset comes built into scikit-learn.
# Features: sepal length, sepal width, petal length, petal width (cm)
# Target: 0 = setosa, 1 = versicolor, 2 = virginica
iris = load_iris()
X_clf = iris.data
y_clf = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

# Logistic regression is a simple classification model.
# max_iter=200 gives it enough steps to finish training without a warning.
clf_model = LogisticRegression(max_iter=200)
clf_model.fit(X_train, y_train)

# Accuracy: fraction of test flowers it classified correctly
accuracy = accuracy_score(y_test, clf_model.predict(X_test))
print(f"Classification accuracy: {accuracy:.3f}")

joblib.dump(clf_model, "models/classification_model.joblib")

print("Both models saved in the 'models' folder.")
