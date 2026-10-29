# train.py
# Trains two models on real datasets and saves them to the "models" folder:
#   1. Linear regression   -> predicts a house price (a number)
#   2. Logistic regression -> predicts an Iris flower species (a category)
# Run with:  python train.py

import os
import joblib
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

# Make sure the folder where we save models exists
os.makedirs("models", exist_ok=True)


# ---------------------------------------------------------------
# PART 1: REGRESSION (California house prices)
# ---------------------------------------------------------------
# Real data from the 1990 US census: 20,640 districts in California.
# The first run downloads it (about 400 KB) and caches it on your computer,
# so later runs work offline.
#
# Each row is one district, with 8 features:
#   MedInc     - median income of households (in tens of thousands of $)
#   HouseAge   - median age of the houses (years)
#   AveRooms   - average number of rooms per house
#   AveBedrms  - average number of bedrooms per house
#   Population - number of people living in the district
#   AveOccup   - average number of people per house
#   Latitude   - location (north/south)
#   Longitude  - location (east/west)
# Target: median house value, in units of $100,000 (so 2.5 means $250,000)
housing = fetch_california_housing()
X_reg = housing.data      # inputs: one row per district
y_reg = housing.target    # what we want to predict: house value

# Keep 20% of the data aside to test the model on districts it has never seen
X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Create the model and fit (train) it on the training data
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# R² score: 1.0 is perfect, 0 means "no better than guessing the average".
# Real data is messy, so expect around 0.6 here, which is normal for a
# simple straight-line model on this dataset.
r2 = r2_score(y_test, reg_model.predict(X_test))
print(f"Regression R² score: {r2:.3f}")

# Save the trained model to a file so the backend can load it later
joblib.dump(reg_model, "models/regression_model.joblib")


# ---------------------------------------------------------------
# PART 2: CLASSIFICATION (Iris species)
# ---------------------------------------------------------------
# The Iris dataset is real data (150 flowers measured by botanist
# Edgar Anderson) and comes bundled inside scikit-learn, no download needed.
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
