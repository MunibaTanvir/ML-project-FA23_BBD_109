# main.py
# FastAPI backend that loads the two saved models and serves predictions.
# Run with:  uvicorn main:app --reload
# Then open: http://127.0.0.1:8000

import joblib
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Create the web app
app = FastAPI(title="ML Prediction API")

# Load the trained models once, when the server starts
reg_model = joblib.load("models/regression_model.joblib")
clf_model = joblib.load("models/classification_model.joblib")

# Iris classes in the same order the model uses (0, 1, 2)
SPECIES = ["setosa", "versicolor", "virginica"]


# Request bodies
# Pydantic checks that the incoming JSON has these fields with the
# right types. If not, FastAPI automatically returns an error.
class HouseInput(BaseModel):
    size: float      # square feet
    bedrooms: int
    age: float       # years

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Routes
@app.get("/")
def home():
    # Serve the HTML page, so frontend and backend run from the same address
    return FileResponse("index.html")


@app.post("/predict/regression")
def predict_regression(data: HouseInput):
    # The model expects a 2D list: a list of rows, here just one row
    features = [[data.size, data.bedrooms, data.age]]
    prediction = reg_model.predict(features)[0]
    # Convert numpy float to a normal Python float so it can become JSON
    return {"predicted_price": round(float(prediction), 2)}


@app.post("/predict/classification")
def predict_classification(data: IrisInput):
    features = [[data.sepal_length, data.sepal_width,
                 data.petal_length, data.petal_width]]
    class_index = int(clf_model.predict(features)[0])
    return {"predicted_species": SPECIES[class_index]}
