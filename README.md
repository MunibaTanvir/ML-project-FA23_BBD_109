# Full-Stack ML App

A small web app with two scikit-learn models served by a FastAPI backend and an HTML frontend.

- **Regression:** Linear regression that predicts a house price from size, bedrooms and age (trained on a generated dataset).
- **Classification:** Logistic regression that predicts the Iris flower species from sepal and petal measurements.

## Project structure

```
train.py           # trains both models, prints R² and accuracy, saves them with joblib
main.py            # FastAPI backend with the two POST endpoints
index.html         # frontend page with one form per model
requirements.txt   # Python dependencies
models/            # saved .joblib models (created by train.py)
```

## How to run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Train and save the models:
   ```
   python train.py
   ```
3. Start the backend:
   ```
   uvicorn main:app --reload
   ```
4. Open http://127.0.0.1:8000 in your browser and use the forms.

## API endpoints

| Method | Endpoint                  | Example body |
|--------|---------------------------|--------------|
| POST   | `/predict/regression`     | `{"size": 1500, "bedrooms": 3, "age": 10}` |
| POST   | `/predict/classification` | `{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}` |

Interactive API docs are available at http://127.0.0.1:8000/docs.
