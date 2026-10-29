# Full-Stack ML App

A small web app with two scikit-learn models served by a FastAPI backend and an HTML frontend.

- **Regression:** Linear regression trained on the [California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) (1990 US census, 20,640 districts). It predicts a district's median house value from income, house age, rooms, population and location.
- **Classification:** Logistic regression trained on the [Iris dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset). It predicts the flower species from sepal and petal measurements.

Both datasets are loaded through scikit-learn. California Housing downloads automatically the first time you run `train.py` (internet needed once, then it's cached). Iris is bundled with scikit-learn.

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
| POST   | `/predict/regression`     | `{"med_inc": 8.33, "house_age": 41, "ave_rooms": 6.98, "ave_bedrms": 1.02, "population": 322, "ave_occup": 2.56, "latitude": 37.88, "longitude": -122.23}` |
| POST   | `/predict/classification` | `{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}` |

Interactive API docs are available at http://127.0.0.1:8000/docs.
