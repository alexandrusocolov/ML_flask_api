import pickle
import numpy as np
from flask import Flask, request
app = Flask(__name__)


# load ML model
with open('xgb_model_iris.pickle', 'rb') as f:
    clf = pickle.load(f)

def predict_w_model(sepal_length = 10,
                    sepal_width = 10,
                    petal_length = 5,
                    petal_width = 5):
    """Wrapper that creates the input X vector and predicts with the model
    """
    x = np.array([[float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)]])
    prediction = clf.predict(x)[0]

    return int(prediction)


@app.route('/')
def index():
    return {'result': 'Hello world!'}


@app.route('/add_one/<n>')
def add_one(n):
    n_upd = int(n) + 1
    return {'result': n_upd}


@app.route('/predict')
def predict():
     if request.method == 'GET':
        pred = predict_w_model(sepal_length=float(request.args.get('sepal_length', 0)),
                               sepal_width=float(request.args.get('sepal_width', 0)),
                               petal_length=float(request.args.get('petal_length', 0)),
                               petal_width=float(request.args.get('petal_width', 0)))

        return {'prediction': pred}