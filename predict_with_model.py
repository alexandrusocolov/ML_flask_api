import pickle
import numpy as np

# load ML model
with open('xgb_model_iris.pickle', 'rb') as f:
    clf = pickle.load(f)


sepal_length = 10
sepal_width = 10
petal_length = 5
petal_width = 5

x = np.array([[float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)]])
prediction = clf.predict(x)[0]

print(prediction)