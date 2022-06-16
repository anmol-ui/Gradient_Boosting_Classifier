from mnist.loader import MNIST
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Loading training and test data from the mnist dataset
mndata = MNIST("/content")
train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

train_images = np.array(train_images)  # converting train images data into numpy array
train_labels = np.array(train_labels)  # converting train labels to numpy array
test_images = np.array(test_images)  # converting test images data into numpy array
test_labels = np.array(test_labels)  # converting test labels to numpy array

# Parameters
lr = 0.1  # learning rate
M = 5  # no of iterations

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# training
fx = train_labels.mean()  # initial prediction
f_init = fx
ensemble = []  # list to store all decision tree regressors

for i in range(M):
    residue = train_labels - fx
    tree = DecisionTreeRegressor(max_depth=1)
    tree.fit(train_images, residue)
    prediction = tree.predict(train_images)  # tree prediction
    fx = fx + (lr * prediction)

    ensemble.append(tree)

# testing
prediction = f_init
for tree in ensemble:
    prediction = prediction + (lr * tree.predict(test_images))
score = r2_score(test_labels, prediction)
print("Final testing accuracy: ", score)
