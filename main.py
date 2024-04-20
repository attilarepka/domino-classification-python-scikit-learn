import argparse
import os
import pickle
import random
import re

import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.util import random_noise
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

parser = argparse.ArgumentParser()

parser.add_argument(
    "-o", "--output", help="Output model file", required=False, default="model.pkl"
)

parser.add_argument(
    "-d", "--dataset", help="Input dataset directory", required=False, default="dataset"
)

args = parser.parse_args()


data = []
labels = []
datalabels = {}

for file in os.listdir(args.dataset):
    img = imread(os.path.join(args.dataset, file))
    img = resize(img, (100, 50))

    data.append(img.flatten())
    label = re.findall(r"\d+", file)
    label = [int(num) for num in label]
    label = sum(label)
    labels.append(label)
    if datalabels.get(label) is None:
        datalabels[label] = []
    datalabels[label].append(img.flatten())

most_frequent_label = labels.count(max(labels, key=labels.count))

for key, value in datalabels.items():
    if len(value) < most_frequent_label:
        for _ in range(most_frequent_label - len(value)):
            random_value = random.choice(value)
            data.append(random_value)
            labels.append(key)

augmented_data = []
augmented_labels = []

for img, label in zip(data, labels):
    augmented_data.append(img)
    augmented_labels.append(label)

    noisy_img = random_noise(img.reshape(50, 100), var=0.01**2)
    augmented_data.append(noisy_img.flatten())
    augmented_labels.append(label)

augmented_data = np.asarray(augmented_data)
augmented_labels = np.asarray(augmented_labels)

classifier = SVC(probability=True)

parameters = [{"gamma": [0.01, 0.001, 0.0001], "C": [1, 10, 100, 1000]}]

grid_search = GridSearchCV(classifier, parameters, cv=2)
grid_search.fit(augmented_data, augmented_labels)

best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(augmented_data)

score = accuracy_score(y_prediction, augmented_labels)

print("{}% of samples were correctly classified".format(str(score * 100)))

pickle.dump(best_estimator, open(args.output, "wb"))
