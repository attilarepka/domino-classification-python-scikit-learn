import os
import re
import pickle
import random

from skimage.io import imread
from skimage.transform import resize
from skimage.util import random_noise
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

input_dir = './dataset'

data = []
labels = []
classes = {}

for file in os.listdir(input_dir):
    img = imread(os.path.join(input_dir, file))
    img = resize(img, (100, 50))

    data.append(img.flatten())
    numbers = re.findall(r'\d+', file)
    numbers = [int(num) for num in numbers]
    numbers = sum(numbers)
    labels.append(numbers)
    if classes.get(numbers) is None:
        classes[numbers] = []
    classes[numbers].append(img.flatten())

for key, value in classes.items():
    if len(value) < 4:
        for _ in range(4 - len(value)):
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

parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

grid_search = GridSearchCV(classifier, parameters, cv=2)
grid_search.fit(augmented_data, augmented_labels)

best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(augmented_data)

score = accuracy_score(y_prediction, augmented_labels)

print('{}% of samples were correctly classified'.format(str(score * 100)))

pickle.dump(best_estimator, open('./model.pkl', 'wb'))
