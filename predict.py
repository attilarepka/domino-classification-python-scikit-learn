import os
import pickle
import re

import cv2
import numpy as np
from skimage.io import imread
from skimage.transform import resize

model = pickle.load(open("model.pkl", "rb"))

INPUT_DIR = "./dataset"

cv2.namedWindow("predict")
cv2.setWindowTitle("predict", "Press ENTER to continue, ESC to exit")

file_count = 0
accurate_count = 0
for file in os.listdir(INPUT_DIR):
    file_count += 1
    img = imread(os.path.join(INPUT_DIR, file), 0)
    img = resize(img, (100, 50))

    pre_img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    red = np.zeros((30, pre_img.shape[1], 3), np.uint8)
    red[:] = (0, 0, 255)
    pre_img = cv2.vconcat((red, pre_img))

    # Resize the image to make it bigger
    pre_img = cv2.resize(pre_img, None, fx=3, fy=3)

    img = img.reshape(1, -1)
    actual = re.findall(r"\d+", file)
    actual = [int(num) for num in actual]

    actual = sum(actual)

    prediction = model.predict(img)[0]
    probability = model.predict_proba(img)[0][prediction]

    if prediction == actual:
        accurate_count += 1

    cv2.putText(
        pre_img,
        f"{prediction}, {probability:.2f}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
    )

    cv2.imshow("predict", pre_img)

    key = cv2.waitKey(0) & 0xFF
    if key == 27:
        exit()
    if key == 13:
        continue

cv2.destroyAllWindows()

print("Accuracy: {}%".format(accurate_count / file_count * 100))
