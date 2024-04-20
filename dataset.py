import os
import cv2
import argparse
from dots import detect_dots
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument(
    "-i", "--input", help="Input image path for dataset creation", required=True
)

args = parser.parse_args()

input_image = cv2.imread(args.input, 0)

cv2.namedWindow("dataset")
cv2.setWindowTitle("dataset", "Press ENTER to save image, ESC to exit")


def on_change(value):
    pass


cv2.createTrackbar("threshold", "dataset", 0, 255, on_change)

threshold = 0
while True:
    threshold = cv2.getTrackbarPos("threshold", "dataset")

    _, thresholded = cv2.threshold(input_image, threshold, 255, cv2.THRESH_BINARY)

    thresholded_inv = cv2.bitwise_not(thresholded)

    contours, _ = cv2.findContours(
        thresholded_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    img = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
    red = np.zeros((30, img.shape[1], 3), np.uint8)
    red[:] = (0, 0, 255)
    img = cv2.vconcat((red, img))

    text = "Press ENTER to save image, ESC to exit"
    font_scale = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2

    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

    x = int((img.shape[1] - text_size[0]) / 2)

    cv2.putText(
        img,
        text,
        (x, 20),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
    )

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        exit()
    if key == 13:
        break

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            x, y, w, h = cv2.boundingRect(contour)
            y += 30
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 6)
            domino_image = img[y : y + h, x : x + w]

    cv2.imshow("dataset", img)

cv2.destroyAllWindows()

OUT_DIR_NAME = "dataset"
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 100:
        x, y, w, h = cv2.boundingRect(contour)
        domino_image = input_image[y : y + h, x : x + w]
        avg_color = cv2.mean(domino_image)
        base_color = "white" if avg_color[0] > 100 else "black"
        min_area = 63 if base_color == "white" else 67

        upper, lower = detect_dots(domino_image, min_area=min_area)
        file_name = os.path.join(
            OUT_DIR_NAME, f"domino_{base_color}_{upper}_{lower}.jpg"
        )

        if not os.path.exists(OUT_DIR_NAME):
            os.mkdir(OUT_DIR_NAME)

        cv2.imwrite(file_name, domino_image)
