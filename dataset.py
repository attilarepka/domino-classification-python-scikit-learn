import os
import cv2
from dots import detect_dots

img = cv2.imread("dominos.jpg", 0)

_, thresholded = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

thresholded_inv = cv2.bitwise_not(thresholded)

contours, _ = cv2.findContours(
    thresholded_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    area = cv2.contourArea(contour)
    if area > 100:
        x, y, w, h = cv2.boundingRect(contour)
        domino_image = img[y:y+h, x:x+w]

        avg_color = cv2.mean(domino_image)
        base_color = "white" if avg_color[0] > 100 else "black"
        min_area = 63 if base_color == "white" else 67

        upper, lower = detect_dots(domino_image, min_area=min_area)

        dir_name = "dataset"
        file_name = os.path.join(
            dir_name, f"domino_{base_color}_{upper}_{lower}.jpg")

        print(cv2.boundingRect(contour), domino_image.shape, file_name)

        cv2.imwrite(file_name, domino_image)
