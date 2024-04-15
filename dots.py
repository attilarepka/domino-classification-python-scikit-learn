import cv2
import numpy as np


def detect_dots(image, min_area=64, max_area=110):
    threshed = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(
        threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    midpoint = image.shape[1]

    upper_dot_count = 0
    lower_dot_count = 0

    img_with_contours = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    red = np.zeros((30, img_with_contours.shape[1], 3), np.uint8)
    red[:] = (0, 0, 255)
    img_with_contours = cv2.vconcat((red, img_with_contours))

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            _, y, h, h = cv2.boundingRect(contour)
            contour[:, 0, 1] += 30
            if y + h < midpoint:
                upper_dot_count += 1
                cv2.drawContours(img_with_contours, [
                                 contour], -1, (0, 0, 255), 2)
            elif y > midpoint:
                lower_dot_count += 1
                cv2.drawContours(img_with_contours, [
                                 contour], -1, (0, 255, 255), 2)

    cv2.putText(img_with_contours, f"{upper_dot_count}, {lower_dot_count}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.imshow("Contours", img_with_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return upper_dot_count, lower_dot_count
