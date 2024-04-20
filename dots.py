import cv2
import numpy as np


def detect_dots(image, min_area=64, max_area=110):
    def on_change(value):
        pass

    cv2.namedWindow("dots")

    cv2.setWindowTitle("dots", "Press ENTER to save image, ESC to exit")
    cv2.createTrackbar("min_area", "dots", min_area, 255, on_change)
    cv2.createTrackbar("max_area", "dots", max_area, 255, on_change)
    cv2.createTrackbar("treshold", "dots", 255, 255, on_change)

    cv2.createTrackbar("blocksize", "dots", 3, 255, on_change)
    cv2.setTrackbarMin("blocksize", "dots", 3)
    cv2.setTrackbarMax("blocksize", "dots", 11)

    cv2.createTrackbar("C", "dots", 2, 255, on_change)
    cv2.setTrackbarMin("C", "dots", 1)
    cv2.setTrackbarMax("C", "dots", 10)

    while True:
        min_area = cv2.getTrackbarPos("min_area", "dots")
        max_area = cv2.getTrackbarPos("max_area", "dots")
        treshold = cv2.getTrackbarPos("treshold", "dots")
        blocksize = cv2.getTrackbarPos("blocksize", "dots")

        if blocksize % 2 == 0:
            blocksize += 1
            cv2.setTrackbarPos("blocksize", "dots", blocksize)

        c = cv2.getTrackbarPos("C", "dots")

        threshed = cv2.adaptiveThreshold(
            image,
            treshold,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blocksize,
            c,
        )

        contours, _ = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

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
                    cv2.drawContours(img_with_contours, [contour], -1, (0, 0, 255), 2)
                elif y > midpoint:
                    lower_dot_count += 1
                    cv2.drawContours(img_with_contours, [contour], -1, (0, 255, 255), 2)

        text = f"{upper_dot_count} / {lower_dot_count}"
        font_scale = 0.5
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2

        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

        x = int((img_with_contours.shape[1] - text_size[0]) / 2)

        cv2.putText(
            img_with_contours,
            text,
            (x, 20),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
        )

        cv2.imshow("dots", img_with_contours)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            exit()
        if key == 13:
            break

    cv2.destroyAllWindows()

    return upper_dot_count, lower_dot_count
