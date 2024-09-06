# opencv

import cv2
import dlib
from imutils import face_utils
import numpy as np

# on image (person.jpg)
image = cv2.imread("person.jpg", cv2.IMREAD_COLOR)
height, width, channel = image.shape
matrix = cv2.getRotationMatrix2D((width / 2, height / 2), -90, 0.7)
dst = cv2.warpAffine(image, matrix, (width, height))

ratio = 600.0 / dst.shape[1]
dim = (600, int(dst.shape[0] * ratio))

resized = cv2.resize(dst, dim, interpolation=cv2.INTER_AREA)

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

while True:
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        cropped_img = resized[rect.top(): rect.bottom(), rect.left(): rect.right()]

    # sunglasses image
    sunglasses_image = cv2.imread("sunglasses.png", cv2.IMREAD_COLOR)
    sunglasses_image_png = cv2.imread("sunglasses.png", cv2.IMREAD_UNCHANGED)

    # resize sunglasses to cropped image
    height, width, channel = cropped_img.shape
    resized_sunglasses_image = cv2.resize(sunglasses_image, (int(width * 0.9), int(height * 0.2)),
                                          interpolation=cv2.INTER_AREA)
    resized_sunglasses_image_png = cv2.resize(sunglasses_image_png, (int(width * 0.9), int(height * 0.2)),
                                              interpolation=cv2.INTER_AREA)

    # put sunglasses on cropped image
    for i in range(0, int(height * 0.2)):
        for j in range(0, int(width * 0.9)):
            if not resized_sunglasses_image_png[i, j, 3:] == 0:  # if this, pass, and else:
                cropped_img[int(height * 0.2) + i, j, :] = resized_sunglasses_image[i, j, :]

    cv2.imshow("Output", resized)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

# 'ESC' to see live cam
cv2.destroyAllWindows()

# live cam
cap = cv2.VideoCapture(0)
while True:
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        cropped_img = image[rect.top(): rect.bottom(), rect.left(): rect.right()]

    sunglasses_image = cv2.imread("sunglasses.png", cv2.IMREAD_COLOR)
    sunglasses_image_png = cv2.imread("sunglasses.png", cv2.IMREAD_UNCHANGED)
    height, width, channel = cropped_img.shape

    resized_sunglasses_image = cv2.resize(sunglasses_image, (int(width * 0.95), int(height * 0.2)),
                                          interpolation=cv2.INTER_AREA)
    resized_sunglasses_image_png = cv2.resize(sunglasses_image_png, (int(width * 0.95), int(height * 0.2)),
                                              interpolation=cv2.INTER_AREA)

    for i in range(0, int(height * 0.2)):
        for j in range(0, int(width * 0.95)):
            if resized_sunglasses_image_png[i, j, 3:] == 0:
                pass
            else:
                cropped_img[int(height * 0.2) + i, j, :] = resized_sunglasses_image[i, j, :]

    cv2.imshow("Output", image)
    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()