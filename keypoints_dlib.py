import cv2
import dlib
from imutils import face_utils, resize
import numpy as np
import math

model_path = 'shape_predictor_68_face_landmarks.dat'
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

def get_head_mask(img):
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    faces = faceCascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))    # Find faces
    if len(faces) != 0:
        x, y, w, h = faces[0]
        (x, y, w, h) = (x - 40, y - 100, w + 80, h + 200)
        rect1 = (x, y, w, h)
        cv2.grabCut(img, mask, rect1, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)     #Crop BG around the head
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')  # Take the mask from BG
    return mask2

file_name="./img/12.jpg"

img1 = cv2.imread(file_name)     # Load image
img1 = resize(img1, height=500)     # We result in 500px in height
mask = get_head_mask(img1)      # We get the mask of the head (without BG)

# Find the contours, take the largest one and memorize its upper point as the top of the head
cnts = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
cv2.drawContours(img1, [cnts[0]], -1, (0, 0, 255), 2)

face_detector = dlib.get_frontal_face_detector()
facial_landmark_predictor = dlib.shape_predictor(model_path)

grayImage = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
faces = face_detector(grayImage, 1)

# for all faces detect facial keypoints
for (i, face) in enumerate(faces):

    facial_landmarks = facial_landmark_predictor(grayImage, face)
    facial_landmarks = face_utils.shape_to_np(facial_landmarks)

    for (i, (x, y)) in enumerate(facial_landmarks):
        cv2.circle(img1, (x, y), 1, (0, 0, 255), -1)
        cv2.putText(img1, str(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

# Ищем левое ухо:

left_x_1 = int((facial_landmarks[0][0] + facial_landmarks[1][0]) / 2)
left_y_1 = int((facial_landmarks[0][1] + facial_landmarks[1][1]) / 2)

range_face = facial_landmarks[15][0] - facial_landmarks[2][0]

min_index = None
min_dist = 10000000
ear_point_x, ear_point_y = facial_landmarks[1][0], facial_landmarks[1][1]
min_dif = None

for idx, ex in enumerate(cnts[0]):
    x, y = ex[0][0], ex[0][1]
    distance = math.sqrt((x - ear_point_x) ** 2 + (y - ear_point_y) ** 2)
    if ((distance < min_dist) and \
            ((x - facial_landmarks[1][0]) < 0) and\
            (np.abs(x - facial_landmarks[1][0]) > int(range_face*0.04)) and \
            (np.abs(x - facial_landmarks[1][0]) < int(range_face * 0.15)) and \
            (y < left_y_1 ) and \
            (y > facial_landmarks[36][1] )):
        min_dist = distance
        min_index = idx +4

if min_index != None:
    e_x, e_y = cnts[0][min_index][0][0], cnts[0][min_index][0][1]
    cv2.circle(img1, (e_x, e_y), 2, (0, 255, 255), -1)
    print('LEFT EAR POINT WAS DETECTED')
else:
    for idx, ex in enumerate(cnts[0]):
        x, y = ex[0][0], ex[0][1]
        distance = math.sqrt((x - ear_point_x) ** 2 + (y - ear_point_y) ** 2)

        if ((x - facial_landmarks[1][0]) < 0) and \
                (np.abs(x - facial_landmarks[1][0]) > int(range_face * 0.1)):
            print('MB THIS IS A GIRL WITH HAIR AND EAR POINT MAY BE HERE')
            min_index = -1
            break

    if min_index == -1:
        cv2.circle(img1, (left_x_1 - 15, left_y_1), 2, (255, 0, 255), -1)

    print('LEFT EAR WAS NOT FOUND ON PIC')
#################################################
right_x_1 = int((facial_landmarks[15][0] + facial_landmarks[16][0]) / 2)
right_y_1 = int((facial_landmarks[15][1] + facial_landmarks[16][1]) / 2)

min_index = None
min_dist = 10000000
ear_point_x, ear_point_y = facial_landmarks[1][0], facial_landmarks[1][1]
min_dif = None
for idx, ex in enumerate(cnts[0]):
    x, y = ex[0][0], ex[0][1]
    distance = math.sqrt((x - ear_point_x) ** 2 + (y - ear_point_y) ** 2)
    if ((distance < min_dist) and \
            ((facial_landmarks[16][0] -x) < 0) and\
            (np.abs(facial_landmarks[16][0] -x) > int(range_face*0.03)) and \
            (np.abs(facial_landmarks[16][0] -x) < int(range_face * 0.15)) and \
            (y < right_y_1 ) and
            (y > facial_landmarks[26][1] )):
        min_dist = distance
        min_index = idx -4

if min_index != None:

    e_x, e_y = cnts[0][min_index][0][0], cnts[0][min_index][0][1]
    cv2.circle(img1, (e_x, e_y), 2, (0, 255, 255), -1)
    print('RIGHT EAR POINT WAS DETECTED')
else:
    for idx, ex in enumerate(cnts[0]):
        x, y = ex[0][0], ex[0][1]
        distance = math.sqrt((x - ear_point_x) ** 2 + (y - ear_point_y) ** 2)

        if ((facial_landmarks[16][0] -x) < 0) and \
                (np.abs(facial_landmarks[16][0] -x) > int(range_face * 0.1)):
            print('MB THIS IS A GIRL WITH HAIR AND EAR POINT MAY BE HERE')
            min_index = -1
            break

    if min_index == -1:
        cv2.circle(img1, (right_x_1 + 15, right_y_1), 2, (255, 0, 255), -1)
    print('RIGHT EAR WAS NOT FOUND')

while True:
    cv2.imshow("image1", img1)
    if cv2.waitKey(5) == 27:
        break
