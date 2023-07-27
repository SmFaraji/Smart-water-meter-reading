#projects channel     -> t.me/EngineeringLab
#aparat channel       -> www.aparat.com/EngineeringLab
#youtube channel      -> https://www.youtube.com/@sm_faraji
#GitHub               -> https://github.com/SmFaraji

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# load model
model = load_model("models/DigitDetector_130epochs.h5")

# ##########################choose this three parameter###############################
readData = 1  # for photo ->0 /// for video ->1
# choose paths
imgPath = 'photo/15.jpg'
videoPath = 'video/10.mp4'
# ################################

cap = cv2.VideoCapture(videoPath)
cap.set(3, 450)
cap.set(4, 450)


def getImg(value):
    global realImg
    if value == 0:
        img = cv2.imread(imgPath)
    else:
        ret, img = cap.read()
        realImg = cv2.resize(img.copy(), (450, 450))
    return img
####################################################################################


#############################
p = [0, 0, 0, 0, 0]
pval = [0, 0, 0, 0, 0]
#############################

# ======================================For Mapping Polygon to Rectangle=============================================
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped
# ==========================================================================================================


def divideImg(value):
    g = 2  # set number boxes
    if value == 1:  # first number
        icon = Rectimg[0 + 4 * g:32 - 2 * g, 0 + 3 * g:32 + g]
    if value == 2:  # second number
        icon = Rectimg[0 + 2 * g:32 - 2 * g, 32 + 3 * g:64 - g]
    if value == 3:  # third number
        icon = Rectimg[0 + 2 * g:32 - 2 * g, 64 + 3 * g:96 - g]
    if value == 4:  # forth number
        icon = Rectimg[0 + 2 * g:32 - 2 * g, 96 + 3 * g:128 - g]
    if value == 5:  # fifth number
        icon = Rectimg[0 + 2 * g:32 - 2 * g, 128 + 2 * g:160 - 3 * g]

    return icon


def mainShape(biggest, imgBigContour):
    global px, py, Rectimg

    px = np.zeros((4,), dtype=int)
    py = np.zeros((4,), dtype=int)
    for j in range(0, len(biggest)):
        px[j] = biggest[j][0][0]
        py[j] = biggest[j][0][1]

    pts = str([(px[0], py[0]), (px[1], py[1]), (px[2], py[2]), (px[3], py[3])])
    pts = np.array(eval(pts), dtype="float32")
    Rectimg = four_point_transform(realImg, pts)
    Rectimg = cv2.resize(Rectimg, (160, 32))
    cv2.imshow('RectImg', Rectimg)
    cv2.polylines(imgBigContour, [biggest], True, (0, 255, 255), 2)


def preProcess(img):
    global  imgThreshold, imgContours, imgBigContour
    img = cv2.resize(img, (450, 450))
    imgContours = img.copy()
    imgBigContour = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)
    #cv2.imshow('imgThreshold', imgThreshold)

    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)
    biggest, maxArea = biggestContour(contours)
    mainShape(biggest, imgBigContour)

    return biggest


def biggestContour(contours):
    global biggest
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


def resdyForPred(img):
    imgModel = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgModel = cv2.equalizeHist(imgModel)
 #   cv2.imshow('imgModel', imgModel)
  #  imgModel = cv2.adaptiveThreshold(imgModel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    imgModel = imgModel / 255
    return imgModel


def detect(biggest):
    for i in range(1, 6):
        if biggest is not None:
            img = cv2.resize(divideImg(i), (32, 32))
            img1 = resdyForPred(img)
            img = img1.reshape(1, 32, 32, 1)
            cv2.imshow('resdyForPred', img1)
            #cv2.waitKey(500)
            predictions = model.predict(img)

            probVal = np.amax(predictions)

            max_index = predictions.argmax()
            if probVal > 0.5:
                p[i - 1] = max_index
                pval[i - 1] = probVal

    return p


def drawResults():
    cv2.putText(imgBigContour, str(p[0]) + " " + str(p[1]) + " " + str(p[2]) + " " + str(p[3]) + " " + str(p[4]),
                (np.amin(px), np.amin(py) - 10), cv2.FONT_ITALIC, 1, (100, 100, 255), 2)
    cv2.putText(imgBigContour, str(int(np.sum(pval) * 10000 / 5) / 100) + '%', (np.amin(px), np.amax(py) + 20),
                cv2.FONT_ITALIC, 1, (0, 0, 0), 2)
    cv2.imshow('imageBigCont', imgBigContour)


while True:
    # get images
    img = getImg(readData)
    # preProcessing
    biggest = preProcess(img)
    # detect digits
    p = detect(biggest)
    print(p)
    # see results
    drawResults()
   # cv2.imshow('number', resdyForPred(cv2.resize(divideImg(4), (50, 50))))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


