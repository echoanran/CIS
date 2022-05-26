from imutils import face_utils
import dlib
import cv2
import glob
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
import time
import os
import math
from PIL import Image

shapePredictorPath = '../../../Dataset/shape_predictor_68_face_landmarks.dat'
faceDetector = dlib.get_frontal_face_detector()
facialLandmarkPredictor = dlib.shape_predictor(shapePredictorPath)
faceDet = cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_default.xml")


def get_face_size(image):
    global faceDetector
    faces = faceDetector(image, 1)
    if len(faces) == 0:
        return -1, -1
    face = faces[0]
    pos_start = tuple([face.left(), face.top()])
    pos_end = tuple([face.right(), face.bottom()])
    height = (face.bottom() - face.top())
    width = (face.right() - face.left())
    return height, width


def get_facelandmark(image):
    global faceDetector, facialLandmarkPredictor

    face = faceDetector(image, 1)
    if len(face) == 0:
        return None

    shape = facialLandmarkPredictor(image, face[0])
    facialLandmarks = face_utils.shape_to_np(shape)

    xyList = []
    for (x, y) in facialLandmarks[0:]: 
        xyList.append(x)
        xyList.append(y)
        
    return xyList


def find_faces(image, normalize=False, resize=None, gray=None):
    global faceDetector
    faces = faceDetector(image, 1)
    if len(faces) == 0:
        return None

    cutted_faces = [image[face.top():face.bottom(), face.left():face.right()] for face in faces]
    faces_coordinates = [(face.left(), face.top(), face.right() - face.left(), face.bottom() - face.top()) for face in faces]

    if normalize:
        if resize is None or gray is None:
            print("Error: resize & gray must be given while normalize is True.")
        normalized_faces = [_normalize_face(face, resize, gray) for face in cutted_faces]
    else:
        normalized_faces = cutted_faces
    return zip(normalized_faces, faces_coordinates)


def _normalize_face(face, resize=350, gray=True):
    if gray:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (resize, resize))
    return face


def transfer(x, y, RotationMatrix):

    x_new = int(round(RotationMatrix[0, 0] * x + RotationMatrix[0, 1] * y + RotationMatrix[0, 2]))
    y_new = int(round(RotationMatrix[1, 0] * x + RotationMatrix[1, 1] * y + RotationMatrix[1, 2]))

    return x_new, y_new


def alignment(img, featureList):

    Xs = featureList[::2]
    Ys = featureList[1::2]

    eye_center =((Xs[36] + Xs[45]) * 1./2, (Ys[36] + Ys[45]) * 1./2)
    dx = Xs[45] - Xs[36]
    dy = Ys[45] - Ys[36]

    angle = math.atan2(dy, dx) * 180. / math.pi

    RotationMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)

    new_img = cv2.warpAffine(img, RotationMatrix, (img.shape[1], img.shape[0])) 

    RotationMatrix = np.array(RotationMatrix)
    alignfeatureList = []
    for i in range(len(Xs)):
        x, y = transfer(Xs[i], Ys[i], RotationMatrix)
        alignfeatureList.append(x)
        alignfeatureList.append(y)

    return new_img, alignfeatureList, RotationMatrix, eye_center
