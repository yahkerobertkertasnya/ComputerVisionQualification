import math

import cv2
import os

import numpy as np


def images_reader(path=None):
    image_path_list = []

    directory_list = os.listdir(path)

    for index, image_folder in enumerate(directory_list):
        for images in os.listdir(f'{path}/{image_folder}'):
            full_path = f'{path}/{image_folder}/{images}'
            image_path_list.append((index, image_folder, full_path))

    return image_path_list


def face_detector(image_data=None, cascade_path=None):
    face_list = []
    classifier = cv2.CascadeClassifier(cascade_path)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    for index, name, image in image_data:
        gray_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        modified_image = clahe.apply(gray_image)

        faces = classifier.detectMultiScale(modified_image, scaleFactor=1.2, minNeighbors=5)

        if len(faces) < 1:
            continue
        else:
            for face_rect in faces:
                x, y, w, h = face_rect
                face_image = modified_image[y:y + h, x:x + w]
                face_list.append((index, name, face_image, (x, y, w, h)))

    return face_list


def train_model(face_data=None):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    class_list = []
    name_list = []
    face_list = []

    for index, name, face, _ in face_data:
        class_list.append(index)
        name_list.append(name)
        face_list.append(face)

    face_recognizer.train(face_list, np.array(class_list))

    result = list(dict.fromkeys(name_list))

    face_model = (face_recognizer, result)
    return face_model


def face_recognize(image_data=None, cascade_path=None, face_model=None):
    classifier = cv2.CascadeClassifier(cascade_path)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    for index, name, image_path in image_data:
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        modified_image = clahe.apply(gray_image)

        faces = classifier.detectMultiScale(modified_image, scaleFactor=1.2, minNeighbors=5)
        face_recognizer, result = face_model
        if len(faces) < 1:
            continue
        else:
            for face_rect in faces:
                x, y, w, h = face_rect
                face_image = modified_image[y:y + h, x:x + w]

                index, confidence = face_recognizer.predict(face_image)

                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.putText(image, f'{result[index]} : {math.floor(confidence)}%', (x - 5, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))

                cv2.imshow('result', image)
                cv2.waitKey()
                cv2.destroyAllWindows()


def face_recognize_single(image_path=None, cascade_path=None, face_model=None):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    modified_image = clahe.apply(gray_image)

    classifier = cv2.CascadeClassifier(cascade_path)
    faces = classifier.detectMultiScale(modified_image, scaleFactor=1.2, minNeighbors=5)
    face_recognizer, result = face_model
    print(result)
    if len(faces) < 1:
        print("No Face Found")
    else:
        for face_rect in faces:
            x, y, w, h = face_rect
            face_image = modified_image[y:y + h, x:x + w]

            index, confidence = face_recognizer.predict(face_image)

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(image, f'{result[index]} : {math.floor(confidence)}%', (x - 5, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1,
                        (0, 255, 0))

            cv2.imshow('result', image)
            cv2.waitKey()
            cv2.destroyAllWindows()

