import threading
import numpy as np
import math
import os
import cv2


class Camera:
    def __init__(self):
        self.capture = cv2.VideoCapture(0)
        self.thread = threading.Thread(target=self.update_camera)
        self.thread.daemon = True
        self.thread.start()

        self.ret = None
        self.data = None
        self.show_data = False

    def update_camera(self):
        while self.capture:
            ret, data = self.capture.read()

            self.ret = ret
            self.data = data

    def get_data(self):
        return self.ret, self.data

    def destroy(self):
        self.capture.release()
        self.capture = None
        cv2.destroyAllWindows()


def video_capture_normal():
    toggleDetector = False
    toggleBlur = False
    camera = Camera()

    while True:
        ret, frame = camera.get_data()
        if ret:

            if toggleDetector:
                face_list = face_detector(frame, './model/haarcascade_frontalface_default.xml')
                if face_list:
                    for _, (x, y, w, h) in face_list:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if toggleBlur:
                face_list = face_detector(frame, './model/haarcascade_frontalface_default.xml')
                if face_list:
                    for image, (x, y, w, h) in face_list:
                        frame[y:y + h, x:x + w] = cv2.GaussianBlur(frame[y:y + h, x:x + w], (101, 101), 0)

            key = cv2.waitKey(1) & 0xFF
            if key == 13:
                camera.destroy()
                break

            if key == ord('f') or key == ord('F'):
                toggleDetector = not toggleDetector

            if key == ord('b') or key == ord('B'):
                toggleBlur = not toggleBlur

            cv2.imshow('Camera', frame)

    print("Loading... Please wait")
    camera.thread.join()
    os.system("cls")


def video_capture_data(name=None):
    faces = []
    name = [name]
    camera = Camera()

    while True:
        image = None
        ret, frame = camera.get_data()

        if ret:
            face_list = face_detector(frame, './model/haarcascade_frontalface_default.xml')
            if face_list:
                image, (x, y, w, h) = face_list[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            key = cv2.waitKey(1) & 0xFF
            if key == 13:
                camera.destroy()
                break

            if key == ord('c') or key == ord('C'):
                faces.append(image)

            cv2.imshow('Camera', frame)

    print("Loading... Please wait")
    camera.thread.join()
    os.system("cls")
    if faces:
        data = [(image, name) for image, name in zip(faces, name * len(faces)) if image is not None]
        return data
    return None


def video_capture_recognizer(image_data=None):
    camera = Camera()

    face_model = train_model(image_data)
    while True:
        ret, frame = camera.get_data()
        if ret:

            frame = face_recognize_single(frame, './model/haarcascade_frontalface_default.xml', face_model)

            key = cv2.waitKey(1) & 0xFF
            if key == 13:
                camera.destroy()
                break

            cv2.imshow('Camera', frame)

    print("Loading... Please wait")
    camera.thread.join()
    os.system("cls")


def train_model(face_data=None):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    class_list = []
    name_list = []
    face_list = []

    for face, name in face_data:
        num = len(set(name_list))
        class_list.append(num - 1 if num > 0 else 0)
        name_list.append(name)
        face_list.append(face)

    face_recognizer.train(face_list, np.array(class_list))

    result = list(dict.fromkeys(name_list))

    face_model = (face_recognizer, result)
    return face_model


def face_detector(image=None, cascade_path=None):
    face_list = []
    classifier = cv2.CascadeClassifier(cascade_path)
    clahe = cv2.createCLAHE(2.0, (8, 8))

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    modified_image = clahe.apply(gray_image)

    faces = classifier.detectMultiScale(modified_image, scaleFactor=1.2, minNeighbors=5)

    if len(faces) < 1:
        return

    for face_rect in faces:
        x, y, w, h = face_rect
        face_image = modified_image[y:y + h, x:x + w]
        face_list.append((face_image, (x, y, w, h)))

    return face_list


def face_recognize_single(image=None, cascade_path=None, face_model=None):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    modified_image = clahe.apply(gray_image)

    classifier = cv2.CascadeClassifier(cascade_path)
    faces = classifier.detectMultiScale(modified_image, scaleFactor=1.2, minNeighbors=5)
    face_recognizer, result = face_model
    if len(faces) > 0:
        for face_rect in faces:
            x, y, w, h = face_rect
            face_image = modified_image[y:y + h, x:x + w]

            index, confidence = face_recognizer.predict(face_image)

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(image, f'{result[index]} : {math.floor(confidence)}%', (x - 5, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1,
                        (0, 255, 0))

    return image
