import cv2 as cv2
import os
import matplotlib.pyplot as plt

from image_processing import gray_image, blur_image, thresh_image
from edge_detection import laplacian, sobel, canny
from shape_recognizer import shape_recognition
from face_recognition import images_reader, face_detector, train_model, face_recognize, face_recognize_single
from capture_detection import video_capture_normal, video_capture_data, video_capture_recognizer

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def menu():
    while True:
        print("Welcome to Computer Vision Application")
        print("1. Image Processing")
        print("2. Edge Detection")
        print("3. Shape Recognition")
        print("4. Face Recognition")
        print("5. Video Capture")
        print("0. Go Back")
        user_input = input("Select your choice")
        if user_input == '0':
            print("Bye")
            return
        elif user_input == '1':
            image_processing()
        elif user_input == '2':
            edge_detection()
        elif user_input == '3':
            show_image('Shape Recognizer', shape_recognition())
            os.system("cls")
        elif user_input == '4':
            face_recognition()
        elif user_input == '5':
            video_capture()


def show_result(row=None, col=None, content=None):
    plt.figure(figsize=(col + 2, row + 2))
    for i, (label, image) in enumerate(content):
        plt.subplot(row, col, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(label)
        plt.axis('off')
    plt.show()


def show_image(winname=None, image=None):
    cv2.imshow(winname, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def image_processing():
    directory = input("Input your image directory: ")
    while True:
        if os.path.isfile(directory):
            image = cv2.imread(directory)
            if image is not None:
                break
        print("Image not found!")
        directory = input("Input your image directory: ")

    image = cv2.imread(directory)
    print("Image found!")
    input("Press enter to continue...")
    os.system("cls")
    while True:
        print("What do you want to do with the image?")
        print("1. Grayscale")
        print("2. Blur")
        print("3. Thresh")
        print("0. Go Back")
        user_input = input("Select your choice: ")
        while True:
            if user_input in ('1', '2', '3', '4', '0'):
                break
            user_input = input("Select your choice")
        if user_input == '0':
            os.system("cls")
            break
        elif user_input == '1':
            print("Loading... Please wait")
            row, col, content = gray_image(image)
            show_result(row, col, content)
            os.system("cls")
        elif user_input == '2':
            print("Loading... Please wait")
            row, col, content = blur_image(image)
            show_result(row, col, content)
            os.system("cls")
        elif user_input == '3':
            print("Loading... Please wait")
            row, col, content = thresh_image(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            show_result(row, col, content)
            os.system("cls")


def edge_detection():
    os.system("cls")
    directory = input("Input your image directory: ")
    while True:
        if os.path.isfile(directory):
            image = cv2.imread(directory)
            if image is not None:
                break
        print("Image not found!")
        directory = input("Input your image directory: ")

    image = cv2.imread(directory)
    print("Image found!")
    input("Press enter to continue...")
    os.system("cls")
    while True:
        print("What method do you want to choose for the edge detection?")
        print("1. Laplacian")
        print("2. Sobel")
        print("3. Canny")
        print("0. Go Back")
        user_input = input("Select your choice: ")

        if user_input == '0':
            os.system("cls")
            break
        elif user_input == '1':
            print("Loading... Please wait")
            row, col, content = laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            show_result(row, col, content)
            os.system("cls")
        elif user_input == '2':
            print("Loading... Please wait")
            row, col, content = sobel(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            show_result(row, col, content)
            os.system("cls")
        elif user_input == '3':
            print("Loading... Please wait")
            row, col, content = canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            show_result(row, col, content)
            os.system("cls")


def face_recognition():
    os.system("cls")
    directory = input("Input your training data directory: ")
    while True:
        if os.path.isdir(directory):
            print("Directory found!")
            input("Press enter to continue...")
            os.system("cls")
            break
        print("Directory not found!")
        directory = input("Input your training data directory: ")

    print("Loading... Please wait")
    images = images_reader(directory)
    face_list = face_detector(images, './model/haarcascade_frontalface_default.xml')
    model = train_model(face_list)
    print("Training completed!")

    user_input = input("Show training data recognition result? (Y/N)")
    while True:
        if user_input == 'Y':
            print("Loading... Please wait")
            face_recognize(images, './model/haarcascade_frontalface_default.xml', model)
            os.system("cls")
            break
        elif user_input == 'N':
            break
        user_input = input("Show training data recognition result? (Y/N)")

    while True:
        print("What do you want to do?")
        print("1. Do Face Recognition on an image")
        print("0. Exit")
        user_input = input("Select your choice: ")
        if user_input == '1':
            directory = input("Input your image directory: ")
            while True:
                if os.path.isfile(directory):
                    image = cv2.imread(directory)
                    if image is not None:
                        break
                print("Image not found!")
                directory = input("Input your image directory: ")

            face_recognize_single(directory, './model/haarcascade_frontalface_default.xml', model)

        elif user_input == '0':
            return
        os.system("cls")


def video_capture():
    os.system("cls")
    model_data = []
    while True:
        print("Welcome to Computer Vision Application")
        print("1. Video Capture (face detection, filters)")
        print("2. Video Capture Data (gather data for face recognition)")
        print("3. Video Capture Recognizer (face recognition)")
        print("0. Go Back")

        user_input = input("Select your choice")
        if user_input == '0':
            return
        elif user_input == '1':
            os.system("cls")
            print("Press F to toggle face detector")
            print("Press B to toggle face blur")
            print("Press Enter to close")
            video_capture_normal()
        elif user_input == '2':
            os.system("cls")
            name = input("Input your name")
            print("Press C to capture image")
            print("Press Enter to close")
            data = video_capture_data(name)
            if data:
                model_data.extend(data)
        elif user_input == '3':
            os.system("cls")
            if len(model_data) > 0:
                video_capture_recognizer(model_data)
            else:
                print("No data!")




if __name__ == '__main__':
    menu()

