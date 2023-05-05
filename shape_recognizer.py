import cv2
import os


def shape_recognition():
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
    
    print("Loading... Please wait")
    igray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(igray, 230, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

    first = True
    for cont in contours:
        if first:
            first = False
            continue

        contour = cv2.approxPolyDP(cont, 0.01 * cv2.arcLength(cont, True), True)

        cv2.drawContours(image, [cont], 0, (0, 0, 0), 2)

        moments = cv2.moments(cont)
        if moments['m00'] != 0.0:
            x = int(moments['m10'] / moments['m00']) - 30
            y = int(moments['m01'] / moments['m00']) + 20

        text = ""
        if len(contour) == 3:
            text = "Triangle"
        elif len(contour) == 4:
            text = "Quadrilateral"
        elif len(contour) == 5:
            text = "Pentagon"
        elif len(contour) == 6:
            text = "Hexagon"
        elif len(contour) == 7:
            text = "Heptagon"
        elif len(contour) == 8:
            text = "Octagon"
        elif len(contour) == 9:
            text = "Nonagon"
        elif len(contour) == 10:
            text = "Decagon"
        else:
            text = "Circle"

        if(text != ""):
            cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

    return image