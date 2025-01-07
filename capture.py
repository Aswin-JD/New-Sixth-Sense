import cv2

class Capture():

    def camera(self):
        cam = cv2.VideoCapture(0)

        count = 0
        filename = "images/photo.png"
        while count < 5:
            _,frame = cam.read()
            cv2.imwrite(filename, frame)
            count = count + 1
        return filename