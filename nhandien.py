import cv2
import numpy as np
import RPi.GPIO as GPIO
import time

# Khởi tạo GPIO cho servo
servoPIN = 18  # Đặt chân GPIO bạn sử dụng cho servo
GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPIN, GPIO.OUT)

# Tạo đối tượng PWM với tần số 50Hz
p = GPIO.PWM(servoPIN, 50)
p.start(2.5)  # servo ở góc 90 độ ban đầu

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)))

    for (x, y, w, h) in faces:
        confidence = recognizer.predict(gray[y : y + h, x : x + w])[1]
        if confidence < 50:
            p.ChangeDutyCycle(7.5)  # Di chuyển servo về vị trí mở khóa
            time.sleep(5)  # Mở khóa trong 5 giây
            p.ChangeDutyCycle(2.5)  # Di chuyển servo về vị trí ban đầu
            print(f"Mở khóa thành công. Độ nhận diện: {100-confidence}")
            time.sleep(7)

        else:
            print("Không thể mở khóa")

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
p.stop()
GPIO.cleanup()