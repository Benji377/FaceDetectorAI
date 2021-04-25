# Video: https://youtu.be/XIrOM9oP3pA?list=PLIdk2M44Rqh-5x22oSpJMoNTNQMbLaZws
# Timestamp: https://youtu.be/XIrOM9oP3pA?t=5711
import cv2

# Pretrained data: https://github.com/opencv/opencv/tree/master/data/haarcascades
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Image to detect face in
img = cv2.imread('imgs/robert_img.jpg')

# Converts the image to grayscale (necessary to simplify algorithm)
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect face and returns coordinates of the rectangle surrounding the face
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
print(face_coordinates)

# face_coordinates is twodimensional array which has the upper left coordinates of the rectangle and its width and
# height
for (x, y, w, h) in face_coordinates:
    # Draws rectangle around the detected face
    # (0, 255, 0) = This controls the color of the rectangle
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 10)


# Shows the image
cv2.imshow('TitleOfWindow', img)
# Waits until programm stops, else image not shown
cv2.waitKey()

print("Code completed")
