# This time, instead of an image, a video is getting captured and
# faces inside it are getting detected
# for example video from webcam

# cv2 is a simple AI engine
import cv2

# Pretrained data: https://github.com/opencv/opencv/tree/master/data/haarcascades
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Video input to detect face in
webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# While loop that loops forever trough all frames
while True:
    # read the current frame
    successful_frame_read, frame = webcam.read()

    # Converts frame to grayscale
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect face and return coordinates
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame)

    # face_coordinates is twodimensional array which has the upper left coordinates of the rectangle and its width and
    # height
    for (x, y, w, h) in face_coordinates:
        # Draws rectangle around the detected face
        # (0, 255, 0) = This controls the color of the rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)

    # Show the webcame frame by frame
    cv2.imshow("TitleOfSomething", frame)
    # waits 1 millisecond
    key = cv2.waitKey(1)

    # Loop stops if Q key is pressed
    if key == 81 or key == 113:
        break

# Clears the webcam after quitting
cv2.destroyAllWindows()

# Just to check if the code worked
print("Code completed")
