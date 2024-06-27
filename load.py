import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the model
loaded_model = load_model('emotion_detection_model.h5')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize webcam
camera = cv2.VideoCapture(0)

while True:
    # Capture frame from webcam
    ret, frame = camera.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Predict emotions for each face
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        resized = cv2.resize(face_roi, (48, 48))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 48, 48, 1))
        result = loaded_model.predict(reshaped)
        emotion_label = emotion_labels[np.argmax(result)]

        # Display emotion label on the frame
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Emotion Detection", frame)

    # Check for 'q' key to exit loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
camera.release()
cv2.destroyAllWindows()
