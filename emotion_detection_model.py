import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
from PIL import Image
import cv2

# Step 1: Load and Preprocess the Dataset
def load_images(folder):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        for image_file in os.listdir(label_folder):
            image_path = os.path.join(label_folder, image_file)
            image = Image.open(image_path)
            image = image.resize((48, 48))  # Resize image to (48, 48)
            image = np.array(image) / 255.0  # Normalize pixel values
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels)

dataset_folder = 'archive'
images, labels = load_images(os.path.join(dataset_folder, 'train'))

# Step 2: Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Step 3: Encode the Labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Reshape the input images to include the single channel
X_train = X_train.reshape(-1, 48, 48, 1)
X_test = X_test.reshape(-1, 48, 48, 1)

# Step 4: Build the Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(7, activation='softmax')  # 7 output classes for 7 emotions
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 5: Train the Model
history = model.fit(X_train, y_train_encoded, epochs=50, batch_size=32, validation_data=(X_test, y_test_encoded))

# Step 6: Evaluate the Model
test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded)
print("Test Accuracy:", test_accuracy)

# Step 7: Real-time Emotion Detection
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Step 7: Save the Model
# Save the model
model.save('emotion_detection_model.h5')

# Now the model is saved as 'emotion_detection_model.h5' file


# # Initialize webcam
# camera = cv2.VideoCapture(0)

# while True:
#     # Capture frame from webcam
#     ret, frame = camera.read()
#     if not ret:
#         break

#     # Convert frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces in the frame
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     # Predict emotions for each face
#     for (x, y, w, h) in faces:
#         face_roi = gray[y:y+h, x:x+w]
#         resized = cv2.resize(face_roi, (48, 48))
#         normalized = resized / 255.0
#         reshaped = np.reshape(normalized, (1, 48, 48, 1))
#         result = model.predict(reshaped)
#         emotion_label = emotion_labels[np.argmax(result)]

#         # Display emotion label on the frame
#         cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#     # Display the frame
#     cv2.imshow("Emotion Detection", frame)

#     # Check for 'q' key to exit loop
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the camera and close OpenCV windows
# camera.release()
# cv2.destroyAllWindows()
