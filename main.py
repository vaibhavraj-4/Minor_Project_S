import streamlit as st
import cv2
from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
import os

# Set up the emotion detection model
emotion_model = Sequential()
emotion_model.add(Input(shape=(48, 48, 1)))
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

# Load model weights
try:
    emotion_model.load_weights('emotion_model.weights.h5')
except Exception as e:
    st.error(f"Error loading model weights: {e}")
    st.stop()

# Emotion labels and corresponding emoji paths
emotion_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 
    4: "Neutral", 5: "Sad", 6: "Surprised"
}
emoji_dist = {
    0: "templates/emojis/angry.png", 1: "templates/emojis/disgusted.png",
    2: "templates/emojis/fearful.png", 3: "templates/emojis/happy.png",
    4: "templates/emojis/neutral.png", 5: "templates/emojis/sad.png",
    6: "templates/emojis/surprised.png"
}

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    st.error("Error loading Haar cascade for face detection")
    st.stop()

# Streamlit app title and layout
st.title("Emotion Detector")
st.write("Detect emotions in a captured or uploaded photo.")

# Function to detect emotion in an image
def detect_emotion(image):
    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        st.error(f"Error converting image to grayscale: {e}")
        return None

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return None

    for (x, y, w, h) in faces:
        roi_gray = gray_image[y:y + h, x:x + w]
        try:
            cropped_img = cv2.resize(roi_gray, (48, 48))
        except Exception as e:
            st.error(f"Error resizing image: {e}")
            return None
        cropped_img = np.expand_dims(cropped_img, -1)
        cropped_img = np.expand_dims(cropped_img, 0)
        prediction = emotion_model.predict(cropped_img)
        emotion_index = int(np.argmax(prediction))
        return emotion_index
    return None

# Section for capturing photo from webcam
st.subheader("Capture Photo from Webcam")
captured_image = st.camera_input("Take a picture", key="camera")

if captured_image:
    # Display the captured image
    st.image(captured_image, caption="Captured Image", use_column_width=True)

    # Convert to OpenCV format
    image = Image.open(captured_image).convert('RGB')
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Detect emotion
    emotion_index = detect_emotion(image_np)

    if emotion_index is not None:
        emotion_label = emotion_dict.get(emotion_index, "Unknown")
        st.write(f"Detected Emotion: {emotion_label}")

        # Display corresponding emoji
        emoji_path = emoji_dist.get(emotion_index, "")
        if os.path.exists(emoji_path):
            emoji_image = Image.open(emoji_path)
            st.image(emoji_image, caption=emotion_label, width=150)
        else:
            st.error(f"Error: Emoji image not found for {emotion_label}")
    else:
        st.warning("No face detected. Please try again.")

# Section for uploading an image
st.subheader("Or Upload an Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="upload_image")

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert('RGB')
    except Exception as e:
        st.error(f"Error opening uploaded image: {e}")
        st.stop()

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to OpenCV format
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Detect emotion
    emotion_index = detect_emotion(image_np)

    if emotion_index is not None:
        emotion_label = emotion_dict.get(emotion_index, "Unknown")
        st.write(f"Detected Emotion: {emotion_label}")

        # Display corresponding emoji
        emoji_path = emoji_dist.get(emotion_index, "")
        if os.path.exists(emoji_path):
            emoji_image = Image.open(emoji_path)
            st.image(emoji_image, caption=emotion_label, width=150)
        else:
            st.error(f"Error: Emoji image not found for {emotion_label}")
    else:
        st.warning("No face detected. Please try again.")




