from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
caffeModel = os.path.join(BASE_DIR, "face_detection_models", "res10_300x300_ssd_iter_140000.caffemodel")
prototxtPath = os.path.join(BASE_DIR, "face_detection_models", "deploy.prototxt.txt")
# Le modéle final_emotion_model.keras n'est pas inclus dans le repos à cause de sa taille.
# Vous devez le télécharger à partir du notebook kaggle ou du notebook local exécuté.
# Notebook kaggle: https://www.kaggle.com/code/kassadiallo/cnn-emotion-classifier
emotion_model_path = os.path.join(BASE_DIR, "final_emotion_model.keras")


print("[INFO] Loading face detection model...")
net = cv2.dnn.readNetFromCaffe(prototxtPath, caffeModel)

print("[INFO] Loading emotion recognition model...")
emotion_model = load_model(emotion_model_path)

print("[INFO] Starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

try:
    while True:
        frame = vs.read()
        if frame is None:
            continue

        frame = imutils.resize(frame, width=700)
        (h, w) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence < 0.5:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]

            if face.size == 0:
                continue  

            # Prétraitement pour le modèle d’émotions
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            resized_face = cv2.resize(gray_face, (48, 48))
            normalized_face = resized_face / 255.0
            reshaped_face = np.reshape(normalized_face, (1, 48, 48, 1))

            emotion_prediction = emotion_model.predict(reshaped_face, verbose=0)
            emotion_label = emotion_labels[np.argmax(emotion_prediction)]
            label_text = f"{emotion_label} ({confidence * 100:.2f}%)"
            y = startY - 10 if startY - 10 > 10 else startY + 10

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Emotion Detection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user.")

finally:
    cv2.destroyAllWindows()
    vs.stop()
