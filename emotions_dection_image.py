import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Labels des émotions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_DIR = os.path.join(BASE_DIR, "exemple_images")
RESULT_DIR = os.path.join(BASE_DIR, "result_images")
os.makedirs(RESULT_DIR, exist_ok=True)

caffeModel = os.path.join(BASE_DIR, "face_detection_models", "res10_300x300_ssd_iter_140000.caffemodel")
prototxtPath = os.path.join(BASE_DIR, "face_detection_models", "deploy.prototxt.txt")
emotion_model_path = os.path.join(BASE_DIR, "final_emotion_model.keras")

# Chargement des modèles
print("[INFO] Loading models...")
net = cv2.dnn.readNetFromCaffe(prototxtPath, caffeModel)
emotion_model = load_model(emotion_model_path)

# Traitement des images
for filename in os.listdir(EXAMPLE_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(EXAMPLE_DIR, filename)
        frame = cv2.imread(img_path)

        if frame is None:
            continue

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.5:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue

            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            resized_face = cv2.resize(gray_face, (48, 48))
            normalized_face = resized_face / 255.0
            reshaped_face = np.reshape(normalized_face, (1, 48, 48, 1))

            emotion_prediction = emotion_model.predict(reshaped_face, verbose=0)
            emotion_label = emotion_labels[np.argmax(emotion_prediction)]
            emotion_score = np.max(emotion_prediction)

            label_text = f"{emotion_label} ({emotion_score * 100:.2f}%)"
            y = startY - 10 if startY - 10 > 10 else startY + 10

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Sauvegarde de l'image annotée
        output_path = os.path.join(RESULT_DIR, filename)
        cv2.imwrite(output_path, frame)
        print(f"[INFO] Saved result: {output_path}")
