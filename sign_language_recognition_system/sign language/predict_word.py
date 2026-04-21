import cv2
import numpy as np
import os
from tensorflow import keras
from keras.models import load_model

model = load_model("model/asl_model.h5")
categories = sorted(os.listdir("dataset"))
img_size = 64

cap = cv2.VideoCapture(0)
print("Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi = frame[50:450, 50:450]
    cv2.rectangle(frame, (50, 50), (450, 450), (0, 255, 0), 2)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (img_size, img_size))
    normalized = resized.reshape(1, img_size, img_size, 1) / 255.0

    prediction = model.predict(normalized)
    label = categories[np.argmax(prediction)]
    cv2.putText(frame, f"{label.upper()}", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Word Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
