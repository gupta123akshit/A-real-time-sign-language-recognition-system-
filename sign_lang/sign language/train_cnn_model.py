import os
import cv2
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

data_dir = "dataset"
categories = sorted(os.listdir(data_dir))
img_size = 64

data = []
labels = []

for category in categories:
    path = os.path.join(data_dir, category)
    class_num = categories.index(category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            resized = cv2.resize(img_array, (img_size, img_size))
            data.append(resized)
            labels.append(class_num)
        except:
            pass

X = np.array(data).reshape(-1, img_size, img_size, 1) / 255.0
y = to_categorical(labels, num_classes=len(categories))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))
model.save("model/asl_model.h5")

print("Model trained and saved successfully!")
