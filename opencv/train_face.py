import cv2
import numpy as np
import os
from PIL import Image
import pickle

base_dir = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(base_dir, 'images')
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

y_labels = []
x_train = []
current_id = 0
label_ids = {}
for root, dirs, files in os.walk(img_dir):
    for file in files:
        if file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root)
            # print(label, path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            print(label_ids, file)

            pil_image = Image.open(path).convert('L')
            image_array = np.array(pil_image, "uint8")

            # print(image_array)
            faces = face_cascade.detectMultiScale(image_array, 1.3, 5)
            x_train.append(image_array)
            y_labels.append(id_)

with open("labels.pikle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save('trainner.yml')
