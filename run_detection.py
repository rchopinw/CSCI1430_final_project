import os
import cv2
import argparse
import re
import tensorflow as tf
from ARGS import ARGS
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Welcome to Our Face Mask Detection System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--model',
        required=True,
        choices=['Vanilla', "ResNet50", "Xception"],
        help="Please select model from [Vanilla, ResNet50, Xception]"
    )
    parser.add_argument(
        '--mode',
        required=True,
        choices=["Single", "Multiple"],
        help="Please select mode from [Single, Multiple]"
    )
    return parser.parse_args()


def get_optimal_model(model_dir):
    model_files = os.listdir(model_dir)
    if len(model_files) == 1:
        optimal_model_dir = model_files[0]
    else:
        model_accs = [
            float(
                re.findall(
                    r"[+-]?\d+\.\d+", file.split("acc")[-1]
                )[0]
            )
            for file in model_files
        ]
        optimal_model_dir = model_files[model_accs.index(max(model_accs))]
    return optimal_model_dir


def main():
    print("Loading Mask Detection Model...")
    mask_model = tf.keras.models.load_model(
            get_optimal_model(
                ARGS.GlobalArgs["model_dir"] + os.sep + AGS.model + "Model"
            )
        )
    print("Success! Mask Detection Model is Loaded as {}".format(AGS.model))

    print("Loading Face Recognition Model...")
    face_recognition_model = cv2.dnn.readNet(
        ARGS.GlobalArgs["face_recognition_dnn_prototxt_dir"],
        ARGS.GlobalArgs["face_recognition_dnn_dir"]
    )
    print("Success! Face Detection Model is Loaded.")

    cap = cv2.VideoCapture(0)

    while True:

        # Read the frame & flip horizontally
        _, img = cap.read()
        img = cv2.flip(img, 1)
        (h, w) = img.shape[:2]

        # Construct a blob from the image
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))

        # Pass the blob through the face recognition DNN and obtain detections
        face_recognition_model.setInput(blob)
        detections = face_recognition_model.forward()

        # Set our face detection confidence threshold
        confidence_threshold = 0.5

        # Loop over the detections
        for i in range(0, detections.shape[2]):

            # Calculate the confidence that this detection is a face
            confidence = detections[0, 0, i, 2]

            # Proceed if the confidence exceed our threshold
            if confidence > confidence_threshold:

                # Get the box that contains human face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, end_x, end_y) = box.astype("int")

                # Add a small boundary on both X & Y to capture whole face
                x_pad, y_pad = 20, 30
                x, y = max(0, x - x_pad), max(0, y - y_pad)
                end_x, end_y = min(w, end_x + x_pad), min(h, end_y + y_pad)

                # Get the corresponding face image & reshape
                face = img[y:end_y, x:end_x]
                face = cv2.resize(face, (500, 700))
                # cv2.imshow('img', face)

                # Convert to tensor
                face = tf.keras.preprocessing.image.img_to_array(face)
                face = np.expand_dims(face, axis=0)
                face = tf.convert_to_tensor(face, dtype="float32")

                # Make predictions using our trained model
                mask_types = ["with mask", "incorrect mask", "without mask"]

                prediction = mask_model.predict(face, verbose=0)[0]
                predict_type = np.argmax(prediction)
                predict_conf = round(np.max(prediction) * 100, 2)

                # Get our predicted label
                label = mask_types[predict_type]

                # Change rectangle's color based on prediction
                if predict_type == 0:
                    color = (0, 255, 0)
                elif predict_type == 1:
                    color = (255, 0, 0)
                else:
                    color = (0, 0, 255)

                # Add our predicted label and our prediction confidence to the image frame
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.putText(img, str(predict_conf) + "%", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

                # Draw the rectangle
                cv2.rectangle(img, (x, y), (end_x, end_y), color, 2)

        # Display the frame
        cv2.imshow('img', img)

        # Stop if escape key is pressed
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    # Release the VideoCapture object
    cap.release()


AGS = parse_args()
main()
