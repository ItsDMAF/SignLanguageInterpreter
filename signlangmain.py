import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.models import load_model

model = load_model("model/efficientnetv2s_asl_model.keras")

class_labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
    'Nothing', 'O', 'P', 'Q', 'R', 'S', 'Space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]

IMG_SIZE = (384, 384)
correct_text = ""
incorrect_text = ""

cap = cv2.VideoCapture(0)
print("[INFO] ASL Interpreter Running. 'Q' to Quit | '1' to Confirm Correct | '0' to Mark Incorrect")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    roi = frame[100:400, 100:400]
    img = cv2.resize(roi, IMG_SIZE)
    img_array = preprocess_input(img.astype(np.float32))
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    pred_idx = np.argmax(predictions[0])
    pred_label = class_labels[pred_idx]
    confidence = predictions[0][pred_idx]

    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)
    cv2.putText(frame, f"{pred_label} ({confidence*100:.1f}%)", (100, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.putText(frame, f"Correct: {correct_text}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  
    cv2.putText(frame, f"Incorrect: {incorrect_text}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  


    cv2.imshow("ASL Interpreter", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('1'):
        if pred_label == "Space":
            correct_text += " "
        elif pred_label != "Nothing":
            correct_text += pred_label
    elif key == ord('0'):
        if pred_label == "Space":
            incorrect_text += " "
        elif pred_label != "Nothing":
            incorrect_text += pred_label
    elif key == 8:
        correct_text = correct_text[:-1]
        incorrect_text = incorrect_text[:-1]
    elif key == ord('x'):
        correct_text = ""
        incorrect_text = ""

cap.release()
cv2.destroyAllWindows()
