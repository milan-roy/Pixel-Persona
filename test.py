import cv2
import numpy as np
from tensorflow.keras.models import load_model
import keras

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load trained CNN model
age_model = load_model(
    "Models/model_age.h5",
    custom_objects={
        "mse": keras.losses.MeanSquaredError
    }
)
gender_model = load_model("Models/model_gender.h5")

# Define image size expected by model
IMG_SIZE = (128, 128)   

# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Draw bounding box
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        
        # Preprocess face for model
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = cv2.resize(face_img, IMG_SIZE)
        face_img = face_img.astype("float32") / 255.0
        face_img = np.expand_dims(face_img, axis=0)
        
        
        age_pred = age_model.predict(face_img, verbose=0)
        gender_pred = gender_model.predict(face_img, verbose=0)
        
        # If model outputs regression (e.g. age as number)
        if age_pred.shape[-1] == 1:
            age = int(age_pred[0][0])
    
        gender_pred = (gender_pred > 0.5).astype("int32") 
        gender = 'Male' if gender_pred==0 else 'Female'

        # Show age above box
        cv2.putText(frame, f"Age: {age}, Gender: {gender}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("Age Detection", frame)
    
    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
