import cv2
import mtcnn
from keras.models import load_model
from keras import preprocessing
import numpy as np

import time
# Function to calculate and display FPS
class FPSCalculator:
    def __init__(self):
        self.prev_time = cv2.getTickCount()

    def get_fps(self):
        current_time = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (current_time - self.prev_time)
        self.prev_time = current_time
        return fps
    
def detect_emotion(frame, detector, model=""):
    crop = [] 
    
    detected = detector.detect_faces(frame)
    
    faces = []
    locs = []
    preds = []

    for i, item in enumerate(detected):
        score = item["confidence"]
        if score >= 0.90:
            x1, y1, width, height = item['box']
            x2, y2 = x1 + width, y1 + height
            crop = frame[y1:y2, x1:x2]
            print(f'Face {i} : {crop.shape}')

            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = preprocessing.image.img_to_array(crop)
            crop = cv2.resize(crop, (224, 224))
            crop = crop / 255.0
            # crop = np.expand_dims(crop, axis=0)

            faces.append(crop)
            locs.append((x1, y1, x2, y2))
    
    if len(faces) > 0 :
        # predict emotion from face detection
        faces = np.array(faces, dtype="float32")
        preds = model.predict(faces,  batch_size=32)

    return locs, preds

cap = cv2.VideoCapture(0)

classifier = 'xml/haarcascade_frontalface_default.xml'
detector = cv2.CascadeClassifier(classifier)

model = "../models/EfficientNetB0_lr001_20231119-172240.h5" # 90%
model = load_model(model)

# facedetection - arry, 
# prediction - class, confidance

while True:
    # Read a frame from the camera
    fps_calculator = FPSCalculator()

    ret, frame = cap.read()

    if not ret:
        break

    start_time = time.time()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(gray, scaleFactor=1.05,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)
    
    print(rects)
    for faces in rects:
        # draw the face bounding box on the image
        (x, y, w, h) = faces
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # ---> ouput : rect, croped
    img_size = 48
    faces = np.empty((0, img_size, img_size, 3), dtype="float32")
    locs = []
    preds = []
    
    for face in rects:
        crop = [] 
        (x1, y1, width, height) = face
        x2, y2 = x1 + width, y1 + height
        crop = frame[y1:y2, x1:x2]

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = preprocessing.image.img_to_array(crop)
        crop = cv2.resize(crop, (img_size, img_size)) #resize img
        crop = crop 
        
        faces = np.append(faces, [crop], axis=0) 
        
        if len(faces) > 0 :
            # predict emotion from face detection
            preds = model.predict(faces,  batch_size=32)

    CLASS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    for box, pred in zip(rects, preds):
        x1, y1, width, height = box
        x2, y2 = x1 + width, y1 + height

        label = CLASS[np.argmax(pred)].upper()

        color = (0, 255, 0)        
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # Display FPS 
    elapsed = time.time() - start_time
    if elapsed == 0:
        fps = 9999
    else :
        fps = 1. / elapsed
        fps = round(fps, 2)
    # print(f'FPS   : {fps} ')
    fps_text = f"FPS: {fps}"

    # 
    cv2.putText(frame, fps_text, (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
