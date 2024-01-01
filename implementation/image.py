import cv2 
import mtcnn
import numpy as np
from keras.models import load_model
from keras import preprocessing

import argparse
from PIL import Image 
import os
import time

"""
    This script for detecting emotion from a frame. the system will process through face detection, and predict.
"""

    #Example cmd 
    #python image.py -p test/filename.jpg
    #python image.py 

def detect_emotion(frame, detector, model=''):
    # model = "../models/LightCNN_lr001_20231119-012120.h5"
    # model = "../models/EfficientNetB0_lr001_224_20231128-081850.h5" # 88.02% 1.52s
    # model = "../models/EfficientNetB0_lr001_20231119-172240.h5"
    model = "../models/EfficientNetB0_lr0005_20231119-225644.h5"
    model = load_model(model)

    detected = detector.detect_faces(frame)
    
    crop = [] 
    faces = []
    locs = []

    for i, item in enumerate(detected):
        score = item["confidence"]
        if score >= 0.90:
            x1, y1, width, height = item['box']
            x2, y2 = x1 + width, y1 + height
            crop = frame[y1:y2, x1:x2]
            print(f'Face {i} : {crop.shape}')

            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = preprocessing.image.img_to_array(crop)
            crop = cv2.resize(crop, (48, 48)) # img input size 
            crop = crop 

            faces.append(crop)
            locs.append((x1, y1, x2, y2))
    
    if len(faces) > 0 :
        # predict emotion from face detection
        faces = np.array(faces, dtype="float32")
        st = time.time() 
        preds = model.predict(faces,  batch_size=32)
        elapsed = time.time() - st

        print(f'waktu prosess : {elapsed:0.2f} s')
    return locs, preds

def process(img_path, output_name='output.jpg'):

    detector = mtcnn.MTCNN()
    img = cv2.imread(img_path)

    locs, preds = detect_emotion(img, detector=detector)
    CLASS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    for box, pred in zip(locs, preds):
        x1, y1, x2, y2 = box

        label = CLASS[np.argmax(pred)].upper()
  
        color = (0, 255, 0)
        SHOW_CONF = False

        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

        if SHOW_CONF :
            confidance = (100 * np.max(pred)).round(1)
            confidance = str(confidance) + ' %'
            conf_pos = x2 - int(0.4*(x2-x1))
            cv2.putText(img, confidance, (conf_pos, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   
    out = Image.fromarray(img)
    out.save(output_name, dpi=(120,120), format='JPEG', quality=95)
    print(f'{output_name} processed')
    return img

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--picture", type=str,
                    default="test/1fr.jpg",
                    help="path to image of virtual meeting ")
    ap.add_argument("-m", "--model", type=str,
                    default="models/EfficientNetB0_lr0001_20231120-012401.h5",
                    help="path to emotion classifier model")
    ap.add_argument("-o", "--output", type=str,
                    default="detected/",
                    help="path to output folder")
    
    args = vars(ap.parse_args())

    # print(args)
    img_path = args["picture"]
    output_path = f'test/output_{img_path.split("/")[-1]}'

    input_folder = 'frames/'
    output_folder = 'detected_eff0005/'

    print(args)

    # start_time = time.time()
    # for fi in os.listdir(input_folder):
    #     # print(fi)
    #     input_path = os.path.join(input_folder, fi)
    #     output_path = os.path.join(output_folder, fi)
    #     # print(input_path)
    #     # print(output_path)
    #     process(input_path, output_name=output_path)
    #     # print(f'{output_path} processed')

    # elapsed_time = time.time() - start_time
    # print(f'elapsed {elapsed_time / 60 :0.2f} menit' )

    # uncomment untuk single file
    img = process(img_path, output_name=output_path)

    # cv2.namedWindow('emot', cv2.WINDOW_FREERATIO)
    # cv2.imshow('emot', img)

    # cv2.waitKey(0)