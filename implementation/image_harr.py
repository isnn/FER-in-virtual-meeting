import cv2 
import mtcnn
import numpy as np
# import matplotlib.pyplot as plt

from keras.models import load_model
from keras import preprocessing

import argparse
from PIL import Image 

import os
import time

# task 
# buat argparse untuk split images > source video, folder output, prefix
# split output name -> frame_menit:detik_framenum

def detect_emotion(frame, detector, model=''):
    model = "../models/EfficientNetB0_lr001_20231119-172240.h5" 
    model = load_model(model)
    crop = [] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(gray, scaleFactor=1.05,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)
    
    faces = []
    img_size = 48
    faces = np.empty((0, img_size, img_size, 3), dtype="float32")
    locs = []
    preds = []

    for face in rects:
        # display face bounding box 
        (x, y, w, h) = face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

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
            faces = np.array(faces, dtype="float32")
            preds = model.predict(faces,  batch_size=32)

    return rects, preds

def process(img_path, output_name='output.jpg'):

    classifier = 'xml/haarcascade_frontalface_default.xml'
    detector = cv2.CascadeClassifier(classifier)
    img = cv2.imread(img_path)

    locs, preds = detect_emotion(img, detector=detector)
    CLASS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    SHOW_CONF = True

    for box, pred in zip(locs, preds):
        x1, y1, x2, y2 = box

        label = CLASS[np.argmax(pred)].upper()
        color = (0, 255, 0)
        
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
                    default="frames/1fr.jpg",
                    help="path to virtual meeting file ")
    ap.add_argument("-m", "--model", type=str,
                    default="mobileNet_fer13.h5",
                    help="path to emotion classifier model")

    args = vars(ap.parse_args())

    # print(args)
    img_path = args["picture"]

    input_folder = 'frames/'
    output_folder = 'detected_v2/'
    fname = '+++/rasyid.jpg'

    process(fname, output_name='gssfx.jpg')

    # start_time = time.time()
    # for fi in os.listdir('frames/'):
    #     # print(fi)
    #     input_fname = os.path.join(input_folder, fi)
    #     output_fname = os.path.join(output_folder, fi)
    #     # print(input_fname)
    #     # print(output_fname)
    #     process(input_fname, output_name=output_fname)
    #     # print(f'{output_fname} processed')

    # elapsed_time = time.time() - start_time
    # print(f'elapsed {elapsed_time / 60 :0.2f} menit' )
    # # process(img_path, output_name=fname)

    # install tensorflow GPU

    # cv2.namedWindow('emot', cv2.WINDOW_FREERATIO)
    # cv2.imshow('emot', img)

    # cv2.waitKey(0)