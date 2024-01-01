import cv2 
import mtcnn
import numpy as np

from keras.models import load_model
from keras import preprocessing

import argparse
from PIL import Image 

import os
import time

def process(img_path, output_name='output.jpg', model=''):

    img_raw = cv2.imread(img_path)
    img_size = 48

    CLASS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    # img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    img = preprocessing.image.img_to_array(img)
    img = cv2.resize(img, (img_size, img_size)) #resize img
    img = np.expand_dims(img, axis=0)

    st = time.time() 
    pred = model.predict(img,  batch_size=32)
    elapsed = time.time() - st

    label = CLASS[np.argmax(pred)].upper()

    confidance = (100 * np.max(pred)).round(1)
    confidance = str(confidance) + ' %'

    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   
    out = Image.fromarray(img_raw)
    out.save(output_name, dpi=(120,120), format='JPEG', quality=95)

    print(f'========================')
    print(f'Time processs : {elapsed:0.3f} s')
    print(f'Label         : {label} ')
    print(f'Confidance    : {confidance}')
    print(f'processed     : {output_name}')
    print(f'========================')


    return img, pred


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--picture", type=str,
                    default="test/1fr.jpg",
                    help="path to image of virtual meeting ")
    ap.add_argument("-m", "--model", type=str,
                    default="models/EfficientNetB0_lr0001_20231120-012401.h5",
                    help="path to emotion classifier model")
    
    args = vars(ap.parse_args())

    img_path = args["picture"]
    output_path = f'test/output_{img_path.split("/")[-1]}'

    # model = "../models/EfficientNetB0_lr001_20231119-172240.h5" #90.04%
    model = "../models/EfficientNetB0_lr0005_20231119-225644.h5"
    # model = "../models/EfficientNetB0_lr001_224_20231128-081850.h5" # 88.02% 224x224
    # model = "../models/LightCNN_lr001_20231119-012120.h5"
    model = load_model(model)

    process(img_path, output_name=output_path, model=model)

   
