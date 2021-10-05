import numpy as np
from keras.models import load_model
import cv2
from keras.models import model_from_json
import os
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def predictions(video_dir, model, nb_frames = 25, img_size = 224):
    X = frames_from_video(video_dir, nb_frames, img_size)
    X = np.reshape(X, (1, nb_frames, img_size, img_size, 3))
    predictions = model.predict(X)
    preds = predictions.argmax(axis = 1)
    classes = []
    
    with open(os.path.join('output', 'classes.txt'), 'r') as fp:
        for line in fp:
            classes.append(line.split()[1])

    for i in range(len(preds)):
        print('Prediction - {} -- {}'.format(preds[i], classes[preds[i]]))
        

def frames_from_video(video_dir, nb_frames = 25, img_size = 224):
    
    cap = cv2.VideoCapture(video_dir)
    i=0
    frames = []

    while(cap.isOpened() and i<nb_frames):
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.resize(frame, (img_size, img_size))
        frames.append(frame)
        i+=1
    cap.release()
    cv2.destroyAllWindows()
    return np.array(frames) / 255.0

json_file = open('output/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("output/model.h5")

predictions(video_dir = 'test/Shoplifting018_x264_19.mp4', model = model, nb_frames = 25, img_size = 224)