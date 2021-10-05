from flask import Flask, request, jsonify, render_template
import numpy as np
# from keras.models import load_model
from keras.models import model_from_json
import cv2
import os
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
app=Flask(__name__,template_folder='templates')

json_file = open('output/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("output/model.h5")

def predictions(video_dir, model, nb_frames = 25, img_size = 224):

    X = frames_from_video(video_dir, nb_frames, img_size)
    X = np.reshape(X, (1, nb_frames, img_size, img_size, 3))
    
    predictions = model.predict(X)
    preds = predictions.argmax(axis = 1)

    classes = []

    with open(os.path.join('output', 'classes.txt'), 'r') as fp:
        for line in fp:
            classes.append(line.split()[1])
    
    res  = []
    for i in range(len(preds)):
        print('Prediction - {} -- {}'.format(preds[i], classes[preds[i]]))
        res.append([preds[i],classes[preds[i]]])
    
    return res

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


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    # '''
    
    # model = load_model('output/slowfast_finalmodel.hd5')
    features = request.form.get('Camera')
    print(features)
    source = ''

    if features == "camera1":
        source = "test/Arrest048_x264_21.mp4"
    elif features == "camera2":
        source = "test/Abuse014_x264_1.mp4"
    elif features == "camera3":
        source = "test/Robbery126_x264_3.mp4"
    else:
        source = "test/Shoplifting018_x264_19.mp4"
        
    res = predictions(video_dir = source, model = model, nb_frames = 25, img_size = 224)
    resstr = 'Suspicious Activity :' + str(res[0][1]) 
    return render_template('index.html', prediction_text=resstr)

if __name__ == "__main__":
    app.run(debug=True)