# importing packages
from flask_bootstrap import Bootstrap
import tensorflow as tf
import keras
from flask import Flask, redirect, render_template, request, session, url_for
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
import numpy as np
from keras.models import load_model
from PIL import Image
from keras.utils.np_utils import to_categorical
import collections
from collections import defaultdict
import os
import io
import requests
from keras.applications.inception_v3 import preprocess_input
from keras.backend import set_session
import pandas as pd 

# session set and model importing
sess = tf.Session()
set_session(sess)
model = load_model('model.hdf5')
graph = tf.get_default_graph()
# loading classes txt
classes = {}
ix_to_class = {}
with open('classes.txt', 'r') as line:
    classes = [l.strip() for l in line.readlines()]
    classes = dict(zip(classes, range(len(classes))))
    ix_to_class = dict(zip(range(len(classes)), classes))
    classes = {v: k for k, v in ix_to_class.items()}
sorted_classes = collections.OrderedDict(sorted(classes.items()))
min_side =299
data = pd.read_csv('calories.csv')

app = Flask(__name__)
dropzone = Dropzone(app)
Bootstrap(app)

app.config['SECRET_KEY'] = 'supersecretkeygoeshere'

# Dropzone settings
app.config['DROPZONE_UPLOAD_MULTIPLE'] = False
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
app.config['DROPZONE_REDIRECT_VIEW'] = 'results'

# Uploads settings
app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd() + '/uploads'

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB


@app.route('/', methods=['GET', 'POST'])
def index():
    
    # set session for image results
    if "file_urls" not in session:
        session['file_urls'] = []
    # list to hold our uploaded image urls
    file_urls = session['file_urls']

    # handle image upload from Dropszone
    if request.method == 'POST':
        file_obj = request.files
        for f in file_obj:
            file = request.files.get(f)
            
            # save the file with to our photos folder
            filename = photos.save(
                file,
                name=file.filename    
            )

            # append image urls
            file_urls.append(photos.url(filename))
            
        session['file_urls'] = file_urls
        return "uploading..."
    # return dropzone template on GET request    
    return render_template('index.html')


@app.route('/results')
def results():
    
    # redirect to home if no images to display
    if "file_urls" not in session or session['file_urls'] == []:
        return redirect(url_for('index'))
        
    # set the file_urls and remove the session variable
    file_urls = session['file_urls']
    session.pop('file_urls', None)
    print(file_urls)
    for files in file_urls:
        # upload_image = Image.open(files)
        response = requests.get(files)
        # image = Image.open(urllib.request.urlopen(files))
        image_bytes = io.BytesIO(response.content)
    image = Image.open(image_bytes)
    upload_image = image      
    keras.backend.clear_session ()
    width, height = upload_image.size   # Get dimensions
    left = (width - min_side)/2
    top = (height - min_side)/2
    right = (width + min_side)/2
    bottom = (height + min_side)/2
    # Crop the center of the image
    upload_image = upload_image.crop((left, top, right, bottom))
    newsize = (min_side, min_side)
    upload_image = upload_image.resize(newsize) 
    upload_array = np.array(upload_image)   
    # importing model
    #==========================================================================================================
    # prediciton part
    def center_crop(x, center_crop_size, **kwargs):
        centerw, centerh = x.shape[0]//2, x.shape[1]//2
        halfw, halfh = center_crop_size[0]//2, center_crop_size[1]//2
        return x[centerw-halfw:centerw+halfw+1,centerh-halfh:centerh+halfh+1, :]
    img=upload_array
    flipped_X = np.fliplr(img)
    crops = [
        img[:299,:299, :], # Upper Left
        img[:299, img.shape[1]-299:, :], # Upper Right
        img[img.shape[0]-299:, :299, :], # Lower Left
        img[img.shape[0]-299:, img.shape[1]-299:, :], # Lower Right
        center_crop(img, (299, 299)),
        
        flipped_X[:299,:299, :],
        flipped_X[:299, flipped_X.shape[1]-299:, :],
        flipped_X[flipped_X.shape[0]-299:, :299, :],
        flipped_X[flipped_X.shape[0]-299:, flipped_X.shape[1]-299:, :],
        center_crop(flipped_X, (299, 299))
        ]
    # if preprocess:
    # crops = [preprocess_input(x.astype('float32')) for x in crops]
    top_n=5
    global sess
    global model
    with graph.as_default():
        set_session(sess)
        y_pred = model.predict(np.array(crops))
        preds = np.argmax(y_pred, axis=1)
        top_n_preds= np.argpartition(y_pred, -top_n)[:,-top_n:]
        # print(upload_image)
    food_name = list(classes.keys())[list(classes.values()).index(preds[1])]
    data1 = data[data['ID'] == food_name]
    name = data1['NAME'].to_string(index = False)
    calories = data1['CALORIES'].to_string(index = False)
    fat = data1['FAT'].to_string(index = False)
    cholestrol = data1['CHOLESTROL'].to_string(index = False)
    carb = data1['CARBOHYDRATES'].to_string(index = False)
    fiber = data1['FIBER'].to_string(index = False)
    protein = data1['PROTEIN'].to_string(index = False)
    return render_template('results.html', file_urls=file_urls[0] ,name = name ,cal = calories,fat = fat,cholestrol = cholestrol,carbohydrates = carb , fiber = fiber,protein = protein)

if __name__ == "__main__":
    app.run(debug=True , port = "5051")
