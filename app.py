import os, sys, re, math, random
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_bootstrap import Bootstrap
from werkzeug import secure_filename
import six.moves.urllib as urlib
from collections import defaultdict
from io import StringIO

from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

import custom
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_instances
from mrcnn.visualize import display_images
from mrcnn import model as modellib, utils
from mrcnn.model import log
from mrcnn.config import Config

app = Flask(__name__)
bootstrap = Bootstrap(app)

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

def get_ax(rows=1, cols=1, size=16):
    """
    Return a matplotlib axes array to be used in all visulizations in the notebook
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('uploaded_file', filename=filename))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
       
    PATH_TO_TEST_IMG_DIR = app.config['UPLOAD_FOLDER']

    INFERENCE_IMG_PATH = os.path.join(PATH_TO_TEST_IMG_DIR, filename)

    config = custom.CustomConfig()

    TEST_MODE = "inference"



    model = modellib.MaskRCNN(mode='inference', model_dir= 'MODEL_DIR',
                            config=config)

    weights_path = "mask_rcnn_object_0004.h5"
    model.load_weights(weights_path, by_name=True)

    img = Image.open(INFERENCE_IMG_PATH)
    if img.mode == "RGBA":
        b = Image.new("RGB", img.size, (255, 255, 255))
        b.paste(img, mask = img.split()[3])
        b.save("sample.jpg", "JPEG")
        image1 = mpimg.imread("sample.jpg")
    else:
        image1 = mpimg.imread(INFERENCE_IMG_PATH)

    result1 = model.detect([image1], verbose=1)

    #display images
    ax = get_ax(1)
    r1 = result1[0]

    class_names = 'crane handle'
    masked_img = visualize.save_display_instances(image1, r1['rois'], r1['masks'], r1['class_ids'],
                                class_names, r1['scores'], ax=ax,
                                title="Predictions")
    #plt.show(block=True)
    ax.imshow(masked_img.astype(np.uint8))

    plt.savefig('uploads/'+filename, bbox_inches='tight',
            pad_inches=-0.5, orientation= 'landscape')


    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)