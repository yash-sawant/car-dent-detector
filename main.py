import json
import numpy as np
import time
import cv2
import os
from flask import Flask, request, Response
import io
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello_world():
    return 'Hello World'


@app.route('/image', methods=['POST'])
def main():
    try:
        if 'img' in request.files:
            img = request.files['img']
        elif 'img' in request.form:
            img = request.form['img']
        else:
            img = io.BytesIO(request.get_data())

        img = Image.open(img)
    except Exception as e:

        print('EXCEPTION:', str(e))
        return 'Error processing image', 500
    np_img = np.array(img)
    H, W, C = np_img.shape
    resp = {'Height': H, 'Width': W, 'Channels': C}
    return Response(response=json.dumps(resp), status=200, mimetype='application/json')


if __name__ == '__main__':
    app.run(port=80, host='0.0.0.0')
