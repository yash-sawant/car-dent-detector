import json
import numpy as np
import time
import cv2
import os
from flask import Flask, request, Response
import io
from PIL import ImageFile, Image
from misc.ml_funcs import get_predictions

ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__)



LABELS = open('./model/data.names').read().strip().split('\n')

def convert_to_json(tags):
    '''
    Converting detected boxes to require json format
    '''
    results = {}
    for tag in tags:
        if len(tag.split()) == 5:
            class_num, top, left, height, width = tag.split()
            results[LABELS[int(class_num)]] = [top, left, height, width]
    return results


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

    # Inference
    # Optimisation is pending
    np_img = np.array(img)
    get_predictions(np_img)
    tags = open('output/temp.txt').read().strip().split('\n')

    # Formatting output
    resp = convert_to_json(tags)
    return Response(response=json.dumps(resp), status=200, mimetype='application/json')


if __name__ == '__main__':
    app.run(port=80, host='0.0.0.0')
