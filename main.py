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

DETECT_CMD = 'python detect.py --weights \
../model/best_ap50.pt --conf 0.5 --img-size 640 \
--conf-thres 0.5 --iou-thres 0.65 --names ../model/data.names \
--source ../input/ --cfg ../model/yolor_p6_care_dent.cfg \
--save-txt --output ../output --device 0'

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
    cv2.imwrite('./input/temp.jpg', np_img)
    os.chdir('./yolor')
    os.system(DETECT_CMD)
    os.chdir('../')
    tags = open('output/temp.txt').read().strip().split('\n')

    # Formatting output
    resp = convert_to_json(tags)
    return Response(response=json.dumps(resp), status=200, mimetype='application/json')


if __name__ == '__main__':
    app.run(port=80, host='0.0.0.0')
