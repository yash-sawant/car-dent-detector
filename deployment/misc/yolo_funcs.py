import time
import numpy as np
import cv2

IMAGE_H = 640
IMAGE_W = 640
LABEL_PATH = '../../model/data.names'
WEIGHTS_PATH = '../../model/best_ap50.pt'
CONFIG_PATH = '../../model/yolor_p6_care_dent.cfg'
LABELS = open(LABEL_PATH).read().strip().split("\n")
CONF_THRES = 0.5
NMS_THRES = 0.1



net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)


def get_prediction(image):
    (H, W) = image.shape[:2]
    global net,LABELS
    output_response = {"predictions": []}
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (IMAGE_H, IMAGE_W),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    # print(layerOutputs)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            # print(scores)
            classID = np.argmax(scores)
            # print(classID)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > CONF_THRES:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                centerX, centerY, width, height = box

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(int(classID))
    output_response = {}
    if len(boxes) > 0:

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRES,
                                NMS_THRES)
        idxs = idxs.flatten()
        boxes = [boxes[i] for i in idxs]
        confidences = [confidences[i] for i in idxs]
        classIDs = [classIDs[i] for i in idxs]

        item_list = []
        # ensure at least one detection exists

        # loop over the indexes we are keeping
        for i, box in enumerate(boxes):
            # extract the bounding box coordinates
            (x, y) = (box[0], box[1])
            (w, h) = (box[2], box[3])

            item = {"boundingBox": {"height": -1, "left": -1, "top": -1, "width": -1}, "probability": -1, "tagId": -1,
                    "tagName": ""}

            item["boundingBox"]["height"] = round(h / H, 7)
            item["boundingBox"]["width"] = round(w / W, 7)
            item["boundingBox"]["left"] = round(x / W, 7)
            item["boundingBox"]["top"] = round(y / H, 7)

            item["probability"] = round(confidences[i], 8)
            item["tagId"] = int(classIDs[i])
            item["tagName"] = LABELS[classIDs[i]]
            
            item_list.append(item)

        output_response["predictions"] = item_list

    return output_response
