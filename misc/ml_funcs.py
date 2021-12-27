from misc.model_inputs import YoloInputArguments
from misc.detect import detect
import cv2


def get_predictions(img):
    yolo_args = YoloInputArguments(
        output='output',
        source='input',
        weights='model/best_ap50.pt',
        view_img=False,
        save_txt=True,
        img_size=640,
        cfg='model/yolor_p6_care_dent.cfg',
        names='model/data.names'
    )

    # In the future directly send image data as numpy array
    cv2.imwrite('./input/temp.jpg', img)
    detect('./input/temp.jpg',yolo_args)
