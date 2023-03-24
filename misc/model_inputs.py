class YoloInputArguments:
    
    def __init__(self, output, source, weights, view_img, save_txt, img_size, cfg, names):
        self.output = output
        self.source =source
        self.weights=[weights]
        self.view_img=view_img
        self.save_txt=save_txt
        self.img_size=img_size
        self.cfg=cfg
        self.names=names
        self.webcam = False #source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
        self.conf_thres = 0.5
        self.iou_thres = 0.65
        self.device = 0
        self.classes = ''
        self.agnostic_nms = False
        self.augment = False
        self.update = False
