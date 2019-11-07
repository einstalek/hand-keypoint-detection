import time

import cv2
import numpy as np

from build_detector import mn_model, decode_y2


class Detector:
    def __init__(self,
                 fp: str,
                 h: int,
                 w: int,
                 parent: 'Cursor',
                 alpha=0.5):
        self.alpha = alpha
        self.model_fp = fp
        self.parent = parent
        self.h = h
        self.w = w
        self._model = None
        self.frame = None
        self.scale_x = self.w / self.parent.w
        self.scale_y = self.h / self.parent.h

        det, *_ = mn_model(image_size=(self.h, self.w, 3),
                           n_classes=2,
                           min_scale=None,
                           max_scale=None,
                           scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
                           aspect_ratios_global=None,
                           aspect_ratios_per_layer=[[0.5, 1.0, 2.0],
                                                    [1.0 / 3.0, 0.5, 1.0, 2.0, 3.0],
                                                    [1.0 / 3.0, 0.5, 1.0, 2.0, 3.0],
                                                    [1.0 / 3.0, 0.5, 1.0, 2.0, 3.0],
                                                    [0.5, 1.0, 2.0],
                                                    [0.5, 1.0, 2.0]],
                           variances=[0.1, 0.1, 0.2, 0.2],
                           # The variances by which the encoded target coordinates
                           # are scaled as in the original implementation
                           normalize_coords=True
                           )
        det.load_weights(self.model_fp)
        self._model = det
        self.log("Detector initialized")

    def run(self) -> None:
        frame = cv2.resize(self.frame, (self.h, self.w))
        _t1 = time.time()
        pred = self._model.predict(255 * frame[np.newaxis])
        _t2 = time.time()
        # self.parent.detector_fps = self.alpha * self.parent.detector_fps + (1 - self.alpha) / (_t2 - _t1)
        self.parent.detector_fps = self.alpha * self.parent.detector_fps + (1 - self.alpha) * (_t2 - _t1)

        boxes = decode_y2(pred,
                          confidence_thresh=0.6,  # _CONF, #уверенность
                          iou_threshold=0.05,  # _IOU, #пересечения
                          top_k=1,
                          normalize_coords=True,
                          img_height=self.h,
                          img_width=self.w)

        if len(boxes[0]) > 0:
            _, p, x1, x2, y1, y2 = boxes[0][0]

            x1 /= self.scale_x
            x2 /= self.scale_x
            y1 /= self.scale_y
            y2 /= self.scale_y

            _t3 = time.time()
            # self.parent.detector_handler_fps = self.alpha * self.parent.detector_handler_fps + (1 - self.alpha) / (_t3 - _t2)
            self.parent.detector_handler_fps = self.alpha * self.parent.detector_handler_fps + (1 - self.alpha) * (
                        _t3 - _t2)
            self.parent.update_on_bbox([p, x1, x2, y1, y2])

    def __call__(self, frame: np.array, *args, **kwargs):
        self.frame = frame
        self.run()

    def log(self, string: str):
        self.parent.log(string)
