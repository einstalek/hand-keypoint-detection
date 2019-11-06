import time
from datetime import datetime

import cv2
import numpy as np

from detector import Detector
from kp_model import KeyPointModel


class Cursor:
    def __init__(self,
                 cap: cv2.VideoCapture,
                 detector_fp: str,
                 kp_model_fp: str,
                 h: int, w: int,
                 detector_h: int, detector_w: int,
                 kp_model_h: int, kp_model_w: int,
                 act_thresh: float = 0.1,
                 val_thresh: float = 0.1,
                 skip_frames: int = 1,
                 time_waiting_max: float = 5,
                 alpha=0.5):
        self.alpha = alpha
        self.cap = cap
        self.skip_frames = skip_frames

        # window params
        self.kp_model_h = kp_model_h
        self.kp_model_w = kp_model_w
        self.detector_h = detector_h
        self.detector_w = detector_w
        self.h = h
        self.w = w
        self.dw = 0
        self.scale_x, self.scale_y = None, None

        # bounding box location
        self.bx, self.by = None, None
        self.bh, self.bw = None, None
        self.kps = None

        self.kp_model_fp = kp_model_fp
        self.detector_fp = detector_fp

        self.detector: Detector = None
        self.kp_model: KeyPointModel = None

        self.frame_counter = 0

        self.val_thresh = val_thresh
        self.act_thresh = act_thresh

        self.started = False
        self.last_update = None
        self.hand_present = False
        self.time_waiting_max = time_waiting_max

        # some statistics
        self.act = 0.
        self.camera_fps = 0.
        self.kp_model_fps = 0.
        self.kp_model_handler_fps = 0.
        self.detector_fps = 0.
        self.detector_handler_fps = 0.
        self.draw_fps = 0.
        self.dt = 0.

        self.initialize_models()

    def initialize_models(self):
        self.log("Initializing detector")
        self.detector = Detector(self.detector_fp,
                                 h=self.detector_h, w=self.detector_w,
                                 parent=self, alpha=self.alpha)
        self.log("Initializing key-point model")
        self.kp_model = KeyPointModel(self.kp_model_fp,
                                      h=self.kp_model_h,
                                      w=self.kp_model_h,
                                      val_thresh=self.val_thresh,
                                      act_thresh=self.act_thresh,
                                      parent=self, alpha=self.alpha)

    @staticmethod
    def log(string: str):
        print("%s: %s" % (datetime.strftime(datetime.now(), "%H:%M:%S"), string))

    def initialize_bbox(self):
        # starting position and size of bounding box
        self.bx, self.by = self.w / 2, self.h / 2
        self.bw = (self.w - self.bx) / 2
        self.bh = (self.h - self.by) / 1

    def _kp_model_task(self, frame):
        x1, y1, x2, y2 = self.current_bbox()

        crop = frame[y1: y2, x1: x2, :]
        self.kp_model(crop, x1, y1)

    def _detector_task(self, frame):
        self.detector(frame)

    def initialize_session(self, frame):
        self.last_update = time.time()
        self._kp_model_task(frame)
        if time.time() - self.last_update > 1:
            pass

    def update_on_kps(self):
        mean_x = self.kps[:, 0].mean()
        mean_y = self.kps[:, 1].mean()

        self.bx = mean_x
        self.by = mean_y * 1.1

        min_x, max_x = np.min(self.kps[:, 0]), np.max(self.kps[:, 0])
        dw = min(abs(min_x - (self.bx - self.bw / 2)),
                 abs(max_x - (self.bx + self.bw / 2)),
                 )
        self.dw = dw / self.bw
        self.hand_present = True

        self.last_update = time.time()

    def update_on_bbox(self, bbox):
        _, x1, x2, y1, y2 = bbox

        self.hand_present = True

        w, h = x2 - x1, y2 - y1

        # self.bw, self.bh = w, h
        self.bw = max(w, h)
        self.bh = max(w, h)

        self.bx = x1 + self.bw / 2
        self.by = y1 + self.bh / 2

        self.last_update = time.time()

    def current_bbox(self):
        """
        :return: x1, y1, x2, y2
        """
        x1 = self.bx - self.bw / 2
        x2 = self.bx + self.bw / 2
        y1 = self.by - self.bh / 2
        y2 = self.by + self.bh / 2

        x1, x2 = int(x1), int(x2)
        y1, y2 = int(y1), int(y2)
        x1, x2 = max(0, x1), min(self.w, x2)
        y1, y2 = max(0, y1), min(self.w, y2)
        return x1, y1, x2, y2

    def on_new_frame(self, frame: np.array):
        self.frame_counter += 1
        self.dt = time.time()

        if self.skip_frames > 1 and self.frame_counter % self.skip_frames == 1:
            return

        if self.last_update is None:
            self.initialize_bbox()
        elif time.time() - self.last_update > self.time_waiting_max:
            self.kps = None
            self.hand_present = False

        if not self.hand_present:
            self.dw = 0.
            self._detector_task(frame)
        elif self.frame_counter % 40 == 1:
            self._detector_task(frame)
        else:
            self._kp_model_task(frame)

        if self.kps is not None and self.hand_present:
            for idx, (_x, _y) in enumerate(self.kps):
                cv2.circle(frame, (int(_x), int(_y)), 3, (0, 0, 255), 3)
        if self.hand_present:
            x1, y1, x2, y2 = self.current_bbox()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

        self.dt = time.time() - self.dt
        return frame
