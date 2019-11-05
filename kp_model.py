import time

import cv2
import numpy as np
import tensorflow as tf
from sklearn.mixture import GaussianMixture
from tensorflow.python import keras


def sigmoid(arr):
    return 1. / (1. + np.exp(-arr))


class KeyPointModel:
    def __init__(self,
                 fp: str,
                 h: int,
                 w: int,
                 parent: 'Cursor',
                 act_thresh: float = 0.3,
                 val_thresh: float = 0.2,
                 alpha=0.5):
        self.alpha = alpha
        self.model_fp = fp
        self.parent = parent
        self.h = h
        self.w = w
        self._model = None
        self.crop = None
        self.x1, self.y1 = None, None

        self.act_thresh = act_thresh
        self.val_thresh = val_thresh
        self.gmm = GaussianMixture(n_components=5)

        self._model = keras.models.load_model(self.model_fp,
                                              custom_objects={'loss': lambda x, y: tf.reduce_mean(x - y)})
        self.log("Key-point model initialized")

    def decode_openpose(self, pred):
        mask = sigmoid(pred[-1][0, ..., 0])
        mask = cv2.resize(mask, (self.w, self.h))
        self.parent.act = mask.max()

        if self.parent.act >= self.act_thresh:
            yy, xx = np.where(mask > self.val_thresh)
            zz = np.concatenate([xx[..., np.newaxis], yy[..., np.newaxis]], axis=1)
            if zz.shape[0] < 5:
                return

            _ = self.gmm.fit(zz)
            kps = self.gmm.means_
            return kps

    def run(self) -> None:
        h, w, _ = self.crop.shape
        scale_x, scale_y = w / self.w, h / self.h

        crop = cv2.resize(self.crop, (self.h, self.w))

        _t1 = time.time()
        pred = self._model.predict(crop[np.newaxis])
        # pred = self._model.predict(crop[np.newaxis])
        # kps = pred[0].reshape((-1, 2))
        _t2 = time.time()
        self.parent.kp_model_fps = self.alpha * self.parent.kp_model_fps + (1 - self.alpha) / (_t2 - _t1)

        kps = self.decode_openpose(pred)
        if kps is not None:
            kps[:, 0] *= scale_x
            kps[:, 1] *= scale_y
            kps[:, 0] += self.x1
            kps[:, 1] += self.y1

            _t3 = time.time()
            self.parent.kp_model_handler_fps = self.alpha * self.parent.kp_model_handler_fps + (1 - self.alpha) / (_t3 - _t2)

            self.parent.kps = kps
            self.parent.update_on_kps()

    def __call__(self, frame: np.array, x1: int, y1: int, *args, **kwargs):
        self.crop = frame.copy()
        self.x1, self.y1 = x1, y1
        self.run()

    def log(self, string: str):
        self.parent.log(string)
