import time

import cv2

from cursor import Cursor


def add_text(pic, text):
    global text_num
    cv2.putText(pic, text, (30, 30 + 25 * text_num), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    text_num += 1
    return pic


if __name__ == "__main__":
    frame_to_draw = None
    new = False
    alpha = 0.5

    scale = 1.8
    cursor = Cursor(cv2.VideoCapture(0),
                    detector_fp="weights/detector_weights.h5",
                    kp_model_fp="weights/openpose-cut-small-v3.hdf5",
                    # kp_model_fp="weights/regression-v2.hdf5",
                    h=720, w=1280,
                    detector_h=512, detector_w=512,
                    kp_model_h=224, kp_model_w=224,
                    act_thresh=0.6,
                    val_thresh=0.35,
                    skip_frames=1,
                    time_waiting_max=0.5,
                    alpha=alpha)

    while True:
        text_num = 0
        t1 = time.time()
        read, pic = cursor.cap.read()
        t2 = time.time()
        cursor.camera_fps = alpha * cursor.camera_fps + (1 - alpha) / (t2 - t1)

        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB) / 255.
        pic = cursor.on_new_frame(pic)

        if pic is not None:
            add_text(pic,
                     "Camera FPS: %.2f" % cursor.camera_fps)
            add_text(pic,
                     "KP model FPS: %.2f" % cursor.kp_model_fps)
            add_text(pic,
                     "KP model handler FPS: %.2f" % cursor.kp_model_handler_fps)
            add_text(pic,
                     "Detector FPS: %.2f" % cursor.detector_fps)
            add_text(pic,
                     "Detector handler FPS: %.2f" % cursor.detector_handler_fps)
            add_text(pic,
                     "Draw FPS: %.3f" % cursor.draw_fps)

            add_text(pic, "act: %.3f" % cursor.act)
            add_text(pic, "dt: %.3f" % cursor.dt)

        t3 = time.time()

        pic = cv2.resize(pic, (int(cursor.w / scale), int(cursor.h / scale)))
        cv2.imshow('frame', pic[:, :, ::-1])
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

        t4 = time.time()
        cursor.draw_fps = alpha * cursor.draw_fps + (1 - alpha) / (t4 - t3)
