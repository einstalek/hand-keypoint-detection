# Code for Hand Key point Detector


## Model architecture
Model consists of two parts: 
- Detector: predicts bounding box for single hand
- Key point model: predicts coordinates for fingers. 


## Code 
* ```detector.py```: code for detector model
* ```kp_model.py```: code for key point model
* ```cursor.py```: Cursor class uses both aforementioned models to track hand
* ```main.py```: runs Cursor activity on video stream from camera using opencv and displays some statistics


## FPS
On MacBook pro 2017 I got following statistics:
* Detector model: 5-6 fps
* KeyPoint model: ~10 fps

## Example
![Alt Text](https://github.com/einstalek/hand-keypoint-detection/blob/master/examples/clip.gif)



