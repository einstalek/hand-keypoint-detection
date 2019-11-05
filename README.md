# Code for Hand Key point Detector


## Model architecture
Model consists of two parts: 
- Detector: predicts bounding box for single hand
- Key point model: predicts coordinates for fingers. 

When a hand is present in picture, Key point model is being used to shift bounding box


## Code 
* ```detector.py```: code for detector model
* ```kp_model.py```: code for key point model
* ```cursor.py```: Cursor class uses both aforementioned models to track hand
* ```main.py```: runs Cursor activity on video stream from camera using opencv and displays some statistics


## Results:




