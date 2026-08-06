# motion-detector (model repo)

YOLOv8-based real-time moving object **detection + tracking** core.
Ye repo sirf detection logic hai — koi frontend/UI nahi. Frontend
alag repo (`yolo-frontend`) me hai, jo isko dependency ki tarah use
karta hai.

## Structure

```
yolo-model/
├── motion_detector/
│   ├── __init__.py      # exports: ObjectDetector, CentroidTracker, config
│   ├── config.py         # camera + model + motion thresholds
│   ├── tracker.py        # CentroidTracker (motion decide karta hai)
│   └── detection.py       # ObjectDetector (YOLO + tracker combine)
├── train_model.py         # custom dataset par fine-tune karne ka script
├── setup.py               # pip-installable package banata hai
└── requirements.txt
```

## Local install / testing

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

```python
from motion_detector import ObjectDetector
import cv2

detector = ObjectDetector()
cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    annotated_frame, stats = detector.process_frame(frame)
    cv2.imshow("Detection", annotated_frame)
    if cv2.waitKey(1) == ord('q'):
        break
```

## Custom model training

Apne objects par model train karna ho:

```bash
python train_model.py
```

Dataset format aur instructions `train_model.py` ke docstring me hain.
Training ke baad best weights ka path `motion_detector/config.py` me
`MODEL_NAME` me daal do.

## GitHub par upload

```bash
cd yolo-model
git init
git add .
git commit -m "YOLOv8 motion detection core (model + tracker)"
git branch -M main
git remote add origin https://github.com/<your-username>/yolo-model.git
git push -u origin main
```

Is repo ko dusre projects (jaise `yolo-frontend`) me directly GitHub
se install kiya ja sakta hai:

```bash
pip install git+https://github.com/<your-username>/yolo-model.git
```
