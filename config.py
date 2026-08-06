"""
Central configuration — camera settings, model settings, aur
detection/motion thresholds. Kahin bhi settings change karni ho
to sirf yahi file edit karo.
"""

# ---- Camera Config ----
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# ---- Model Config ----
MODEL_NAME = "yolov8n.pt"    # custom trained model ho to uska path daalo
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.45

# ---- Motion Detection Config ----
MOTION_THRESHOLD = 8
TRACK_HISTORY_LEN = 10
MAX_DISAPPEARED = 20
MAX_DISTANCE = 60
