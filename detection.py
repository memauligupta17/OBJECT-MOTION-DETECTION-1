"""
Detection core — YOLO model load karta hai aur har frame ko process
karke annotated frame + stats deta hai. Koi bhi frontend (Streamlit,
Flask, ya kuch aur) isi class ko import karke use kar sakta hai.
"""

import cv2
from ultralytics import YOLO

from . import config
from .tracker import CentroidTracker

COLOR_MOVING = (0, 0, 255)
COLOR_STATIC = (0, 200, 0)
COLOR_PENDING = (0, 200, 200)


class ObjectDetector:
    def __init__(self, model_name=None):
        self.model = YOLO(model_name or config.MODEL_NAME)
        self.tracker = CentroidTracker(
            max_disappeared=config.MAX_DISAPPEARED,
            max_distance=config.MAX_DISTANCE,
            history_len=config.TRACK_HISTORY_LEN,
            motion_threshold=config.MOTION_THRESHOLD,
        )

    def process_frame(self, frame):
        """
        Ek frame leta hai, YOLO detection + motion tracking chalata hai,
        aur (annotated_frame, stats_dict) return karta hai.
        """
        results = self.model.predict(
            frame,
            conf=config.CONF_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
            detections.append((centroid, label, (int(x1), int(y1), int(x2), int(y2))))

        tracked = self.tracker.update(detections)

        moving_count = 0
        static_count = 0

        for oid, info in tracked.items():
            x1, y1, x2, y2 = info["box"]
            status = info["status"]
            label = info["label"]

            if status == "Moving":
                color = COLOR_MOVING
                moving_count += 1
            elif status == "Static":
                color = COLOR_STATIC
                static_count += 1
            else:
                color = COLOR_PENDING

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, f"ID {oid} {label}: {status}", (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )

        stats = {
            "total_objects": len(tracked),
            "moving": moving_count,
            "static": static_count,
        }
        return frame, stats
