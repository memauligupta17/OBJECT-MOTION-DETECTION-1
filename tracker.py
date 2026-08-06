"""
Simple centroid-based multi-object tracker.

Har detected box ko ek unique ID assign hota hai, aur uske centroid
ki history rakhi jaati hai. Agar centroid frame-to-frame kaafi move
ho raha hai (threshold se zyada), to object "Moving" mark hota hai,
warna "Static".
"""

from collections import OrderedDict, deque
import numpy as np


class CentroidTracker:
    def __init__(self, max_disappeared=20, max_distance=60,
                 history_len=10, motion_threshold=8):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.history = OrderedDict()
        self.labels = OrderedDict()

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.history_len = history_len
        self.motion_threshold = motion_threshold

    def register(self, centroid, label):
        oid = self.next_object_id
        self.objects[oid] = centroid
        self.disappeared[oid] = 0
        self.history[oid] = deque(maxlen=self.history_len)
        self.history[oid].append(centroid)
        self.labels[oid] = label
        self.next_object_id += 1
        return oid

    def deregister(self, oid):
        del self.objects[oid]
        del self.disappeared[oid]
        del self.history[oid]
        del self.labels[oid]

    def update(self, detections):
        if len(detections) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return {}

        input_centroids = [d[0] for d in detections]
        input_labels = [d[1] for d in detections]
        input_boxes = [d[2] for d in detections]

        if len(self.objects) == 0:
            ids = []
            for c, l in zip(input_centroids, input_labels):
                ids.append(self.register(c, l))
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = np.linalg.norm(
                np.array(object_centroids)[:, np.newaxis] - np.array(input_centroids),
                axis=2
            )

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            ids = [None] * len(input_centroids)

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue

                oid = object_ids[row]
                self.objects[oid] = input_centroids[col]
                self.history[oid].append(input_centroids[col])
                self.labels[oid] = input_labels[col]
                self.disappeared[oid] = 0
                ids[col] = oid

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(D.shape[1])) - used_cols

            for row in unused_rows:
                oid = object_ids[row]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

            for col in unused_cols:
                oid = self.register(input_centroids[col], input_labels[col])
                ids[col] = oid

        results = {}
        for i, oid in enumerate(ids):
            if oid is None:
                continue
            status = self._motion_status(oid)
            results[oid] = {
                "centroid": input_centroids[i],
                "label": input_labels[i],
                "box": input_boxes[i],
                "status": status,
            }
        return results

    def _motion_status(self, oid):
        hist = self.history[oid]
        if len(hist) < 2:
            return "Detecting..."

        total_disp = 0.0
        for i in range(1, len(hist)):
            p1, p2 = np.array(hist[i - 1]), np.array(hist[i])
            total_disp += np.linalg.norm(p2 - p1)

        avg_disp = total_disp / (len(hist) - 1)
        return "Moving" if avg_disp > self.motion_threshold else "Static"
