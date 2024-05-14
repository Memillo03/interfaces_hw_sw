"""Program to follow people using Kalman filter."""

import cv2
import numpy as np
import torch
from sort import Sort

model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)


def person_tracking():
    """Function to follow people in video."""
    cap = cv2.VideoCapture(r"test.mp4")
    mot_tracker = Sort()
    frameID = 0
    skip_frame = 10000

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frameID % skip_frame == 0:
            results = model(frame)
            results = results.xyxy[0].numpy()
            person_detections = results[results[:, 5] == 0]

        dets = []
        for *xyxy, conf, cls in person_detections:
            x1, y1, x2, y2 = map(int, xyxy)
            dets.append([x1, y1, x2, y2, conf])
        dets = np.array(dets)

        trackers = mot_tracker.update(dets)

        for d in trackers:
            x1, y1, x2, y2, track_id = map(int, d[:5])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                frame,
                str(track_id),
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 255, 0),
                thickness=2,
            )

        cv2.imshow("view", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    person_tracking()
