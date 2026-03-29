import cv2
import numpy as np

class PlateTracker:

    def __init__(self, memory_frames=10):
        self.tracks = {}
        self.memory_frames = memory_frames

    def update(self, ids, boxes):
        updated = {}

        for track_id, box in zip(ids, boxes):

            if track_id in self.tracks:
                prev = self.tracks[track_id]["box"]

                smooth_box = [
                    int(prev[i]*0.7 + box[i]*0.3)
                    for i in range(4)
                ]
            else:
                smooth_box = box

            updated[track_id] = {
                "box": smooth_box,
                "life": self.memory_frames
            }

        for track_id in list(self.tracks.keys()):

            if track_id not in updated:

                self.tracks[track_id]["life"] -= 1

                if self.tracks[track_id]["life"] > 0:
                    updated[track_id] = self.tracks[track_id]

        self.tracks = updated

        return [t["box"] for t in self.tracks.values()]


def pixelate(frame, box):

    h, w = frame.shape[:2]

    x1,y1,x2,y2 = box

    x1 = max(0, x1-12)
    y1 = max(0, y1-12)
    x2 = min(w, x2+12)
    y2 = min(h, y2+12)

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return frame

    small = cv2.resize(roi, (16,16), interpolation=cv2.INTER_LINEAR)

    pixelated = cv2.resize(
        small,
        (x2-x1, y2-y1),
        interpolation=cv2.INTER_NEAREST
    )

    frame[y1:y2, x1:x2] = pixelated

    return frame


def process_video(input_path, output_path, model):

    cap = cv2.VideoCapture(input_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (w,h)
    )

    # out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (w,h))

    tracker = PlateTracker()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.15,
            imgsz=960,
            verbose=False
        )

        ids = []
        boxes = []

        if results[0].boxes.id is not None:

            ids = results[0].boxes.id.cpu().numpy().astype(int)

            boxes = (
                results[0]
                .boxes
                .xyxy
                .cpu()
                .numpy()
                .astype(int)
            )

        stable_boxes = tracker.update(ids, boxes)

        for box in stable_boxes:
            frame = pixelate(frame, box)

        out.write(frame)

    cap.release()
    out.release()