
from pathlib import Path
from collections import defaultdict, deque

import cv2
from ultralytics import YOLO

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"
VIDEO_PATH = BASE_DIR / "videos" / "neymar.mp4"
OUTPUT_DIR = BASE_DIR / "output"

# Visual settings
TRAIL_LENGTH = 12
CONFIDENCE = 0.25
BOX_COLOR = (0, 255, 0)       # Green
TRAIL_COLOR = (0, 165, 255)   # Orange
TEXT_COLOR = (255, 255, 255)  # White


def main():
    # Check that required files exist
    for path in (MODEL_PATH, VIDEO_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load your trained model
    model = YOLO(str(MODEL_PATH))

    # Keep a short trail for each tracked object
    trails = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = OUTPUT_DIR / "neymar_tracked_clean.mp4"

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video.")

    frame_number = 0

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=CONFIDENCE,
                iou=0.50,
                verbose=False
            )

            result = results[0]
            annotated = frame.copy()

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().tolist()

                if result.boxes.id is not None:
                    track_ids = (
                        result.boxes.id.int().cpu().tolist()
                    )
                else:
                    track_ids = [None] * len(boxes)

                for box, confidence, track_id in zip(
                    boxes, confidences, track_ids
                ):
                    x1, y1, x2, y2 = map(int, box)

                    # Draw a box around the detected object
                    cv2.rectangle(
                        annotated,
                        (x1, y1),
                        (x2, y2),
                        BOX_COLOR,
                        2
                    )

                    if track_id is not None:
                        center = (
                            (x1 + x2) // 2,
                            (y1 + y2) // 2
                        )
                        trails[track_id].append(center)

                        label = f"ID {track_id} | {confidence:.2f}"
                    else:
                        label = f"Detected | {confidence:.2f}"

                    # Add a solid background behind the label
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.55
                    thickness = 2

                    (text_width, text_height), baseline = (
                        cv2.getTextSize(
                            label, font, font_scale, thickness
                        )
                    )

                    label_y = max(y1, text_height + 8)

                    cv2.rectangle(
                        annotated,
                        (x1, label_y - text_height - 8),
                        (x1 + text_width + 8, label_y + baseline - 4),
                        BOX_COLOR,
                        -1
                    )

                    cv2.putText(
                        annotated,
                        label,
                        (x1 + 4, label_y - 4),
                        font,
                        font_scale,
                        TEXT_COLOR,
                        thickness,
                        cv2.LINE_AA
                    )

            # Draw short trails, without cluttering the entire pitch
            for points in trails.values():
                for i in range(1, len(points)):
                    cv2.line(
                        annotated,
                        points[i - 1],
                        points[i],
                        TRAIL_COLOR,
                        2,
                        cv2.LINE_AA
                    )

            frame_number += 1

            # Small frame counter
            cv2.putText(
                annotated,
                f"Frame: {frame_number}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                TEXT_COLOR,
                2,
                cv2.LINE_AA
            )

            writer.write(annotated)

            if frame_number % 100 == 0:
                print(f"Processed {frame_number} frames...")

    finally:
        cap.release()
        writer.release()

    print("\nTracking complete!")
    print(f"Output video: {output_path}")


if __name__ == "__main__":
    main()
