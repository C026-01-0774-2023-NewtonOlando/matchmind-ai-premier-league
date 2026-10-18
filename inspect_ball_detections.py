
import os
import cv2
import pandas as pd
from PIL import Image, ImageDraw

VIDEO_PATH = r"videos\neymar.mp4"
CSV_PATH = r"output\tracking_data.csv"
OUTPUT_DIR = r"output\ball_audit"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

# Frames where multiple ball detections were recorded
ball = df[df["class_name"] == "ball"].copy()
frames = (
    ball.groupby("frame")
    .size()
    .loc[lambda counts: counts > 1]
    .index.astype(int)
    .tolist()
)

if not frames:
    raise RuntimeError("No frames with multiple ball detections were found.")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

thumbnails = []

for frame_number in frames:
    # CSV frame numbers start at 1; OpenCV frame positions start at 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    success, frame = cap.read()

    if not success:
        print(f"Could not read frame {frame_number}")
        continue

    frame_rows = ball[ball["frame"] == frame_number]

    for _, row in frame_rows.iterrows():
        x1 = int(row["x1"])
        y1 = int(row["y1"])
        x2 = int(row["x2"])
        y2 = int(row["y2"])

        confidence = float(row["confidence"])
        track_id = row["track_id"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        label = f"BALL ID:{track_id} conf:{confidence:.2f}"
        cv2.putText(
            frame, label, (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (0, 0, 255), 2
        )

    cv2.putText(
        frame, f"FRAME {frame_number}",
        (25, 40), cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (0, 255, 255), 2
    )

    frame_path = os.path.join(
        OUTPUT_DIR, f"frame_{frame_number:04d}.jpg"
    )

    cv2.imwrite(frame_path, frame)

    # Resize for a contact sheet while preserving aspect ratio
    height, width = frame.shape[:2]
    thumb_width = 480
    thumb_height = int(height * thumb_width / width)

    resized = cv2.resize(frame, (thumb_width, thumb_height))
    thumbnails.append(
        (frame_number, Image.fromarray(cv2.cvtColor(
            resized, cv2.COLOR_BGR2RGB
        )))
    )

cap.release()

if thumbnails:
    columns = 3
    rows = (len(thumbnails) + columns - 1) // columns
    thumb_w, thumb_h = thumbnails[0][1].size

    sheet = Image.new(
        "RGB", (columns * thumb_w, rows * (thumb_h + 30)), "white"
    )
    draw = ImageDraw.Draw(sheet)

    for index, (number, image) in enumerate(thumbnails):
        x = (index % columns) * thumb_w
        y = (index // columns) * (thumb_h + 30)

        sheet.paste(image, (x, y + 30))
        draw.text((x + 8, y + 7), f"Frame {number}", fill="black")

    sheet_path = os.path.join(OUTPUT_DIR, "ball_contact_sheet.jpg")
    sheet.save(sheet_path)

    print(f"\nFrames inspected: {len(thumbnails)}")
    print(f"Contact sheet: {sheet_path}")
    print(f"Individual frames: {OUTPUT_DIR}")
