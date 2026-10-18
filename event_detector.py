from pathlib import Path
from collections import defaultdict, deque
import csv
import math
import time

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# MATCHMIND AI - COMPLETE EVENT & PASS DETECTOR (FINAL)
# Includes:
# - Deterministic K-Means Team Clustering + Majority Voting
# - Person Track-ID Class Locking & Watermark/Teleport Filters
# - Late-Gap Carrier-to-Carrier Pass Resolution (Frames 266-625)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best.pt"
VIDEO_PATH = BASE_DIR / "videos" / "neymar.mp4"
OUTPUT_DIR = BASE_DIR / "output"
CUSTOM_TRACKER = BASE_DIR / "tracker.yaml"

VIDEO_OUT = OUTPUT_DIR / "matchmind_analysis.mp4"
TRACKING_OUT = OUTPUT_DIR / "tracking_data.csv"
EVENTS_OUT = OUTPUT_DIR / "pass_events.csv"
STATS_OUT = OUTPUT_DIR / "player_statistics.csv"

# Model & Tracker settings
IMAGE_SIZE = 1280
CONFIDENCE = 0.15
IOU_THRESHOLD = 0.50
TRACKER_CONFIG = str(CUSTOM_TRACKER) if CUSTOM_TRACKER.exists() else "botsort.yaml"

# Pass & Possession heuristics
POSSESSION_DISTANCE_RATIO = 0.045
MIN_POSSESSION_FRAMES = 2        # Consecutive frames required to confirm ball control
BALL_MAX_GAP_FRAMES = 10         # Interpolate ball across short detection dropouts
MIN_BALL_MOVEMENT_PX = 10.0      # Minimum ball movement to start a pass attempt
MIN_PASS_MOVEMENT_PX = 28.0      # Minimum total ball travel from passer to receiver
MAX_PASS_FLIGHT_SECONDS = 3.0    # Max visible flight window before timeout
MAX_GAP_PASS_SECONDS = 8.0       # Max carrier-to-carrier window across ball detection gaps
MIN_PASS_COOLDOWN_SECONDS = 0.5  # Cooldown between logged pass events

# Ball false-positive filters
MAX_BALL_BOX_SIZE_PX = 24.0          # Rejects player boots/kits misclassified as ball
MAX_BALL_SPEED_PX_PER_FRAME = 140.0  # Rejects 1-frame background/watermark teleports


def get_class_name(model, class_id):
    names = model.names
    if isinstance(names, dict):
        return str(names.get(class_id, f"class_{class_id}"))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return f"class_{class_id}"


def point_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def player_foot_point(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)


def extract_jersey_feature(frame, box):
    """Extract average non-grass HSV color from the upper torso of a player box."""
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    # Crop central upper torso (jersey region)
    cx1 = max(0, x1 + int(bw * 0.20))
    cx2 = min(w_img, x2 - int(bw * 0.20))
    cy1 = max(0, y1 + int(bh * 0.15))
    cy2 = min(h_img, y1 + int(bh * 0.55))

    if cx2 <= cx1 or cy2 <= cy1:
        return None

    crop = frame[cy1:cy2, cx1:cx2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Mask out green pitch pixels (Hue 32 to 88)
    grass_mask = cv2.inRange(hsv, (32, 35, 35), (88, 255, 255))
    non_grass = hsv[grass_mask == 0]

    if len(non_grass) < 10:
        return None

    return np.mean(non_grass, axis=0).astype(np.float32)


def assign_teams_deterministic(player_features):
    """
    Cluster tracked players into Team 0 and Team 1 using K-Means,
    sorting cluster centroids so Team 0 and Team 1 never flip labels.
    """
    if len(player_features) < 4:
        return {}

    track_ids = list(player_features.keys())
    samples = np.array([player_features[tid] for tid in track_ids], dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 1.0)
    _, labels, centers = cv2.kmeans(
        samples, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )

    # Sort centroids deterministically by Hue + Saturation + Value
    order = np.argsort(centers[:, 0] * 2.0 + centers[:, 1] + centers[:, 2] * 0.5)
    remap = {int(order[0]): 0, int(order[1]): 1}

    return {tid: remap[int(label[0])] for tid, label in zip(track_ids, labels)}


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("MATCHMIND AI: COMPLETE PLAYER TRACKING & PASS ENGINE")
    print("=" * 65)
    print(f"Using tracker config: {TRACKER_CONFIG}")

    model = YOLO(str(MODEL_PATH))
    print("Model classes:", model.names)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_writer = cv2.VideoWriter(
        str(VIDEO_OUT),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not video_writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create video: {VIDEO_OUT}")

    diagonal = math.hypot(width, height)
    possession_limit = max(35.0, diagonal * POSSESSION_DISTANCE_RATIO)

    # Per-player tracking metadata
    player_stats = defaultdict(lambda: {
        "team_id": None,
        "frames_seen": 0,
        "first_seen": None,
        "last_seen": None,
        "pass_attempt_candidates": 0,
        "completed_reception_candidates": 0,
        "interception_candidates": 0,
        "failed_pass_candidates": 0,
        "receiver_candidates": 0,
    })

    # Jersey feature buffer & cumulative majority-vote table per player
    jersey_history = defaultdict(lambda: deque(maxlen=25))
    team_votes = defaultdict(lambda: [0, 0])  # tid -> [votes_team_0, votes_team_1]
    team_map = {}

    # Track IDs ever seen as a person so they can never switch class to ball
    non_ball_track_ids = set()

    # Visual trails & HUD state
    ball_trail = deque(maxlen=15)
    recent_pass_visual = None

    # Ball interpolation & possession state
    last_known_ball = None        # {"center": (x, y), "frame": int}
    previous_ball_center = None
    candidate_holder_id = None
    candidate_holder_streak = 0
    confirmed_holder_id = None
    last_confirmed_holder = None  # {"id": int, "time": float, "frame": int, "ball_center": (x, y)}
    pending_pass = None
    last_pass_event_time = -999.0

    # Store events in memory so final majority-voted team_ids are applied cleanly
    logged_events = []

    tracking_file = open(TRACKING_OUT, "w", newline="", encoding="utf-8-sig")
    tracking_csv = csv.writer(tracking_file)
    tracking_csv.writerow([
        "frame", "timestamp_seconds",
        "class_id", "class_name", "confidence", "track_id",
        "x1", "y1", "x2", "y2", "center_x", "center_y",
    ])

    frame_number = 0

    def record_event(timestamp, frame, passer, receiver, is_direct, raw_outcome, evidence):
        logged_events.append({
            "event_id": len(logged_events) + 1,
            "timestamp_seconds": round(timestamp, 3),
            "frame": frame,
            "event_type": "pass_candidate",
            "passer_id": passer,
            "receiver_id": receiver,
            "is_direct": is_direct,
            "raw_outcome": raw_outcome,
            "evidence": evidence,
        })

    start_time = time.time()
    print(f"Video: {width}x{height}, {fps:.2f} FPS | Possession radius: {possession_limit:.1f}px")
    print("Processing frames...")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_number += 1
            timestamp = (frame_number - 1) / fps

            result = model.track(
                frame,
                persist=True,
                tracker=TRACKER_CONFIG,
                conf=CONFIDENCE,
                iou=IOU_THRESHOLD,
                imgsz=IMAGE_SIZE,
                verbose=False,
            )[0]

            players = []
            raw_balls = []
            others = []

            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy_list = boxes.xyxy.cpu().tolist()
                class_list = boxes.cls.cpu().tolist()
                conf_list = boxes.conf.cpu().tolist()
                ids = (
                    boxes.id.int().cpu().tolist()
                    if boxes.id is not None
                    else [None] * len(boxes)
                )

                for coords, class_value, conf, track_id in zip(
                    xyxy_list, class_list, conf_list, ids
                ):
                    x1, y1, x2, y2 = map(float, coords)
                    bw, bh = (x2 - x1), (y2 - y1)
                    class_id = int(class_value)
                    class_name = get_class_name(model, class_id)
                    center = box_center((x1, y1, x2, y2))
                    normalized = class_name.strip().lower()
                    tid = int(track_id) if track_id is not None else None

                    # Lock any track_id that has person class or person height
                    if normalized in ("player", "referee", "goalkeeper") or bh > MAX_BALL_BOX_SIZE_PX:
                        if tid is not None:
                            non_ball_track_ids.add(tid)

                    item = {
                        "class_id": class_id,
                        "class_name": class_name,
                        "normalized": normalized,
                        "confidence": float(conf),
                        "track_id": tid,
                        "box": (x1, y1, x2, y2),
                        "center": center,
                    }

                    if normalized == "player" and tid is not None:
                        players.append(item)

                        stats = player_stats[tid]
                        stats["frames_seen"] += 1
                        if stats["first_seen"] is None:
                            stats["first_seen"] = round(timestamp, 3)
                        stats["last_seen"] = round(timestamp, 3)

                        if frame_number % 3 == 0:
                            feat = extract_jersey_feature(frame, item["box"])
                            if feat is not None:
                                jersey_history[tid].append(feat)

                    elif normalized == "ball":
                        raw_balls.append(item)
                    else:
                        others.append(item)

            # ----------------------------------------------------
            # 1. FILTER FALSE BALL DETECTIONS
            # ----------------------------------------------------
            valid_balls = []
            for b in raw_balls:
                bx1, by1, bx2, by2 = b["box"]
                bw, bh = (bx2 - bx1), (by2 - by1)
                cx, cy = b["center"]
                tid = b["track_id"]

                # Reject if track_id was ever a player/referee/goalkeeper
                if tid is not None and tid in non_ball_track_ids:
                    continue
                # Reject oversized boxes (boots/kits)
                if bw > MAX_BALL_BOX_SIZE_PX or bh > MAX_BALL_BOX_SIZE_PX:
                    continue
                # Reject bottom-right SCOUTINGFEED watermark zone
                if cx > width * 0.74 and cy > height * 0.80:
                    continue
                # Reject impossible short-frame teleportation jumps
                if last_known_ball is not None:
                    gap = max(1, frame_number - last_known_ball["frame"])
                    if gap <= 4:
                        speed = point_distance((cx, cy), last_known_ball["center"]) / gap
                        if speed > MAX_BALL_SPEED_PX_PER_FRAME:
                            continue

                valid_balls.append(b)

            # Write clean detections to tracking_data.csv
            for det in players + others + valid_balls:
                x1, y1, x2, y2 = det["box"]
                cx, cy = det["center"]
                tracking_csv.writerow([
                    frame_number,
                    round(timestamp, 3),
                    det["class_id"],
                    det["class_name"],
                    round(det["confidence"], 4),
                    det["track_id"] if det["track_id"] is not None else "",
                    round(x1, 2), round(y1, 2),
                    round(x2, 2), round(y2, 2),
                    round(cx, 2), round(cy, 2),
                ])

            # ----------------------------------------------------
            # 2. STABILIZED TEAM CLUSTERING WITH MAJORITY VOTE
            # ----------------------------------------------------
            if frame_number % 10 == 1 and len(jersey_history) >= 4:
                avg_features = {
                    tid: np.mean(feats, axis=0)
                    for tid, feats in jersey_history.items()
                    if len(feats) > 0
                }
                instant_teams = assign_teams_deterministic(avg_features)
                for tid, t_id in instant_teams.items():
                    team_votes[tid][t_id] += 1
                    majority_team = 0 if team_votes[tid][0] >= team_votes[tid][1] else 1
                    team_map[tid] = majority_team
                    if tid in player_stats:
                        player_stats[tid]["team_id"] = majority_team

            # ----------------------------------------------------
            # 3. BALL SELECTION & SHORT-GAP INTERPOLATION
            # ----------------------------------------------------
            best_ball = max(valid_balls, key=lambda d: d["confidence"]) if valid_balls else None
            active_ball_center = None
            ball_is_interpolated = False

            if best_ball is not None:
                active_ball_center = best_ball["center"]
                if (
                    last_known_ball is not None
                    and 1 < (frame_number - last_known_ball["frame"]) <= BALL_MAX_GAP_FRAMES
                ):
                    prev_c = last_known_ball["center"]
                    active_ball_center = (
                        (prev_c[0] + active_ball_center[0]) / 2.0,
                        (prev_c[1] + active_ball_center[1]) / 2.0,
                    )
                last_known_ball = {"center": best_ball["center"], "frame": frame_number}
                ball_trail.append((int(best_ball["center"][0]), int(best_ball["center"][1])))
            elif (
                last_known_ball is not None
                and (frame_number - last_known_ball["frame"]) <= BALL_MAX_GAP_FRAMES
            ):
                active_ball_center = last_known_ball["center"]
                ball_is_interpolated = True

            # ----------------------------------------------------
            # 4. FIND & CONFIRM BALL CARRIER
            # ----------------------------------------------------
            raw_holder_id = None

            if active_ball_center is not None and players:
                candidates = []
                for player in players:
                    foot = player_foot_point(player["box"])
                    d = point_distance(active_ball_center, foot)
                    candidates.append((d, player))

                nearest_distance, nearest_player = min(candidates, key=lambda item: item[0])
                if nearest_distance <= possession_limit:
                    raw_holder_id = int(nearest_player["track_id"])

            if raw_holder_id is not None and raw_holder_id == candidate_holder_id:
                candidate_holder_streak += 1
            else:
                candidate_holder_id = raw_holder_id
                candidate_holder_streak = 1 if raw_holder_id is not None else 0

            confirmed_holder_id = (
                candidate_holder_id
                if candidate_holder_streak >= MIN_POSSESSION_FRAMES
                else None
            )

            ball_displacement = (
                point_distance(active_ball_center, previous_ball_center)
                if (active_ball_center is not None and previous_ball_center is not None)
                else 0.0
            )

            # ----------------------------------------------------
            # 5. PASS & INTERCEPTION STATE MACHINE (FINAL)
            # ----------------------------------------------------
            # Start an open-flight pending pass only when NO player is touching the ball
            if (
                pending_pass is None
                and last_confirmed_holder is not None
                and confirmed_holder_id is None
                and raw_holder_id is None
                and active_ball_center is not None
                and (timestamp - last_confirmed_holder["time"]) <= MAX_PASS_FLIGHT_SECONDS
            ):
                dist_from_holder = point_distance(
                    active_ball_center, last_confirmed_holder["ball_center"]
                )
                if (
                    (ball_displacement >= MIN_BALL_MOVEMENT_PX or dist_from_holder >= MIN_PASS_MOVEMENT_PX)
                    and (timestamp - last_pass_event_time) >= MIN_PASS_COOLDOWN_SECONDS
                ):
                    pending_pass = {
                        "passer_id": last_confirmed_holder["id"],
                        "start_time": timestamp,
                        "holder_time": last_confirmed_holder["time"],
                        "start_frame": last_confirmed_holder["frame"],
                        "start_ball": last_confirmed_holder["ball_center"],
                    }

            if confirmed_holder_id is not None and active_ball_center is not None:
                # Case A: Pending pass received by a different player
                if (
                    pending_pass is not None
                    and confirmed_holder_id != pending_pass["passer_id"]
                ):
                    passer_id = pending_pass["passer_id"]
                    receiver_id = confirmed_holder_id
                    elapsed = timestamp - pending_pass["holder_time"]
                    travel_px = point_distance(active_ball_center, pending_pass["start_ball"])

                    if travel_px >= MIN_PASS_MOVEMENT_PX:
                        same_team = (
                            passer_id in team_map
                            and receiver_id in team_map
                            and team_map[passer_id] == team_map[receiver_id]
                        )
                        vis_color = (0, 255, 0) if same_team else (0, 0, 255)
                        vis_label = (
                            f"PASS: #{passer_id} -> #{receiver_id}"
                            if same_team
                            else f"TURNOVER: #{passer_id} -> #{receiver_id}"
                        )

                        record_event(
                            timestamp, frame_number, passer_id, receiver_id,
                            is_direct=False, raw_outcome="received",
                            evidence=f"Flight={elapsed:.2f}s, Travel={travel_px:.1f}px"
                        )
                        recent_pass_visual = {
                            "p1": (int(pending_pass["start_ball"][0]), int(pending_pass["start_ball"][1])),
                            "p2": (int(active_ball_center[0]), int(active_ball_center[1])),
                            "until": timestamp + 1.2,
                            "text": vis_label,
                            "color": vis_color,
                        }
                        last_pass_event_time = timestamp

                    pending_pass = None

                # Case B: Carrier-to-carrier transfer (handles both fast transfers & occluded ball gaps)
                elif (
                    pending_pass is None
                    and last_confirmed_holder is not None
                    and confirmed_holder_id != last_confirmed_holder["id"]
                    and (timestamp - last_confirmed_holder["time"]) <= MAX_GAP_PASS_SECONDS
                    and (timestamp - last_pass_event_time) >= MIN_PASS_COOLDOWN_SECONDS
                ):
                    passer_id = last_confirmed_holder["id"]
                    receiver_id = confirmed_holder_id
                    elapsed = timestamp - last_confirmed_holder["time"]
                    travel_px = point_distance(
                        active_ball_center, last_confirmed_holder["ball_center"]
                    )
                    if travel_px >= MIN_PASS_MOVEMENT_PX:
                        same_team = (
                            passer_id in team_map
                            and receiver_id in team_map
                            and team_map[passer_id] == team_map[receiver_id]
                        )
                        vis_color = (0, 255, 0) if same_team else (0, 0, 255)
                        vis_label = (
                            f"PASS: #{passer_id} -> #{receiver_id}"
                            if same_team
                            else f"TURNOVER: #{passer_id} -> #{receiver_id}"
                        )
                        is_direct_flag = elapsed <= 0.6

                        record_event(
                            timestamp, frame_number, passer_id, receiver_id,
                            is_direct=is_direct_flag, raw_outcome="received",
                            evidence=f"Flight={elapsed:.2f}s, Travel={travel_px:.1f}px"
                        )
                        recent_pass_visual = {
                            "p1": (int(last_confirmed_holder["ball_center"][0]), int(last_confirmed_holder["ball_center"][1])),
                            "p2": (int(active_ball_center[0]), int(active_ball_center[1])),
                            "until": timestamp + 1.2,
                            "text": vis_label,
                            "color": vis_color,
                        }
                        last_pass_event_time = timestamp

                # Same player regained ball -> cancel false pass start
                elif pending_pass is not None and confirmed_holder_id == pending_pass["passer_id"]:
                    pending_pass = None

                last_confirmed_holder = {
                    "id": confirmed_holder_id,
                    "time": timestamp,
                    "frame": frame_number,
                    "ball_center": active_ball_center,
                }

            # Check if pending pass timed out
            if (
                pending_pass is not None
                and raw_holder_id is None
                and (timestamp - pending_pass["start_time"]) > MAX_PASS_FLIGHT_SECONDS
            ):
                # Only log unreceived if the ball is actually visible and loose;
                # if the ball was simply undetected by YOLO, clear pending_pass so Case B links the next carrier
                if active_ball_center is not None:
                    record_event(
                        timestamp, frame_number, pending_pass["passer_id"], None,
                        is_direct=False, raw_outcome="unreceived_ball_candidate",
                        evidence=f"No receiver within {MAX_PASS_FLIGHT_SECONDS:.1f}s window."
                    )
                    last_pass_event_time = timestamp
                pending_pass = None

            # ----------------------------------------------------
            # 6. CLEAN BROADCAST VISUALIZATION
            # ----------------------------------------------------
            annotated = frame.copy()

            for i in range(1, len(ball_trail)):
                cv2.line(annotated, ball_trail[i - 1], ball_trail[i], (0, 165, 255), 2, cv2.LINE_AA)

            if recent_pass_visual is not None and timestamp <= recent_pass_visual["until"]:
                cv2.arrowedLine(
                    annotated,
                    recent_pass_visual["p1"],
                    recent_pass_visual["p2"],
                    recent_pass_visual["color"],
                    3,
                    cv2.LINE_AA,
                    tipLength=0.04,
                )

            for player in players:
                x1, y1, x2, y2 = map(int, player["box"])
                tid = int(player["track_id"])
                t_id = team_map.get(tid, None)

                if tid == confirmed_holder_id:
                    color = (0, 255, 255)
                    thickness = 3
                elif t_id == 0:
                    color = (255, 140, 0)
                    thickness = 2
                elif t_id == 1:
                    color = (60, 220, 60)
                    thickness = 2
                else:
                    color = (200, 200, 200)
                    thickness = 1

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(
                    annotated,
                    f"#{tid}",
                    (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            for other in others:
                x1, y1, x2, y2 = map(int, other["box"])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (180, 180, 0), 1)

            if active_ball_center is not None:
                bx, by = int(active_ball_center[0]), int(active_ball_center[1])
                ball_color = (0, 140, 255) if ball_is_interpolated else (0, 0, 255)
                cv2.circle(annotated, (bx, by), 7, ball_color, 2, cv2.LINE_AA)

            cv2.rectangle(annotated, (12, 10), (470, 115), (20, 20, 20), -1)
            cv2.putText(
                annotated, "MATCHMIND AI | PASS & EVENT ENGINE",
                (22, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA
            )
            cv2.putText(
                annotated, f"Time: {timestamp:.2f}s | Frame: {frame_number} | Events: {len(logged_events)}",
                (22, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA
            )

            holder_display = (
                f"Ball Carrier: ID #{confirmed_holder_id} (Team {team_map.get(confirmed_holder_id, '?')})"
                if confirmed_holder_id is not None
                else (
                    f"In Flight from ID #{pending_pass['passer_id']}..."
                    if pending_pass is not None
                    else f"Last Carrier: ID #{last_confirmed_holder['id']}"
                    if last_confirmed_holder is not None
                    else "Ball Carrier: searching..."
                )
            )
            cv2.putText(
                annotated, holder_display,
                (22, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 180), 2, cv2.LINE_AA
            )

            if recent_pass_visual is not None and timestamp <= recent_pass_visual["until"]:
                cv2.putText(
                    annotated, recent_pass_visual["text"],
                    (22, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.52, recent_pass_visual["color"], 2, cv2.LINE_AA
                )

            video_writer.write(annotated)

            if active_ball_center is not None:
                previous_ball_center = active_ball_center

            if frame_number % 100 == 0:
                if total_frames > 0:
                    percent = frame_number / total_frames * 100
                    print(f"Processed {frame_number}/{total_frames} frames ({percent:.1f}%) | Events: {len(logged_events)}")
                else:
                    print(f"Processed {frame_number} frames | Events: {len(logged_events)}")

        if pending_pass is not None and active_ball_center is not None:
            record_event(
                timestamp if frame_number else 0,
                frame_number,
                pending_pass["passer_id"],
                None,
                is_direct=False,
                raw_outcome="outcome_unknown_end_of_video",
                evidence="Video ended while pass was still in flight.",
            )

    finally:
        cap.release()
        video_writer.release()
        tracking_file.close()

    # --------------------------------------------------------
    # 7. FINAL TEAM RECONCILIATION BEFORE WRITING CSVs
    # --------------------------------------------------------
    if len(jersey_history) >= 4:
        final_features = {
            tid: np.mean(feats, axis=0)
            for tid, feats in jersey_history.items()
            if len(feats) > 0
        }
        final_clusters = assign_teams_deterministic(final_features)
        for tid, c_id in final_clusters.items():
            team_votes[tid][c_id] += 2
            team_map[tid] = 0 if team_votes[tid][0] >= team_votes[tid][1] else 1

    for tid in player_stats:
        if tid in team_map:
            player_stats[tid]["team_id"] = team_map[tid]

    # Write reconciled pass_events.csv and accumulate final player stats
    with open(EVENTS_OUT, "w", newline="", encoding="utf-8-sig") as events_file:
        events_csv = csv.writer(events_file)
        events_csv.writerow([
            "event_id", "timestamp_seconds", "frame",
            "event_type", "passer_id", "receiver_id",
            "passer_team", "receiver_team",
            "outcome_candidate", "evidence",
        ])

        for ev in logged_events:
            passer_id = ev["passer_id"]
            receiver_id = ev["receiver_id"]
            p_team = team_map.get(passer_id, None)
            r_team = team_map.get(receiver_id, None) if receiver_id is not None else None

            if passer_id is not None:
                player_stats[passer_id]["pass_attempt_candidates"] += 1

            if receiver_id is not None and ev["raw_outcome"] == "received":
                player_stats[receiver_id]["receiver_candidates"] += 1
                if p_team is not None and r_team is not None:
                    if p_team == r_team:
                        outcome = (
                            "direct_completed_pass_same_team"
                            if ev["is_direct"]
                            else "completed_pass_same_team"
                        )
                        player_stats[passer_id]["completed_reception_candidates"] += 1
                    else:
                        outcome = "interception_or_turnover_candidate"
                        player_stats[passer_id]["interception_candidates"] += 1
                        player_stats[passer_id]["failed_pass_candidates"] += 1
                else:
                    outcome = "completed_reception_team_unknown"
                    player_stats[passer_id]["completed_reception_candidates"] += 1
            else:
                outcome = ev["raw_outcome"]
                if outcome == "unreceived_ball_candidate" and passer_id is not None:
                    player_stats[passer_id]["failed_pass_candidates"] += 1

            events_csv.writerow([
                ev["event_id"],
                ev["timestamp_seconds"],
                ev["frame"],
                ev["event_type"],
                passer_id if passer_id is not None else "",
                receiver_id if receiver_id is not None else "",
                p_team if p_team is not None else "",
                r_team if r_team is not None else "",
                outcome,
                ev["evidence"],
            ])

    # Write reconciled player_statistics.csv
    with open(STATS_OUT, "w", newline="", encoding="utf-8-sig") as stats_file:
        stats_csv = csv.writer(stats_file)
        stats_csv.writerow([
            "player_track_id",
            "team_id",
            "frames_seen",
            "first_seen_seconds",
            "last_seen_seconds",
            "pass_attempt_candidates",
            "completed_reception_candidates",
            "interception_candidates",
            "failed_pass_candidates",
            "receiver_candidates",
            "candidate_completion_rate_pct",
        ])

        for player_id in sorted(player_stats):
            s = player_stats[player_id]
            attempts = s["completed_reception_candidates"] + s["failed_pass_candidates"]
            rate = (
                100.0 * s["completed_reception_candidates"] / attempts
                if attempts > 0 else None
            )

            stats_csv.writerow([
                player_id,
                s["team_id"] if s["team_id"] is not None else "",
                s["frames_seen"],
                s["first_seen"],
                s["last_seen"],
                s["pass_attempt_candidates"],
                s["completed_reception_candidates"],
                s["interception_candidates"],
                s["failed_pass_candidates"],
                s["receiver_candidates"],
                round(rate, 2) if rate is not None else "",
            ])

    elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    print("MATCHMIND AI ANALYSIS FINISHED")
    print("=" * 65)
    print(f"Frames processed: {frame_number} | Clean Pass Events Logged: {len(logged_events)}")
    print(f"Elapsed time: {elapsed / 60:.2f} minutes")
    print("Video:", VIDEO_OUT)
    print("Tracking data:", TRACKING_OUT)
    print("Pass event log:", EVENTS_OUT)
    print("Player statistics:", STATS_OUT)


if __name__ == "__main__":
    main()