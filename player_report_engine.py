from pathlib import Path
import json
import math
import re
import numpy as np
import pandas as pd

# ============================================================
# MATCHMIND AI - STEP 1: PLAYER DOSSIER & MATCH INTELLIGENCE ENGINE
# Generates Dribbles, Carry Distance, Speed, Pass Difficulty (xP),
# and Control-vs-Chaos Explainability Metrics (Hackathon Section 4.1)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

TRACKING_CSV = OUTPUT_DIR / "tracking_data.csv"
PASS_CSV = OUTPUT_DIR / "pass_events.csv"
STATS_CSV = OUTPUT_DIR / "player_statistics.csv"

DOSSIERS_OUT = OUTPUT_DIR / "player_dossiers_enriched.csv"
PASS_ENRICHED_OUT = OUTPUT_DIR / "pass_events_enriched.csv"
TIMELINE_OUT = OUTPUT_DIR / "control_chaos_timeline.csv"
INTEL_JSON_OUT = OUTPUT_DIR / "match_intelligence_payload.json"

FPS = 24.0
WIDTH, HEIGHT = 1920, 1080
POSSESSION_LIMIT_PX = max(35.0, math.hypot(WIDTH, HEIGHT) * 0.045)
PX_TO_METERS = 105.0 / 1920.0  # Calibrated pitch scale (meters per pixel)

SYNTHETIC_ROSTER = {
    12: "M. Silva (#12 - CM)",
    5: "J. Olando (#5 - CDM)",
    13: "K. Sterling (#13 - RW)",
    4: "D. Brooks (#4 - LCM)",
    14: "A. Diaz (#14 - CAM)",
    10: "L. Vance (#10 - CF)",
    6: "R. Sterling-Hayes (#6 - LB)",
    8: "T. Partey-Cole (#8 - CM)",
    28: "C. Gallagher-West (#28 - OPP)",
    1: "E. Martinez-Sim (#1 - CB)",
    2: "W. Saliba-Sim (#2 - RCB)",
    3: "G. Magalhaes-Sim (#3 - LCB)",
    9: "E. Haaland-Sim (#9 - ST)",
}


def main():
    for path in (TRACKING_CSV, PASS_CSV, STATS_CSV):
        if not path.exists():
            raise FileNotFoundError(f"Required input file not found: {path}")

    track_df = pd.read_csv(TRACKING_CSV)
    pass_df = pd.read_csv(PASS_CSV)
    stats_df = pd.read_csv(STATS_CSV)

    # 1. Player kinematics: step distance, speed (km/h), sprint thresholds
    player_rows = (
        track_df[track_df["class_name"] == "player"]
        .sort_values(["track_id", "frame"])
        .copy()
    )
    player_rows["foot_x"] = (player_rows["x1"] + player_rows["x2"]) / 2.0
    player_rows["foot_y"] = player_rows["y2"]

    player_rows["dt"] = player_rows.groupby("track_id")["timestamp_seconds"].diff()
    player_rows["dx"] = player_rows.groupby("track_id")["foot_x"].diff()
    player_rows["dy"] = player_rows.groupby("track_id")["foot_y"].diff()
    player_rows["step_px"] = np.hypot(player_rows["dx"], player_rows["dy"])

    valid_step = (
        (player_rows["dt"] > 0)
        & (player_rows["dt"] <= 0.5)
        & (player_rows["step_px"] <= 35.0)
    )
    player_rows["clean_step_px"] = np.where(valid_step, player_rows["step_px"], 0.0)
    player_rows["speed_px_s"] = np.where(
        valid_step, player_rows["step_px"] / player_rows["dt"], 0.0
    )
    player_rows["speed_px_s_smooth"] = player_rows.groupby("track_id")[
        "speed_px_s"
    ].transform(lambda s: s.rolling(3, min_periods=1).median())
    player_rows["speed_kmh"] = np.clip(
        player_rows["speed_px_s_smooth"] * PX_TO_METERS * 3.6, 0.0, 34.5
    )

    team_lookup = dict(zip(stats_df["player_track_id"], stats_df["team_id"]))
    player_rows["team_id"] = player_rows["track_id"].map(team_lookup)

    # 2. Ball trajectory interpolation & frame-by-frame carrier/pressure detection
    max_frame = int(track_df["frame"].max())
    ball_rows = (
        track_df[track_df["class_name"] == "ball"]
        .sort_values(["frame", "confidence"], ascending=[True, False])
        .drop_duplicates("frame")
    )

    ball_by_frame = {}
    for f in range(1, max_frame + 1):
        b_sub = ball_rows[ball_rows["frame"] == f]
        if not b_sub.empty:
            ball_by_frame[f] = (
                float(b_sub.iloc[0]["center_x"]),
                float(b_sub.iloc[0]["center_y"]),
            )
        else:
            ball_by_frame[f] = (np.nan, np.nan)

    ball_series = pd.DataFrame.from_dict(
        ball_by_frame, orient="index", columns=["bx", "by"]
    )
    ball_series["bx_interp"] = ball_series["bx"].interpolate("linear", limit=15)
    ball_series["by_interp"] = ball_series["by"].interpolate("linear", limit=15)

    carrier_by_frame = {}
    defenders_within_150_by_frame = {}

    for f in range(1, max_frame + 1):
        bx, by = ball_series.loc[f, "bx_interp"], ball_series.loc[f, "by_interp"]
        if pd.isna(bx) or pd.isna(by):
            carrier_by_frame[f] = None
            defenders_within_150_by_frame[f] = 0
            continue

        p_sub = player_rows[player_rows["frame"] == f]
        if p_sub.empty:
            carrier_by_frame[f] = None
            defenders_within_150_by_frame[f] = 0
            continue

        dists = np.hypot(p_sub["foot_x"] - bx, p_sub["foot_y"] - by)
        min_idx = dists.idxmin()
        min_d = dists.loc[min_idx]
        carrier_tid = (
            int(p_sub.loc[min_idx, "track_id"])
            if min_d <= POSSESSION_LIMIT_PX
            else None
        )
        carrier_team = (
            p_sub.loc[min_idx, "team_id"] if carrier_tid is not None else np.nan
        )
        carrier_by_frame[f] = carrier_tid

        opps = (
            p_sub[p_sub["team_id"] != carrier_team]
            if pd.notna(carrier_team)
            else p_sub
        )
        if not opps.empty:
            opp_dists = np.hypot(opps["foot_x"] - bx, opps["foot_y"] - by)
            defenders_within_150_by_frame[f] = int((opp_dists <= 150.0).sum())
        else:
            defenders_within_150_by_frame[f] = 0

    player_rows["is_carrier"] = player_rows.apply(
        lambda r: carrier_by_frame.get(int(r["frame"])) == int(r["track_id"]), axis=1
    )
    player_rows["carry_step_px"] = np.where(
        player_rows["is_carrier"], player_rows["clean_step_px"], 0.0
    )

    # 3. Extract contiguous Dribble & Progressive Carry spells
    dribble_counts = {tid: 0 for tid in stats_df["player_track_id"]}
    progressive_carries = {tid: 0 for tid in stats_df["player_track_id"]}

    carrier_spells = []
    cur_tid, start_f, last_f, start_pos, last_pos = None, None, None, None, None

    for f in range(1, max_frame + 1):
        cid = carrier_by_frame.get(f)
        if cid is not None:
            p_row = player_rows[
                (player_rows["frame"] == f) & (player_rows["track_id"] == cid)
            ]
            pos = (
                (float(p_row.iloc[0]["foot_x"]), float(p_row.iloc[0]["foot_y"]))
                if not p_row.empty
                else None
            )
            if cur_tid == cid and (f - last_f) <= 12:
                last_f = f
                if pos:
                    last_pos = pos
            else:
                if cur_tid is not None and start_pos and last_pos:
                    disp = math.hypot(
                        last_pos[0] - start_pos[0], last_pos[1] - start_pos[1]
                    )
                    carrier_spells.append(
                        (cur_tid, start_f, last_f, disp, last_pos[0] - start_pos[0])
                    )
                cur_tid, start_f, last_f, start_pos, last_pos = cid, f, f, pos, pos

    if cur_tid is not None and start_pos and last_pos:
        disp = math.hypot(last_pos[0] - start_pos[0], last_pos[1] - start_pos[1])
        carrier_spells.append(
            (cur_tid, start_f, last_f, disp, last_pos[0] - start_pos[0])
        )

    for tid, sf, lf, disp, dx in carrier_spells:
        if disp >= 20.0 or (lf - sf) >= 3:
            dribble_counts[tid] = dribble_counts.get(tid, 0) + 1
        if abs(dx) >= 35.0:
            progressive_carries[tid] = progressive_carries.get(tid, 0) + 1

    # 4. Pass Quality & Difficulty Engine (Section 4.1(ii))
    pass_enriched = pass_df.copy()
    pass_enriched["flight_s"] = pass_enriched["evidence"].apply(
        lambda x: float(re.search(r"Flight=([\d\.]+)s", str(x)).group(1))
        if re.search(r"Flight=([\d\.]+)s", str(x))
        else 0.5
    )
    pass_enriched["travel_px"] = pass_enriched["evidence"].apply(
        lambda x: float(re.search(r"Travel=([\d\.]+)px", str(x)).group(1))
        if re.search(r"Travel=([\d\.]+)px", str(x))
        else 100.0
    )
    pass_enriched["travel_m"] = (pass_enriched["travel_px"] * PX_TO_METERS).round(2)
    pass_enriched["ball_speed_px_s"] = (
        pass_enriched["travel_px"] / pass_enriched["flight_s"].clip(lower=0.08)
    ).round(1)
    pass_enriched["ball_speed_kmh"] = np.clip(
        pass_enriched["ball_speed_px_s"] * PX_TO_METERS * 3.6 * 0.45, 14.0, 98.0
    ).round(1)

    opps_in_corridor, receiver_pressure_px, pass_difficulty = [], [], []

    for _, ev in pass_enriched.iterrows():
        f_rec = int(ev["frame"])
        f_start = max(1, int(round(f_rec - ev["flight_s"] * FPS)))
        pid = int(ev["passer_id"])
        rid = int(ev["receiver_id"]) if pd.notna(ev["receiver_id"]) else -1
        p_team = int(ev["passer_team"]) if pd.notna(ev["passer_team"]) else 0

        p_sub = player_rows[
            (player_rows["frame"].between(f_start - 3, f_start + 3))
            & (player_rows["track_id"] == pid)
        ]
        r_sub = player_rows[
            (player_rows["frame"].between(f_rec - 3, f_rec + 3))
            & (player_rows["track_id"] == rid)
        ]

        x1, y1 = (
            (p_sub["foot_x"].mean(), p_sub["foot_y"].mean())
            if not p_sub.empty
            else (600.0, 500.0)
        )
        x2, y2 = (
            (r_sub["foot_x"].mean(), r_sub["foot_y"].mean())
            if not r_sub.empty
            else (1000.0, 500.0)
        )

        f_players = player_rows[player_rows["frame"] == f_rec]
        opps = f_players[f_players["team_id"] != p_team]

        seg_len = math.hypot(x2 - x1, y2 - y1)
        corridor_count = 0
        min_rec_opp_dist = 250.0
        for _, opp in opps.iterrows():
            ox, oy = opp["foot_x"], opp["foot_y"]
            d_rec = math.hypot(ox - x2, oy - y2)
            if d_rec < min_rec_opp_dist:
                min_rec_opp_dist = d_rec
            if seg_len > 1.0:
                t = max(
                    0.0,
                    min(
                        1.0,
                        ((ox - x1) * (x2 - x1) + (oy - y1) * (y2 - y1)) / (seg_len**2),
                    ),
                )
                proj_x, proj_y = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
                if math.hypot(ox - proj_x, oy - proj_y) <= 85.0:
                    corridor_count += 1

        opps_in_corridor.append(corridor_count)
        receiver_pressure_px.append(round(min_rec_opp_dist, 1))

        dist_score = min(40.0, (ev["travel_px"] / 900.0) * 40.0)
        cong_score = min(35.0, corridor_count * 11.5)
        press_score = max(0.0, min(25.0, (180.0 - min_rec_opp_dist) / 180.0 * 25.0))
        diff_rating = round(
            min(98.0, max(12.0, dist_score + cong_score + press_score)), 1
        )
        pass_difficulty.append(diff_rating)

    pass_enriched["opponents_in_corridor"] = opps_in_corridor
    pass_enriched["receiver_defender_dist_px"] = receiver_pressure_px
    pass_enriched["pass_difficulty_rating"] = pass_difficulty
    pass_enriched["expected_pass_completion_xp"] = (
        100.0 - pass_enriched["pass_difficulty_rating"] * 0.65
    ).round(1)
    pass_enriched.to_csv(PASS_ENRICHED_OUT, index=False, encoding="utf-8-sig")

    # 5. Merge Player Dossiers
    player_agg = (
        player_rows.groupby("track_id")
        .agg(
            total_distance_px=("clean_step_px", "sum"),
            avg_speed_kmh=("speed_kmh", "mean"),
            max_speed_kmh=("speed_kmh", "max"),
            ball_touch_frames=("is_carrier", "sum"),
            carry_distance_px=("carry_step_px", "sum"),
        )
        .reset_index()
        .rename(columns={"track_id": "player_track_id"})
    )
    player_agg["total_distance_m"] = (
        player_agg["total_distance_px"] * PX_TO_METERS
    ).round(1)
    player_agg["carry_distance_m"] = (
        player_agg["carry_distance_px"] * PX_TO_METERS
    ).round(2)
    player_agg["avg_speed_kmh"] = player_agg["avg_speed_kmh"].round(1)
    player_agg["max_speed_kmh"] = player_agg["max_speed_kmh"].round(1)
    player_agg["dribble_count"] = (
        player_agg["player_track_id"].map(dribble_counts).fillna(0).astype(int)
    )
    player_agg["progressive_carries"] = (
        player_agg["player_track_id"].map(progressive_carries).fillna(0).astype(int)
    )

    full_player_report = pd.merge(
        stats_df, player_agg, on="player_track_id", how="left"
    )
    passer_metrics = (
        pass_enriched.groupby("passer_id")
        .agg(
            avg_pass_dist_m=("travel_m", "mean"),
            avg_pass_difficulty=("pass_difficulty_rating", "mean"),
            max_ball_speed_kmh=("ball_speed_kmh", "max"),
        )
        .reset_index()
        .rename(columns={"passer_id": "player_track_id"})
    )
    full_player_report = pd.merge(
        full_player_report, passer_metrics, on="player_track_id", how="left"
    )
    full_player_report["avg_pass_dist_m"] = full_player_report[
        "avg_pass_dist_m"
    ].round(1)
    full_player_report["avg_pass_difficulty"] = full_player_report[
        "avg_pass_difficulty"
    ].round(1)
    full_player_report["synthetic_player_name"] = full_player_report[
        "player_track_id"
    ].map(lambda tid: SYNTHETIC_ROSTER.get(int(tid), f"PremierSim Player #{int(tid)}"))

    full_player_report.to_csv(DOSSIERS_OUT, index=False, encoding="utf-8-sig")

    # 6. Control vs. Chaos & Tactical Pressure Timeline (Section 4.1(iv))
    total_secs = int(math.ceil(max_frame / FPS))
    windows = []
    for sec in range(total_secs):
        f_start = sec * int(FPS) + 1
        f_end = min(max_frame, (sec + 1) * int(FPS))
        ev_sub = pass_enriched[pass_enriched["frame"].between(f_start, f_end)]
        turnovers = (
            ev_sub["outcome_candidate"].str.contains("interception|turnover").sum()
        )
        completed = ev_sub["outcome_candidate"].str.contains("completed").sum()
        defs_near = [
            defenders_within_150_by_frame.get(f, 0) for f in range(f_start, f_end + 1)
        ]
        mean_defs = float(np.mean(defs_near)) if defs_near else 0.0

        raw_control = 40.0 + completed * 28.0 - turnovers * 65.0 - mean_defs * 14.0
        if sec < 4:
            raw_control -= 35.0
        control_idx = float(np.clip(raw_control, -95.0, 95.0))
        pressure_idx = float(
            np.clip(25.0 + mean_defs * 20.0 + turnovers * 35.0, 10.0, 98.0)
        )
        windows.append(
            {
                "second": sec + 1,
                "control_vs_chaos_index": round(control_idx, 1),
                "tactical_pressure_index": round(pressure_idx, 1),
                "state_label": "CHAOS (Transition/Turnover)"
                if control_idx < 0
                else "CONTROL (Structured Buildup)",
            }
        )

    timeline_df = pd.DataFrame(windows)
    timeline_df.to_csv(TIMELINE_OUT, index=False, encoding="utf-8-sig")

    # 7. Export JSON payload for Microsoft Foundry Multi-Agent Orchestrator
    active_players = full_player_report[
        (full_player_report["pass_attempt_candidates"] > 0)
        | (full_player_report["receiver_candidates"] > 0)
        | (full_player_report["dribble_count"] > 0)
    ].sort_values(
        ["completed_reception_candidates", "carry_distance_m"], ascending=False
    )

    payload = {
        "hackathon_pipeline": "Ingest -> Interpret -> Explain -> Render -> Personalize",
        "match_summary": {
            "total_frames": max_frame,
            "duration_seconds": round(max_frame / FPS, 2),
            "total_events": len(pass_enriched),
            "completed_passes": int(
                pass_enriched["outcome_candidate"].str.contains("completed").sum()
            ),
            "turnovers": int(
                pass_enriched["outcome_candidate"]
                .str.contains("interception|turnover")
                .sum()
            ),
            "mean_pass_difficulty": round(
                float(pass_enriched["pass_difficulty_rating"].mean()), 1
            ),
            "mean_expected_completion_xp": round(
                float(pass_enriched["expected_pass_completion_xp"].mean()), 1
            ),
            "mean_control_index": round(
                float(timeline_df["control_vs_chaos_index"].mean()), 1
            ),
        },
        "active_player_dossiers": active_players[
            [
                "player_track_id",
                "synthetic_player_name",
                "team_id",
                "frames_seen",
                "pass_attempt_candidates",
                "completed_reception_candidates",
                "receiver_candidates",
                "dribble_count",
                "progressive_carries",
                "ball_touch_frames",
                "carry_distance_m",
                "total_distance_m",
                "max_speed_kmh",
                "avg_pass_difficulty",
            ]
        ]
        .fillna(0)
        .to_dict(orient="records"),
        "enriched_pass_events": pass_enriched.fillna("").to_dict(orient="records"),
    }

    with open(INTEL_JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Step 1 Complete! Files created:")
    print(" -", DOSSIERS_OUT)
    print(" -", PASS_ENRICHED_OUT)
    print(" -", TIMELINE_OUT)
    print(" -", INTEL_JSON_OUT)


if __name__ == "__main__":
    main()