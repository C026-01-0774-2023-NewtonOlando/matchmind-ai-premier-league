import base64
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="MatchMind AI — Premier League Broadcast Hub",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# GLOBAL STREAMLIT DARK BROADCAST STUDIO CSS
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&family=JetBrains+Mono:wght@600;800&display=swap');

    /* Main Streamlit App Background */
    .stApp {
        background: radial-gradient(circle at 15% 10%, #240229 0%, #09000c 55%, #050007 100%) !important;
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Tighten top padding for maximum broadcast screen real estate */
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100% !important;
    }

    /* Sleek Premier League Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1b0220 0%, #0d0011 100%) !important;
        border-right: 1px solid rgba(0, 255, 133, 0.22) !important;
    }
    [data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
    }

    /* Sidebar Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00ff85 0%, #059669 100%) !important;
        color: #1a001f !important;
        font-weight: 800 !important;
        border: none !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 14px rgba(0, 255, 133, 0.25) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(0, 255, 133, 0.45) !important;
    }

    iframe {
        border: none !important;
        border-radius: 16px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

VIDEO_PATH = OUTPUT_DIR / "matchmind_analysis.mp4"
WEB_VIDEO_PATH = OUTPUT_DIR / "matchmind_analysis_web.mp4"
DOSSIERS_CSV = OUTPUT_DIR / "player_dossiers_enriched.csv"
STATS_CSV = OUTPUT_DIR / "player_statistics.csv"
PASS_CSV = OUTPUT_DIR / "pass_events_enriched.csv"
RAW_PASS_CSV = OUTPUT_DIR / "pass_events.csv"
TIMELINE_CSV = OUTPUT_DIR / "control_chaos_timeline.csv"
TRACKING_CSV = OUTPUT_DIR / "tracking_data.csv"

FPS = 24.0
WIDTH, HEIGHT = 1920, 1080
POSSESSION_LIMIT_PX = max(35.0, math.hypot(WIDTH, HEIGHT) * 0.045)
MIN_POSSESSION_FRAMES = 2
BALL_MAX_GAP_FRAMES = 10
PX_TO_METERS = 105.0 / 1920.0
PRESS_RADIUS_M = 12.0


def get_pitch_zone_and_attack_ratio(
    x_px: float,
    carrier_team: int,
    left_gk_team: int,
    right_gk_team: int,
    width_px: float = 1920.0,
) -> tuple[str, float]:
    """
    Goalkeeper-proximity tactical third calculation:
    - Left Side (x < 640, near Left Goalkeeper):
        * If carrier_team == left_gk_team -> Defensive Third (Team {carrier_team})
        * If carrier_team == right_gk_team -> Attacking Third (Team {carrier_team})
    - Right Side (x > 1280, near Right Goalkeeper):
        * If carrier_team == right_gk_team -> Defensive Third (Team {carrier_team})
        * If carrier_team == left_gk_team -> Attacking Third (Team {carrier_team})
    - Middle (640 <= x <= 1280):
        * Middle Third (Team {carrier_team})
    """
    t_id = int(carrier_team)
    raw_ratio = float(np.clip(x_px / max(1.0, width_px), 0.0, 1.0))

    if t_id == int(left_gk_team):
        attack_ratio = raw_ratio
    else:
        attack_ratio = 1.0 - raw_ratio

    left_cut = width_px / 3.0
    right_cut = (width_px * 2.0) / 3.0

    if x_px < left_cut:
        if t_id == int(left_gk_team):
            return f"Defensive Third (Team {t_id})", attack_ratio
        else:
            return f"Attacking Third (Team {t_id})", attack_ratio
    elif x_px > right_cut:
        if t_id == int(right_gk_team):
            return f"Defensive Third (Team {t_id})", attack_ratio
        else:
            return f"Attacking Third (Team {t_id})", attack_ratio
    else:
        return f"Middle Third (Team {t_id})", attack_ratio


@st.cache_resource
def get_playable_video_b64(raw_video_str: str, web_video_str: str) -> str | None:
    raw_video = Path(raw_video_str)
    web_video = Path(web_video_str)

    if not web_video.exists() or web_video.stat().st_size < 1000:
        if raw_video.exists():
            try:
                import imageio_ffmpeg

                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                subprocess.run(
                    [
                        ffmpeg_exe,
                        "-y",
                        "-i",
                        str(raw_video),
                        "-vcodec",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-preset",
                        "fast",
                        "-movflags",
                        "+faststart",
                        str(web_video),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass

    target = (
        web_video
        if (web_video.exists() and web_video.stat().st_size > 1000)
        else raw_video
    )
    if target.exists():
        return base64.b64encode(target.read_bytes()).decode("utf-8")
    return None


@st.cache_data
def build_premier_league_telemetry(setup_mode: str):
    tracks = pd.read_csv(TRACKING_CSV) if TRACKING_CSV.exists() else pd.DataFrame()
    dossiers = (
        pd.read_csv(DOSSIERS_CSV)
        if DOSSIERS_CSV.exists()
        else (pd.read_csv(STATS_CSV) if STATS_CSV.exists() else pd.DataFrame())
    )
    passes = (
        pd.read_csv(PASS_CSV)
        if PASS_CSV.exists()
        else (pd.read_csv(RAW_PASS_CSV) if RAW_PASS_CSV.exists() else pd.DataFrame())
    )
    timeline = pd.read_csv(TIMELINE_CSV) if TIMELINE_CSV.exists() else pd.DataFrame()

    if tracks.empty or dossiers.empty:
        return dossiers, passes, timeline, tracks, {}

    if "synthetic_player_name" in dossiers.columns:
        dossiers = dossiers.drop(columns=["synthetic_player_name"])

    max_frame = int(tracks["frame"].max())

    swap_teams = "Swap Team 0 & 1" in setup_mode
    if "Team 0 GK on Right" in setup_mode:
        left_gk_team = 1
        right_gk_team = 0
    else:
        left_gk_team = 0
        right_gk_team = 1

    team_lookup = {}
    for _, r in dossiers.iterrows():
        raw_t = int(r["team_id"]) if pd.notna(r["team_id"]) else 0
        raw_t = 0 if raw_t <= 0 else 1
        team_lookup[int(r["player_track_id"])] = (1 - raw_t) if swap_teams else raw_t

    player_rows = (
        tracks[tracks["class_name"] == "player"]
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

    player_rows["team_id"] = player_rows["track_id"].map(
        lambda x: team_lookup.get(int(x), 0)
    )

    last_known_ball = None
    candidate_holder_id = None
    candidate_holder_streak = 0
    confirmed_holder_id = None
    last_confirmed_id = None

    hud_carrier_by_frame = {}
    raw_carrier_by_frame = {}
    radar_frames = {}
    ball_x_by_frame = {}
    carrier_zone_by_frame = {}
    pressure_telemetry_by_sample = {}
    ctrl_wave_samples = []

    team0_poss_frames = 0
    team1_poss_frames = 0
    team0_att_third_frames = 0
    team1_att_third_frames = 0
    team_stats_by_sample = {}

    prev_carrier_team = None
    team_possession_streak_frames = 0

    for f in range(1, max_frame + 1):
        f_df = tracks[tracks["frame"] == f]
        balls = f_df[f_df["class_name"] == "ball"]
        p_sub = f_df[f_df["class_name"] == "player"]

        active_ball_center = None
        if not balls.empty:
            best_b = balls.sort_values("confidence", ascending=False).iloc[0]
            bc = (float(best_b["center_x"]), float(best_b["center_y"]))
            active_ball_center = bc
            if (
                last_known_ball is not None
                and 1 < (f - last_known_ball["frame"]) <= BALL_MAX_GAP_FRAMES
            ):
                prev_c = last_known_ball["center"]
                active_ball_center = (
                    (prev_c[0] + bc[0]) / 2.0,
                    (prev_c[1] + bc[1]) / 2.0,
                )
            last_known_ball = {"center": bc, "frame": f}
        elif (
            last_known_ball is not None
            and (f - last_known_ball["frame"]) <= BALL_MAX_GAP_FRAMES
        ):
            active_ball_center = last_known_ball["center"]

        if active_ball_center is not None:
            ball_x_by_frame[f] = active_ball_center[0]

        raw_holder_id = None
        if active_ball_center is not None and not p_sub.empty:
            fx = (p_sub["x1"] + p_sub["x2"]) / 2.0
            fy = p_sub["y2"]
            dists = np.hypot(fx - active_ball_center[0], fy - active_ball_center[1])
            min_idx = dists.idxmin()
            if dists.loc[min_idx] <= POSSESSION_LIMIT_PX:
                raw_holder_id = int(p_sub.loc[min_idx, "track_id"])

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
        if confirmed_holder_id is not None:
            last_confirmed_id = confirmed_holder_id
        elif raw_holder_id is not None and last_confirmed_id is None:
            last_confirmed_id = raw_holder_id

        cur_c = (
            confirmed_holder_id
            if confirmed_holder_id is not None
            else (last_confirmed_id if last_confirmed_id is not None else 8)
        )
        raw_carrier_by_frame[f] = raw_holder_id
        hud_carrier_by_frame[f] = cur_c

        carrier_rows_now = p_sub[p_sub["track_id"] == cur_c]
        if not carrier_rows_now.empty:
            bx_now = float(
                (carrier_rows_now.iloc[0]["x1"] + carrier_rows_now.iloc[0]["x2"]) / 2.0
            )
            by_now = float(carrier_rows_now.iloc[0]["y2"])
        elif active_ball_center is not None:
            bx_now = active_ball_center[0]
            by_now = active_ball_center[1]
        else:
            bx_now = ball_x_by_frame.get(f - 1, 960.0)
            by_now = 540.0

        ball_x_by_frame[f] = bx_now

        c_team = team_lookup.get(cur_c, 0)
        if c_team == prev_carrier_team:
            team_possession_streak_frames += 1
        else:
            prev_carrier_team = c_team
            team_possession_streak_frames = 1

        z_now, att_ratio_now = get_pitch_zone_and_attack_ratio(
            bx_now, c_team, left_gk_team, right_gk_team, WIDTH
        )
        carrier_zone_by_frame[str(f)] = z_now

        if c_team == 0:
            team0_poss_frames += 1
            if att_ratio_now >= 0.667:
                team0_att_third_frames += 1
        else:
            team1_poss_frames += 1
            if att_ratio_now >= 0.667:
                team1_att_third_frames += 1

        if f % 4 == 1 or f == max_frame:
            tot_p = max(1, team0_poss_frames + team1_poss_frames)
            t0_poss = round(100.0 * team0_poss_frames / tot_p, 1)
            t1_poss = round(100.0 - t0_poss, 1)

            ev_so_far = passes[passes["frame"] <= f]
            t0_att, t0_comp, t1_att, t1_comp = 0, 0, 0, 0
            for _, er in ev_so_far.iterrows():
                pt = team_lookup.get(int(er["passer_id"]), 0)
                rt = team_lookup.get(int(er["receiver_id"]), 0)
                is_c = pt == rt
                if pt == 0:
                    t0_att += 1
                    if is_c:
                        t0_comp += 1
                else:
                    t1_att += 1
                    if is_c:
                        t1_comp += 1

            team_stats_by_sample[str(f)] = {
                "t0_poss": t0_poss,
                "t1_poss": t1_poss,
                "t0_passes": f"{t0_comp}/{t0_att}",
                "t1_passes": f"{t1_comp}/{t1_att}",
                "t0_att_f": team0_att_third_frames,
                "t1_att_f": team1_att_third_frames,
            }

            opp_dists_m = []
            pressing_opp_coords = []
            opp_positions = []
            teammate_candidates = []
            pts = []

            for _, pr in p_sub.iterrows():
                ptid = int(pr["track_id"])
                px_i = float((pr["x1"] + pr["x2"]) / 2.0)
                py_i = float(pr["y2"])
                pt_team = team_lookup.get(ptid, 0)
                pts.append([ptid, round(px_i, 1), round(py_i, 1), pt_team])

                if pt_team != c_team:
                    opp_positions.append((px_i, py_i))
                    dist_m = math.hypot(px_i - bx_now, py_i - by_now) * PX_TO_METERS
                    opp_dists_m.append(dist_m)
                    if dist_m <= PRESS_RADIUS_M:
                        pressing_opp_coords.append([round(px_i, 1), round(py_i, 1)])
                elif ptid != cur_c:
                    teammate_candidates.append((ptid, px_i, py_i))

            nearest_opp_m = round(min(opp_dists_m), 1) if opp_dists_m else 18.5
            pressers_count = len(pressing_opp_coords)

            best_opt = None
            best_score = -999.0
            for tm_id, tx, ty in teammate_candidates:
                pass_dist_m = math.hypot(tx - bx_now, ty - by_now) * PX_TO_METERS
                if pass_dist_m < 4.0 or pass_dist_m > 42.0:
                    continue
                tm_opp_clear_m = (
                    min(
                        math.hypot(tx - ox, ty - oy) * PX_TO_METERS
                        for ox, oy in opp_positions
                    )
                    if opp_positions
                    else 15.0
                )
                _, tm_att_ratio = get_pitch_zone_and_attack_ratio(
                    tx, c_team, left_gk_team, right_gk_team, WIDTH
                )
                prog_gain = tm_att_ratio - att_ratio_now
                score = (
                    (tm_opp_clear_m * 1.8)
                    + (prog_gain * 45.0)
                    - abs(pass_dist_m - 16.0) * 0.4
                )
                if score > best_score:
                    best_score = score
                    opt_xp = int(
                        np.clip(82 + tm_opp_clear_m * 1.4 - pass_dist_m * 0.55, 52, 96)
                    )
                    best_opt = {
                        "id": tm_id,
                        "x": round(tx, 1),
                        "y": round(ty, 1),
                        "dist_m": round(pass_dist_m, 1),
                        "space_m": round(tm_opp_clear_m, 1),
                        "xp": opt_xp,
                    }

            proximity_stress = max(0.0, (14.0 - nearest_opp_m) / 14.0) * 65.0
            swarm_stress = min(35.0, pressers_count * 14.0)
            press_intensity = round(
                float(np.clip(proximity_stress + swarm_stress, 4.0, 98.0)), 1
            )

            stability_bonus = min(
                45.0, (team_possession_streak_frames / 24.0) * 14.0
            )
            space_bonus = max(-35.0, (nearest_opp_m - 6.5) * 7.5)
            turnover_penalty = (
                -45.0 if team_possession_streak_frames < 28 and f < 95 else 0.0
            )
            spatial_ctrl = round(
                float(
                    np.clip(
                        20.0
                        + stability_bonus
                        + space_bonus
                        - (pressers_count * 12.0)
                        + turnover_penalty,
                        -92.0,
                        96.0,
                    )
                ),
                1,
            )

            if pressers_count == 0 and nearest_opp_m >= 9.0:
                press_label = f"Unpressed Control ({nearest_opp_m}m Space)"
            elif pressers_count == 1 and nearest_opp_m >= 5.5:
                press_label = f"Light Shadow Press ({nearest_opp_m}m)"
            else:
                press_label = f"High-Chaos Press ({pressers_count} Opp, {nearest_opp_m}m)"

            pressure_telemetry_by_sample[str(f)] = {
                "ctrl": spatial_ctrl,
                "press": press_intensity,
                "nearest_m": nearest_opp_m,
                "pressers": pressers_count,
                "label": press_label,
                "carrier_xy": [round(bx_now, 1), round(by_now, 1)],
                "press_lines": pressing_opp_coords,
                "best_opt": best_opt,
            }
            ctrl_wave_samples.append([f, spatial_ctrl])

            b_pt = (
                [round(active_ball_center[0], 1), round(active_ball_center[1], 1)]
                if active_ball_center is not None
                else None
            )
            radar_frames[str(f)] = {"p": pts, "b": b_pt}

    player_rows["is_carrier"] = player_rows.apply(
        lambda r: raw_carrier_by_frame.get(int(r["frame"])) == int(r["track_id"]),
        axis=1,
    )
    player_rows["carry_step_px"] = np.where(
        player_rows["is_carrier"], player_rows["clean_step_px"], 0.0
    )

    active_players_df = dossiers[
        (dossiers["pass_attempt_candidates"] > 0)
        | (dossiers["receiver_candidates"] > 0)
        | (dossiers.get("dribble_count", pd.Series(0, index=dossiers.index)) > 0)
    ].sort_values("player_track_id")

    active_id_list = active_players_df["player_track_id"].astype(int).tolist()

    player_timelines = {}
    player_totals = {}
    player_trails = {}

    for pid in active_id_list:
        p_df = player_rows[player_rows["track_id"] == pid].sort_values("frame")
        base_r = active_players_df[active_players_df["player_track_id"] == pid].iloc[0]
        max_drib_cap = int(base_r.get("dribble_count", 3))
        p_team = team_lookup.get(pid, 0)

        total_dist_m = round(float(p_df["clean_step_px"].sum() * PX_TO_METERS), 1)
        total_carry_m = round(float(p_df["carry_step_px"].sum() * PX_TO_METERS), 2)
        total_touches = int(p_df["is_carrier"].sum())
        peak_spd = round(float(p_df["speed_kmh"].max()), 1) if not p_df.empty else 0.0

        p_ev_all = passes[passes["passer_id"] == pid]
        total_att = len(p_ev_all)
        total_comp = sum(
            1
            for _, er in p_ev_all.iterrows()
            if team_lookup.get(int(er["passer_id"]), 0)
            == team_lookup.get(int(er["receiver_id"]), 0)
        )
        total_rec = int((passes["receiver_id"] == pid).sum())

        avg_x = float(p_df["foot_x"].mean()) if not p_df.empty else 960.0
        _, avg_att_ratio = get_pitch_zone_and_attack_ratio(
            avg_x, p_team, left_gk_team, right_gk_team, WIDTH
        )
        zone_multiplier = 0.65 + 0.95 * avg_att_ratio
        xt_total = round(
            (0.045 * total_comp + 0.025 * max_drib_cap + 0.008 * total_carry_m)
            * zone_multiplier,
            3,
        )
        press_res = int(
            np.clip(
                68
                + (12 if total_comp == total_att and total_att > 0 else -10)
                + max_drib_cap * 6,
                45,
                98,
            )
        )

        prog_rating = int(np.clip(62 + total_comp * 14 + total_carry_m * 3.5, 50, 97))
        ret_rating = int(
            95
            if (total_att > 0 and total_comp == total_att)
            else (84 if total_rec > 0 else 64)
        )
        work_rating = int(np.clip(55 + (total_dist_m / 95.0) * 42, 52, 98))
        drib_rating = int(np.clip(58 + max_drib_cap * 11 + total_carry_m * 3.2, 50, 96))
        pace_rating = int(np.clip((peak_spd / 35.0) * 96, 55, 97))

        player_totals[str(pid)] = {
            "pid": pid,
            "team": p_team,
            "total_dist_m": max(0.1, total_dist_m),
            "total_carry_m": max(0.0, total_carry_m),
            "total_drib": max_drib_cap,
            "total_touches": total_touches,
            "peak_spd": peak_spd,
            "total_att": total_att,
            "total_comp": total_comp,
            "total_rec": total_rec,
            "xt": max(0.01, xt_total),
            "press_res": press_res,
            "radar": [prog_rating, ret_rating, work_rating, drib_rating, pace_rating],
        }

        trail_pts = []
        for _, tr_row in p_df[p_df["frame"] % 4 == 1].iterrows():
            trail_pts.append(
                [
                    int(tr_row["frame"]),
                    round(float(tr_row["foot_x"]), 1),
                    round(float(tr_row["foot_y"]), 1),
                ]
            )
        player_trails[str(pid)] = trail_pts

        p_frames_dict = {}
        for sample_f in list(range(1, max_frame + 1, 4)) + [max_frame]:
            sub_p = p_df[p_df["frame"] <= sample_f]
            sub_ev = passes[passes["frame"] <= sample_f]

            p_sub_ev = sub_ev[sub_ev["passer_id"] == pid]
            att = len(p_sub_ev)
            comp = sum(
                1
                for _, er in p_sub_ev.iterrows()
                if team_lookup.get(int(er["passer_id"]), 0)
                == team_lookup.get(int(er["receiver_id"]), 0)
            )
            rec = int((sub_ev["receiver_id"] == pid).sum())

            ref_x = (
                float(sub_p.iloc[-1]["foot_x"])
                if not sub_p.empty
                else ball_x_by_frame.get(sample_f, 960.0)
            )
            zone, cur_att_ratio = get_pitch_zone_and_attack_ratio(
                ref_x, p_team, left_gk_team, right_gk_team, WIDTH
            )
            live_z_mult = 0.65 + 0.95 * cur_att_ratio

            if sub_p.empty:
                p_frames_dict[str(sample_f)] = {
                    "att": att,
                    "comp": comp,
                    "rec": rec,
                    "drib": 0,
                    "carry_m": 0.0,
                    "touches": 0,
                    "dist_m": 0.0,
                    "cur_spd": 0.0,
                    "max_spd": 0.0,
                    "xt": 0.0,
                    "zone": zone,
                }
            else:
                touches = int(sub_p["is_carrier"].sum())
                carry_m = round(float(sub_p["carry_step_px"].sum() * PX_TO_METERS), 2)
                dist_m = round(float(sub_p["clean_step_px"].sum() * PX_TO_METERS), 1)
                max_spd = round(float(sub_p["speed_kmh"].max()), 1)
                cur_spd = round(float(sub_p.iloc[-1]["speed_kmh"]), 1)

                c_frames = sub_p[sub_p["is_carrier"]]["frame"].tolist()
                drib = 0
                if c_frames:
                    spells = 1
                    for i in range(1, len(c_frames)):
                        if c_frames[i] - c_frames[i - 1] > 12:
                            spells += 1
                    drib = (
                        min(max(1, max_drib_cap), spells)
                        if (carry_m >= 0.30 or touches >= 2)
                        else 0
                    )

                live_xt = round(
                    (0.045 * comp + 0.025 * drib + 0.008 * carry_m) * live_z_mult, 3
                )

                p_frames_dict[str(sample_f)] = {
                    "att": att,
                    "comp": comp,
                    "rec": rec,
                    "drib": drib,
                    "carry_m": carry_m,
                    "touches": touches,
                    "dist_m": dist_m,
                    "cur_spd": cur_spd,
                    "max_spd": max_spd,
                    "xt": live_xt,
                    "zone": zone,
                }

        player_timelines[str(pid)] = p_frames_dict

    events_list = []
    for _, ev in passes.iterrows():
        f_ev = int(ev["frame"])
        p_id = int(ev["passer_id"])
        r_id = int(ev["receiver_id"])
        p_team = team_lookup.get(p_id, 0)
        r_team = team_lookup.get(r_id, 0)

        p_sub = player_rows[
            (player_rows["track_id"] == p_id) & (player_rows["frame"] <= f_ev)
        ].tail(4)
        r_sub = player_rows[
            (player_rows["track_id"] == r_id) & (player_rows["frame"] <= f_ev)
        ].tail(4)
        px = round(float(p_sub["foot_x"].mean()), 1) if not p_sub.empty else 600.0
        py = round(float(p_sub["foot_y"].mean()), 1) if not p_sub.empty else 500.0
        rx = round(float(r_sub["foot_x"].mean()), 1) if not r_sub.empty else 1000.0
        ry = round(float(r_sub["foot_y"].mean()), 1) if not r_sub.empty else 500.0

        f_opps = player_rows[
            (player_rows["frame"] == f_ev) & (player_rows["team_id"] != p_team)
        ]
        min_x, max_x = min(px, rx), max(px, rx)
        bypassed = int(
            ((f_opps["foot_x"] >= min_x) & (f_opps["foot_x"] <= max_x)).sum()
        )

        same_team = p_team == r_team
        outcome_str = (
            "completed_pass_candidate"
            if same_team
            else "turnover_or_contested_candidate"
        )

        events_list.append(
            {
                "event_id": int(ev["event_id"]),
                "sec": round(float(ev["timestamp_seconds"]), 2),
                "frame": f_ev,
                "passer": p_id,
                "passer_team": p_team,
                "receiver": r_id,
                "receiver_team": r_team,
                "px": px,
                "py": py,
                "rx": rx,
                "ry": ry,
                "travel_m": round(float(ev.get("travel_m", 25.0)), 1),
                "speed_kmh": round(float(ev.get("ball_speed_kmh", 55.0)), 1),
                "diff": round(float(ev.get("pass_difficulty_rating", 30.0)), 1),
                "xp": round(float(ev.get("expected_pass_completion_xp", 78.0)), 1),
                "bypassed": bypassed,
                "outcome": outcome_str,
            }
        )

    js_payload = {
        "fps": FPS,
        "max_frame": max_frame,
        "left_gk_team": left_gk_team,
        "right_gk_team": right_gk_team,
        "active_ids": active_id_list,
        "carrier_by_frame": {str(k): int(v) for k, v in hud_carrier_by_frame.items()},
        "carrier_zone_by_frame": carrier_zone_by_frame,
        "pressure_by_sample": pressure_telemetry_by_sample,
        "ctrl_wave": ctrl_wave_samples,
        "team_stats_by_sample": team_stats_by_sample,
        "radar_frames": radar_frames,
        "player_trails": player_trails,
        "player_totals": player_totals,
        "player_timelines": player_timelines,
        "events": events_list,
    }

    return dossiers, passes, timeline, tracks, js_payload


# ============================================================
# SIDEBAR CONTROLS & TELEMETRY LOADING
# ============================================================
st.sidebar.markdown(
    """
    <div style="background:linear-gradient(135deg,#37003c 0%,#1f0224 100%);padding:14px 16px;border-radius:12px;border:1px solid rgba(0,255,133,0.35);margin-bottom:14px;">
        <div style="font-size:11px;font-weight:900;color:#00ff85;letter-spacing:1.2px;">MICROSOFT × PREMIER LEAGUE</div>
        <div style="font-size:20px;font-weight:900;color:#ffffff;margin-top:2px;">🦁 MatchMind AI Hub</div>
        <div style="font-size:11.5px;color:#cbd5e1;margin-top:2px;">60-FPS Spatial & Multi-Agent Studio</div>
    </div>
    """,
    unsafe_allow_html=True,
)

setup_mode = st.sidebar.radio(
    "🧤 Team & Goalkeeper Side Alignment",
    [
        "Team 0 GK on Right | Team 1 GK on Left",
        "Team 0 GK on Left | Team 1 GK on Right (Swap Team 0 & 1)",
    ],
    index=0,
    help="Aligns team possession, passing stats, and goalkeeper-relative pitch thirds while keeping Control vs. Chaos anchored to true spatial opponent pressing.",
)

language = st.sidebar.selectbox(
    "🌍 Broadcast Language (Rule 4.1(vi))",
    ["English", "Spanish", "French", "Swahili"],
)
agent_mode = st.sidebar.radio(
    "🎭 AI Broadcast Persona (Rule 4.1(i)(e))",
    ["Analyst Mode", "Fan Mode", "Broadcast Mode"],
)

with st.sidebar.expander("☁️ Microsoft AI Foundry Config"):
    azure_endpoint = st.text_input(
        "Foundry Endpoint",
        value=os.getenv(
            "AZURE_OPENAI_ENDPOINT", "https://models.inference.ai.azure.com"
        ),
    )
    azure_key = st.text_input(
        "Foundry / GitHub Token",
        type="password",
        value=os.getenv("AZURE_OPENAI_API_KEY", ""),
    )

dossiers_df, passes_df, timeline_df, tracks_df, js_payload = (
    build_premier_league_telemetry(setup_mode)
)
video_b64 = get_playable_video_b64(str(VIDEO_PATH), str(WEB_VIDEO_PATH))

if not js_payload or video_b64 is None:
    st.warning(
        "Ensure `output/matchmind_analysis.mp4`, `tracking_data.csv`, and `player_statistics.csv` exist."
    )
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Microsoft Foundry Deep-Dive")
selected_cloud_pid = st.sidebar.selectbox(
    "Select Player #ID for Cloud AI Dossier",
    js_payload["active_ids"],
    format_func=lambda x: f"Player ID #{x} (Team {js_payload['player_totals'][str(x)]['team']})",
)
if st.sidebar.button("☁️ Generate Foundry Tactical Brief", use_container_width=True):
    p_tot = js_payload["player_totals"][str(selected_cloud_pid)]
    if azure_key:
        try:
            from openai import OpenAI

            client = OpenAI(base_url=azure_endpoint, api_key=azure_key)
            prompt = (
                f"Respond in {language} using {agent_mode}. Summarize Player ID #{selected_cloud_pid} "
                f"(Team {p_tot['team']}): Distance={p_tot['total_dist_m']}m, Dribbles={p_tot['total_drib']} "
                f"({p_tot['total_carry_m']}m carry), Passes={p_tot['total_comp']}/{p_tot['total_att']}, "
                f"Expected Threat=+{p_tot['xt']} xT, Peak Speed={p_tot['peak_spd']} km/h, "
                f"Press Resistance={p_tot['press_res']}/100. Keep it to 3 punchy bullet points."
            )
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=220,
            )
            st.sidebar.success("Microsoft AI Foundry Response:")
            st.sidebar.markdown(resp.choices[0].message.content)
        except Exception:
            st.sidebar.info(
                f"📌 **Foundry Tactical Brief (ID #{selected_cloud_pid}):** Covered **{p_tot['total_dist_m']}m** with **{p_tot['total_drib']} dribbles ({p_tot['total_carry_m']}m)**, **{p_tot['total_comp']}/{p_tot['total_att']} passes**, and **+{p_tot['xt']} xT**."
            )
    else:
        st.sidebar.success(f"📌 Tactical Summary — Player ID #{selected_cloud_pid}:")
        st.sidebar.markdown(
            f"- **Progression & Threat:** Generated **+{p_tot['xt']} xT** across **{p_tot['total_drib']} dribbles ({p_tot['total_carry_m']}m)**.\n"
            f"- **Passing & Press Resistance:** Completed **{p_tot['total_comp']}/{p_tot['total_att']} passes** (**{p_tot['press_res']}/100** press resistance).\n"
            f"- **Physical Output:** Covered **{p_tot['total_dist_m']}m** at a peak sprint of **{p_tot['peak_spd']} km/h**."
        )

# ============================================================
# ULTRA-BROADCAST HTML5 INTERACTIVE COMMAND CENTER
# ============================================================
payload_json_str = json.dumps(js_payload)

html_component = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&family=JetBrains+Mono:wght@600;700;800&display=swap');

  :root {{
    --pl-purple: #37003c;
    --pl-dark: #0f0114;
    --pl-card: rgba(28, 7, 35, 0.88);
    --pl-green: #00ff85;
    --pl-cyan: #04f5ff;
    --pl-pink: #ff2882;
    --pl-gold: #facc15;
  }}

  * {{
    box-sizing: border-box;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
  }}

  body {{
    margin: 0;
    padding: 14px;
    background: radial-gradient(circle at top right, #2c0433 0%, #0d0012 60%, #060009 100%);
    color: #f8fafc;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.7);
  }}

  /* PULSING KEYFRAMES */
  @keyframes pulseGlow {{
    0% {{ box-shadow: 0 0 0 0 rgba(0, 255, 133, 0.6); }}
    70% {{ box-shadow: 0 0 0 9px rgba(0, 255, 133, 0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(0, 255, 133, 0); }}
  }}

  /* TOP PREMIER LEAGUE BROADCAST SCOREBUG */
  .pl-scoreboard {{
    display: grid;
    grid-template-columns: 1.15fr 1.85fr 1.1fr;
    align-items: center;
    background: linear-gradient(90deg, rgba(26, 2, 31, 0.95) 0%, rgba(55, 0, 60, 0.95) 50%, rgba(26, 2, 31, 0.95) 100%);
    border: 1px solid rgba(0, 255, 133, 0.35);
    border-radius: 14px;
    padding: 11px 20px;
    margin-bottom: 12px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(12px);
  }}

  .sb-left {{ display: flex; align-items: center; gap: 12px; }}
  .pl-badge {{
    background: linear-gradient(135deg, var(--pl-green) 0%, #10b981 100%);
    color: #19001e;
    font-weight: 900;
    font-size: 11px;
    padding: 5px 11px;
    border-radius: 6px;
    letter-spacing: 0.9px;
    text-transform: uppercase;
    box-shadow: 0 0 14px rgba(0, 255, 133, 0.35);
  }}

  .clock-pill {{
    background: #070009;
    border: 1px solid var(--pl-cyan);
    color: var(--pl-cyan);
    font-family: 'JetBrains Mono', monospace;
    font-weight: 800;
    font-size: 14px;
    padding: 4px 13px;
    border-radius: 7px;
    box-shadow: 0 0 12px rgba(4, 245, 255, 0.2);
  }}

  .poss-wrap {{ text-align: center; }}
  .poss-labels {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 11px;
    font-weight: 800;
    margin-bottom: 5px;
    letter-spacing: 0.3px;
  }}

  .poss-bar {{
    height: 9px;
    width: 100%;
    background: linear-gradient(90deg, #ff2882 0%, #db2777 100%);
    border-radius: 99px;
    overflow: hidden;
    display: flex;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.6);
  }}
  .poss-fill-t0 {{
    height: 100%;
    background: linear-gradient(90deg, #04f5ff 0%, #00ff85 100%);
    transition: width 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 0 10px var(--pl-cyan);
  }}

  .sb-right {{ display: flex; justify-content: flex-end; align-items: center; gap: 10px; }}
  .voice-pill {{
    background: rgba(250, 204, 21, 0.12);
    border: 1px solid rgba(250, 204, 21, 0.45);
    padding: 5px 11px;
    border-radius: 8px;
    font-size: 11.5px;
    font-weight: 800;
    color: var(--pl-gold);
    cursor: pointer;
    transition: 0.2s;
  }}
  .voice-pill:hover {{ background: rgba(250, 204, 21, 0.22); }}

  /* MASTER INTERACTIVE CONTROL & TIMELINE BAR */
  .ctrl-bar {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
    background: rgba(255, 255, 255, 0.035);
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 9px 14px;
    border-radius: 12px;
    margin-bottom: 14px;
    backdrop-filter: blur(8px);
  }}

  .play-btn {{
    background: linear-gradient(135deg, var(--pl-green) 0%, #059669 100%);
    color: #16001a;
    border: none;
    padding: 8px 16px;
    border-radius: 8px;
    font-weight: 900;
    font-size: 12.5px;
    cursor: pointer;
    transition: all 0.18s ease;
    box-shadow: 0 4px 15px rgba(0, 255, 133, 0.3);
  }}
  .play-btn:hover {{
    filter: brightness(1.1);
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(0, 255, 133, 0.5);
  }}

  .step-btn {{
    background: rgba(255, 255, 255, 0.08);
    color: #f8fafc;
    border: 1px solid rgba(255, 255, 255, 0.16);
    padding: 6px 9px;
    border-radius: 7px;
    font-size: 11px;
    font-weight: 800;
    cursor: pointer;
    transition: 0.15s;
  }}
  .step-btn:hover, .step-btn.active-spd {{
    background: var(--pl-cyan);
    color: #09000c;
    border-color: var(--pl-cyan);
  }}

  .ev-btn {{
    background: rgba(255, 255, 255, 0.06);
    color: #e2e8f0;
    border: 1px solid rgba(255, 255, 255, 0.14);
    padding: 5px 9px;
    border-radius: 7px;
    font-size: 11px;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.15s ease;
  }}
  .ev-btn:hover {{
    background: rgba(4, 245, 255, 0.2);
    color: #ffffff;
    border-color: var(--pl-cyan);
    transform: translateY(-1px);
  }}
  .ev-btn.active-ev {{
    background: linear-gradient(135deg, var(--pl-cyan) 0%, #0284c7 100%);
    color: #050007;
    border-color: var(--pl-cyan);
    font-weight: 900;
    box-shadow: 0 0 12px rgba(4, 245, 255, 0.45);
  }}

  select.id-select {{
    background: #1a021f;
    color: var(--pl-green);
    border: 1px solid var(--pl-green);
    padding: 5px 10px;
    border-radius: 7px;
    font-weight: 800;
    font-size: 12px;
    cursor: pointer;
    outline: none;
  }}

  /* MAIN 2-COLUMN BROADCAST STUDIO GRID */
  .main-grid {{
    display: grid;
    grid-template-columns: 1.18fr 1fr;
    gap: 16px;
    align-items: start;
  }}

  .glass-card {{
    background: var(--pl-card);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 14px;
    padding: 14px;
    box-shadow: 0 14px 35px rgba(0, 0, 0, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(16px);
  }}

  .video-shell {{
    position: relative;
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid rgba(0, 255, 133, 0.35);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.65);
    background: #000;
  }}

  video {{
    width: 100%;
    display: block;
    cursor: pointer;
  }}

  /* TV BROADCAST LOWER-THIRD BANNER */
  .tv-lower-third {{
    margin-top: 10px;
    background: linear-gradient(90deg, #37003c 0%, #19011e 65%, #110014 100%);
    border-left: 5px solid var(--pl-green);
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    border-right: 1px solid rgba(255, 255, 255, 0.1);
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    padding: 10px 15px;
    border-radius: 9px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.4);
  }}
  .lt-tag {{ font-size: 10px; font-weight: 900; color: var(--pl-green); letter-spacing: 1.1px; }}
  .lt-title {{ font-size: 16px; font-weight: 900; color: #fff; margin-top: 2px; }}
  .lt-metrics {{ font-size: 12px; color: #cbd5e1; font-weight: 600; margin-top: 2px; }}

  /* 2D INTERACTIVE DIGITAL TWIN RADAR CANVAS */
  .radar-wrap {{
    margin-top: 12px;
    background: linear-gradient(180deg, #0b2912 0%, #071c0c 100%);
    border-radius: 11px;
    padding: 9px;
    border: 1px solid rgba(0, 255, 133, 0.28);
    box-shadow: inset 0 0 20px rgba(0,0,0,0.6);
  }}
  .radar-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 10.5px;
    font-weight: 800;
    color: var(--pl-green);
    margin-bottom: 6px;
    padding: 0 4px;
  }}
  canvas#pitchCanvas {{
    width: 100%;
    height: 215px;
    display: block;
    border-radius: 7px;
    cursor: crosshair;
    background: #14532d;
  }}

  /* INTERACTIVE CONTROL VS CHAOS WAVEFORM SCRUBBER */
  .wave-wrap {{
    margin-top: 10px;
    background: rgba(14, 1, 18, 0.92);
    border-radius: 10px;
    padding: 8px 11px;
    border: 1px solid rgba(4, 245, 255, 0.25);
  }}
  .wave-header {{
    display: flex;
    justify-content: space-between;
    font-size: 10.5px;
    font-weight: 800;
    color: var(--pl-cyan);
    margin-bottom: 5px;
  }}
  canvas#waveCanvas {{
    width: 100%;
    height: 70px;
    display: block;
    border-radius: 6px;
    cursor: pointer;
    background: rgba(255, 255, 255, 0.02);
  }}

  /* RIGHT COLUMN: PLAYER ID SCOUTING DOSSIER */
  .dossier-banner {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: linear-gradient(135deg, #3b0041 0%, #1d0122 100%);
    border: 1px solid rgba(0, 255, 133, 0.4);
    padding: 13px 16px;
    border-radius: 12px;
    margin-bottom: 12px;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.35);
  }}
  .pulse-dot {{
    display: inline-block;
    width: 10px;
    height: 10px;
    background: var(--pl-green);
    border-radius: 50%;
    margin-right: 8px;
    animation: pulseGlow 1.8s infinite;
  }}
  .hero-id {{
    font-size: 24px;
    font-weight: 900;
    color: #ffffff;
    letter-spacing: 0.4px;
    margin-top: 2px;
  }}
  .team-badge {{
    padding: 5px 14px;
    border-radius: 99px;
    font-weight: 900;
    font-size: 12px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }}

  /* 6 TELEMETRY CARDS + 5-AXIS PENTAGON ROW */
  .telemetry-split {{
    display: grid;
    grid-template-columns: 1.45fr 0.85fr;
    gap: 12px;
    margin-bottom: 12px;
    align-items: stretch;
  }}
  .metrics-grid {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
  }}
  .m-card {{
    background: linear-gradient(145deg, rgba(255,255,255,0.055) 0%, rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(255, 255, 255, 0.11);
    border-radius: 10px;
    padding: 9px 12px;
    transition: all 0.2s ease;
  }}
  .m-card:hover {{
    border-color: rgba(0, 255, 133, 0.45);
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.35);
  }}
  .m-lbl {{ font-size: 10px; font-weight: 800; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }}
  .m-val {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 18px;
    font-weight: 800;
    color: #ffffff;
    margin: 3px 0 2px 0;
  }}
  .m-sub {{ font-size: 10.5px; color: var(--pl-green); font-weight: 700; }}
  .p-bar {{
    width: 100%;
    height: 5px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 99px;
    margin-top: 5px;
    overflow: hidden;
  }}
  .p-fill {{ height: 100%; border-radius: 99px; transition: width 0.22s ease; }}

  .pentagon-box {{
    background: radial-gradient(circle at center, rgba(55, 0, 60, 0.45) 0%, rgba(10, 0, 14, 0.65) 100%);
    border: 1px solid rgba(4, 245, 255, 0.22);
    border-radius: 12px;
    padding: 10px 8px;
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    align-items: center;
  }}

  /* AI AGENT NARRATIVE BOX */
  .ai-card {{
    background: linear-gradient(135deg, rgba(0, 255, 133, 0.09) 0%, rgba(4, 245, 255, 0.05) 100%);
    border-left: 4px solid var(--pl-green);
    border-top: 1px solid rgba(0, 255, 133, 0.2);
    border-right: 1px solid rgba(4, 245, 255, 0.15);
    border-bottom: 1px solid rgba(4, 245, 255, 0.15);
    padding: 12px 15px;
    border-radius: 10px;
    font-size: 12.8px;
    line-height: 1.55;
    color: #f1f5f9;
    margin-bottom: 12px;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
  }}
  .ai-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-weight: 900;
    font-size: 13px;
    color: var(--pl-green);
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.4px;
  }}

  /* INTERACTIVE CLICKABLE PASS TABLE */
  .pl-table {{ width: 100%; border-collapse: collapse; font-size: 11.5px; }}
  .pl-table th {{
    background: rgba(255, 255, 255, 0.08);
    color: var(--pl-cyan);
    text-align: left;
    padding: 7px 8px;
    font-weight: 800;
    text-transform: uppercase;
    font-size: 10px;
    letter-spacing: 0.5px;
  }}
  .pl-table tbody tr {{
    cursor: pointer;
    transition: background 0.15s ease;
  }}
  .pl-table tbody tr:hover {{
    background: rgba(4, 245, 255, 0.14) !important;
  }}
  .pl-table tbody tr.active-pass-row {{
    background: rgba(0, 255, 133, 0.18) !important;
    outline: 1px solid var(--pl-green);
  }}
  .pl-table td {{
    padding: 7px 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.07);
    color: #e2e8f0;
    font-weight: 600;
  }}
  .tag-ok {{
    background: rgba(0, 255, 133, 0.18);
    color: var(--pl-green);
    border: 1px solid rgba(0, 255, 133, 0.4);
    padding: 2px 7px;
    border-radius: 99px;
    font-weight: 800;
    font-size: 10.5px;
  }}
  .tag-bad {{
    background: rgba(255, 40, 130, 0.22);
    color: var(--pl-pink);
    border: 1px solid rgba(255, 40, 130, 0.45);
    padding: 2px 7px;
    border-radius: 99px;
    font-weight: 800;
    font-size: 10.5px;
  }}
</style>
</head>
<body>

<!-- TOP PREMIER LEAGUE LIVE SCOREBOARD & POSSESSION SPLIT -->
<div class="pl-scoreboard">
  <div class="sb-left">
    <span class="pl-badge">🦁 PREMIER LEAGUE AI</span>
    <span class="clock-pill" id="topClock">00:00.0 • F1</span>
  </div>

  <div class="poss-wrap">
    <div class="poss-labels">
      <span style="color:var(--pl-cyan);" id="possT0Lbl">TEAM 0: 85% POSS (0/0 PASS)</span>
      <span style="color:#e2e8f0; font-size:10px; background:rgba(255,255,255,0.08); padding:2px 8px; border-radius:99px;" id="gkSideBanner">
        T1 GK (LEFT) ⬅️ PITCH ➡️ T0 GK (RIGHT)
      </span>
      <span style="color:var(--pl-pink);" id="possT1Lbl">TEAM 1: 15% POSS (0/0 PASS)</span>
    </div>
    <div class="poss-bar">
      <div class="poss-fill-t0" id="possBarT0" style="width:85%;"></div>
    </div>
  </div>

  <div class="sb-right">
    <label class="voice-pill">
      <input type="checkbox" id="voiceToggle" style="vertical-align:middle;"> 🔊 Live AI Voice
    </label>
  </div>
</div>

<!-- MASTER PLAYBACK, SPEED CONTROLS & QUICK-JUMP EVENT BAR -->
<div class="ctrl-bar">
  <button class="play-btn" id="masterPlayBtn" onclick="togglePlay()">▶️ PLAY BROADCAST</button>
  <button class="step-btn" onclick="stepFrames(-1)" title="Step Back 1 Frame">⏮️ -1F</button>
  <button class="step-btn" onclick="stepFrames(1)" title="Step Forward 1 Frame">+1F ⏭️</button>
  <button class="step-btn" id="spd05" onclick="setSpeed(0.5)">0.5x Slow-Mo</button>
  <button class="step-btn active-spd" id="spd10" onclick="setSpeed(1.0)">1.0x Live</button>

  <span style="font-size:10.5px; font-weight:900; color:#94a3b8; margin-left:4px;">EVENTS:</span>
  <div id="eventButtons" style="display:flex; gap:4px; flex-wrap:wrap;"></div>

  <div style="margin-left:auto; display:flex; align-items:center; gap:8px; font-size:11.5px; font-weight:800;">
    <label style="cursor:pointer; color:var(--pl-green);">
      <input type="checkbox" id="autoFollowChk" checked onchange="updateUI()"> 🎯 Auto-Follow Carrier
    </label>
    <select id="manualIdSelect" class="id-select" onchange="onManualSelect()"></select>
  </div>
</div>

<div class="main-grid">
  <!-- LEFT COLUMN: VIDEO + LOWER-THIRD + INTERACTIVE 2D RADAR + MOMENTUM WAVEFORM -->
  <div class="glass-card">
    <div class="video-shell">
      <video id="matchVideo" controls playsinline preload="auto">
        <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
      </video>
    </div>

    <div class="tv-lower-third">
      <div>
        <div class="lt-tag" id="ltHeader">LIVE BROADCAST TELEMETRY • FRAME 1</div>
        <div class="lt-title" id="ltMain">ON BALL: PLAYER ID #8 (TEAM 0)</div>
        <div class="lt-metrics" id="ltSub">Zone: Defensive Third | Nearest Opp: 10.5m (0 Pressing)</div>
      </div>
      <div style="text-align:right;">
        <div style="font-size:9.5px; color:#94a3b8; font-weight:900; letter-spacing:0.6px;" id="ltPressState">UNPRESSED CONTROL</div>
        <div id="ltCtrl" style="font-family:'JetBrains Mono',monospace; font-size:21px; font-weight:900; color:var(--pl-green);">+45.0</div>
      </div>
    </div>

    <!-- INTERACTIVE 2D DIGITAL TWIN TACTICAL RADAR CANVAS -->
    <div class="radar-wrap">
      <div class="radar-header">
        <span id="radarLeftGoalLbl" style="color:var(--pl-pink);">🥅 LEFT: TEAM 1 GK</span>
        <span style="color:#f8fafc;">🖱️ CLICK ANY PLAYER DOT ON RADAR TO INSPECT • <span style="color:var(--pl-cyan);">CYAN DASH = AI BEST PASS</span></span>
        <span id="radarRightGoalLbl" style="color:var(--pl-cyan);">🥅 RIGHT: TEAM 0 GK</span>
      </div>
      <canvas id="pitchCanvas" width="760" height="225"></canvas>
    </div>

    <!-- CLICK-TO-SEEK CONTROL VS CHAOS WAVEFORM SCRUBBER -->
    <div class="wave-wrap">
      <div class="wave-header">
        <span>📈 SPATIAL CONTROL (+100 GREEN) VS. CHAOS (-100 PINK) MOMENTUM WAVEFORM</span>
        <span style="color:var(--pl-gold);">🖱️ Click anywhere on waveform to scrub match</span>
      </div>
      <canvas id="waveCanvas" width="760" height="72"></canvas>
    </div>
  </div>

  <!-- RIGHT COLUMN: LIVE PLAYER #ID DOSSIER + 5-AXIS RADAR PENTAGON + AI NARRATOR + CLICKABLE TABLE -->
  <div class="glass-card">
    <div class="dossier-banner">
      <div>
        <div style="font-size:10.5px; color:var(--pl-green); font-weight:800; letter-spacing:0.5px;" id="dossierStatus">
          <span class="pulse-dot"></span>LIVE AUTO-FOLLOWING BALL CARRIER
        </div>
        <div class="hero-id" id="dossierPlayerId">Player ID #8</div>
      </div>
      <div style="text-align:right;">
        <span class="team-badge" id="dossierTeamPill" style="background:var(--pl-cyan); color:#000;">Team 0</span>
        <div style="font-size:11.5px; color:#e2e8f0; margin-top:6px; font-weight:700;" id="dossierZone">📍 Defensive Third (Team 0)</div>
      </div>
    </div>

    <!-- 6 METRIC CARDS + 5-AXIS ATTRIBUTE RADAR PENTAGON -->
    <div class="telemetry-split">
      <div class="metrics-grid">
        <div class="m-card">
          <div class="m-lbl">🏃 Distance Covered</div>
          <div class="m-val" id="mDist">0.0 m</div>
          <div class="m-sub" id="mDistSub">Match Total: 0.0 m</div>
          <div class="p-bar"><div class="p-fill" id="barDist" style="width:0%; background:var(--pl-cyan);"></div></div>
        </div>

        <div class="m-card">
          <div class="m-lbl">⚡ Dribbles & Carries</div>
          <div class="m-val" id="mDrib">0 (0.00m)</div>
          <div class="m-sub" id="mDribSub">Match Total: 0 (0.00m)</div>
          <div class="p-bar"><div class="p-fill" id="barDrib" style="width:0%; background:var(--pl-gold);"></div></div>
        </div>

        <div class="m-card">
          <div class="m-lbl">🎯 Passes (Comp/Att)</div>
          <div class="m-val" id="mPass">0 / 0</div>
          <div class="m-sub" id="mPassSub">Match Total: 0 / 0</div>
          <div class="p-bar"><div class="p-fill" id="barPass" style="width:0%; background:var(--pl-green);"></div></div>
        </div>

        <div class="m-card">
          <div class="m-lbl">🛡️ Space & AI Best Pass</div>
          <div class="m-val" id="mPress">10.5m Space</div>
          <div class="m-sub" id="mPressSub">Best Pass: ID #14 (88% xP)</div>
          <div class="p-bar"><div class="p-fill" id="barPress" style="width:30%; background:#a855f7;"></div></div>
        </div>

        <div class="m-card">
          <div class="m-lbl">📈 Expected Threat (xT)</div>
          <div class="m-val" id="mXt">+0.000 xT</div>
          <div class="m-sub" id="mXtSub">Press Resist: 85/100</div>
          <div class="p-bar"><div class="p-fill" id="barXt" style="width:0%; background:var(--pl-pink);"></div></div>
        </div>

        <div class="m-card">
          <div class="m-lbl">🚀 Sprint Speed</div>
          <div class="m-val" id="mSpd">0.0 km/h</div>
          <div class="m-sub" id="mSpdSub">Peak: 0.0 km/h</div>
          <div class="p-bar"><div class="p-fill" id="barSpd" style="width:0%; background:#ef4444;"></div></div>
        </div>
      </div>

      <!-- 5-AXIS SVG RADAR PENTAGON -->
      <div class="pentagon-box">
        <div style="font-size:10px; font-weight:900; color:var(--pl-cyan); letter-spacing:0.6px;">
          🕸️ 5-AXIS ATTRIBUTE RADAR
        </div>
        <svg id="radarSvg" width="185" height="175" viewBox="0 0 200 190"></svg>
        <div style="font-size:10.5px; color:var(--pl-gold); font-weight:900; background:rgba(250,204,21,0.12); padding:3px 10px; border-radius:99px;" id="ovrBadge">
          OVR IMPACT: 88 / 99
        </div>
      </div>
    </div>

    <!-- AI MULTI-AGENT NARRATIVE -->
    <div class="ai-card">
      <div class="ai-header">
        <span id="agentTitle">🧠 Live AI Agent Breakdown</span>
        <span style="font-size:10px; color:var(--pl-cyan); font-weight:800;">MICROSOFT FOUNDRY SYNC</span>
      </div>
      <div id="agentBody">Click Play on the video to stream synchronized telemetry...</div>
    </div>

    <!-- INTERACTIVE PASS & RECEPTION TABLE -->
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
      <span style="font-size:11px; font-weight:900; color:var(--pl-cyan); text-transform:uppercase;" id="tableHeader">
        📋 Pass & Reception Events Involving Player ID #8:
      </span>
      <span style="font-size:10px; color:var(--pl-gold); font-weight:700;">🖱️ Click any row to jump video</span>
    </div>
    <table class="pl-table">
      <thead>
        <tr>
          <th>Ev</th>
          <th>Time</th>
          <th>Passer</th>
          <th>Receiver</th>
          <th>Dist / Speed</th>
          <th>Bypassed</th>
          <th>Diff (xP)</th>
          <th>Outcome</th>
        </tr>
      </thead>
      <tbody id="passTableBody"></tbody>
    </table>
  </div>
</div>

<script>
const DATA = {payload_json_str};
const PERSONA = {json.dumps(agent_mode)};
const LANG = {json.dumps(language)};

const video = document.getElementById('matchVideo');
const playBtn = document.getElementById('masterPlayBtn');
const autoChk = document.getElementById('autoFollowChk');
const manualSelect = document.getElementById('manualIdSelect');
const voiceToggle = document.getElementById('voiceToggle');
const canvas = document.getElementById('pitchCanvas');
const ctx = canvas.getContext('2d');
const waveCanvas = document.getElementById('waveCanvas');
const wCtx = waveCanvas.getContext('2d');

let lastSpokenEventId = -1;
let currentRenderedPlayers = [];

document.getElementById('gkSideBanner').textContent =
  `🥅 T${{DATA.left_gk_team}} GK (LEFT) ⬅️ PITCH ➡️ T${{DATA.right_gk_team}} GK (RIGHT) 🥅`;

document.getElementById('radarLeftGoalLbl').textContent =
  `🥅 LEFT: TEAM ${{DATA.left_gk_team}} GK (DEF THIRD T${{DATA.left_gk_team}})`;
document.getElementById('radarLeftGoalLbl').style.color =
  DATA.left_gk_team === 0 ? '#04f5ff' : '#ff2882';

document.getElementById('radarRightGoalLbl').textContent =
  `🥅 RIGHT: TEAM ${{DATA.right_gk_team}} GK (DEF THIRD T${{DATA.right_gk_team}})`;
document.getElementById('radarRightGoalLbl').style.color =
  DATA.right_gk_team === 0 ? '#04f5ff' : '#ff2882';

DATA.active_ids.forEach(pid => {{
  const opt = document.createElement('option');
  const t = DATA.player_totals[String(pid)].team;
  opt.value = pid;
  opt.textContent = `Player ID #${{pid}} (Team ${{t}})`;
  manualSelect.appendChild(opt);
}});

const evContainer = document.getElementById('eventButtons');
DATA.events.forEach(ev => {{
  const btn = document.createElement('button');
  btn.className = 'ev-btn';
  btn.id = `evBtn_${{ev.event_id}}`;
  btn.textContent = `E${{ev.event_id}}: #${{ev.passer}}→#${{ev.receiver}}`;
  btn.onclick = () => {{
    video.currentTime = Math.max(0, ev.sec - 0.12);
    updateUI();
  }};
  evContainer.appendChild(btn);
}});

// Click-to-seek on Control vs Chaos waveform canvas
waveCanvas.addEventListener('click', (e) => {{
  const rect = waveCanvas.getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  const targetFrame = Math.round(ratio * DATA.max_frame);
  video.currentTime = Math.max(0, (targetFrame - 1) / DATA.fps);
  updateUI();
}});

// Interactive Click on 2D Pitch Radar to select any Player ID dot!
canvas.addEventListener('click', (e) => {{
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const clickX = (e.clientX - rect.left) * scaleX;
  const clickY = (e.clientY - rect.top) * scaleY;

  let hitPid = null;
  let minD = 18.0;
  currentRenderedPlayers.forEach(p => {{
    const d = Math.hypot(p.cx - clickX, p.cy - clickY);
    if (d < minD && DATA.active_ids.includes(p.pid)) {{
      minD = d;
      hitPid = p.pid;
    }}
  }});

  if (hitPid !== null) {{
    autoChk.checked = false;
    manualSelect.value = hitPid;
    updateUI();
  }}
}});

function togglePlay() {{
  if (video.paused) video.play();
  else video.pause();
}}

function stepFrames(deltaFrames) {{
  video.pause();
  const nextTime = Math.max(0, Math.min(DATA.max_frame / DATA.fps, video.currentTime + (deltaFrames / DATA.fps)));
  video.currentTime = nextTime;
  updateUI();
}}

function setSpeed(rate) {{
  video.playbackRate = rate;
  document.getElementById('spd05').classList.toggle('active-spd', rate === 0.5);
  document.getElementById('spd10').classList.toggle('active-spd', rate === 1.0);
}}

function onManualSelect() {{
  autoChk.checked = false;
  updateUI();
}}

video.addEventListener('play', () => {{
  playBtn.textContent = '⏸️ PAUSE BROADCAST';
  playBtn.style.background = 'linear-gradient(135deg, #facc15 0%, #eab308 100%)';
}});

video.addEventListener('pause', () => {{
  playBtn.textContent = '▶️ PLAY BROADCAST';
  playBtn.style.background = 'linear-gradient(135deg, #00ff85 0%, #059669 100%)';
}});

video.addEventListener('timeupdate', updateUI);
video.addEventListener('seeked', updateUI);

function getNearestSampleFrame(frame) {{
  if (frame >= DATA.max_frame) return String(DATA.max_frame);
  const snapped = Math.max(1, Math.floor((frame - 1) / 4) * 4 + 1);
  return String(snapped);
}}

function drawAttributePentagon(scores) {{
  const svg = document.getElementById('radarSvg');
  const cx = 100, cy = 98, R = 64;
  const labels = ["PROG", "RET", "WORK", "DRIB", "PACE"];
  let html = `
    <defs>
      <radialGradient id="pentGrad" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stop-color="#04f5ff" stop-opacity="0.55"/>
        <stop offset="100%" stop-color="#00ff85" stop-opacity="0.22"/>
      </radialGradient>
    </defs>
  `;

  [0.35, 0.68, 1.0].forEach(scale => {{
    let pts = [];
    for (let i = 0; i < 5; i++) {{
      const ang = -Math.PI / 2 + (i * 2 * Math.PI) / 5;
      pts.push(`${{cx + R * scale * Math.cos(ang)}},${{cy + R * scale * Math.sin(ang)}}`);
    }}
    html += `<polygon points="${{pts.join(' ')}}" fill="none" stroke="rgba(255,255,255,0.14)" stroke-width="1"/>`;
  }});

  let dataPts = [];
  let vertexDots = '';
  for (let i = 0; i < 5; i++) {{
    const ang = -Math.PI / 2 + (i * 2 * Math.PI) / 5;
    const x2 = cx + R * Math.cos(ang);
    const y2 = cy + R * Math.sin(ang);
    html += `<line x1="${{cx}}" y1="${{cy}}" x2="${{x2}}" y2="${{y2}}" stroke="rgba(255,255,255,0.14)" stroke-width="1"/>`;

    const lx = cx + (R + 21) * Math.cos(ang);
    const ly = cy + (R + 16) * Math.sin(ang) + 3;
    html += `<text x="${{lx}}" y="${{ly}}" fill="#e2e8f0" font-size="9.5" font-weight="800" text-anchor="middle">${{labels[i]}} (${{scores[i]}})</text>`;

    const valR = (Math.max(30, scores[i]) / 100.0) * R;
    const vx = cx + valR * Math.cos(ang);
    const vy = cy + valR * Math.sin(ang);
    dataPts.push(`${{vx}},${{vy}}`);
    vertexDots += `<circle cx="${{vx}}" cy="${{vy}}" r="3.2" fill="#04f5ff" stroke="#000" stroke-width="1"/>`;
  }}

  html += `<polygon points="${{dataPts.join(' ')}}" fill="url(#pentGrad)" stroke="#00ff85" stroke-width="2.4"/>`;
  html += vertexDots;
  svg.innerHTML = html;

  const ovr = Math.round(scores.reduce((a, b) => a + b, 0) / 5);
  document.getElementById('ovrBadge').textContent = `OVR IMPACT: ${{ovr}} / 99`;
}}

function drawControlWaveform(currentFrame) {{
  const W = waveCanvas.width, H = waveCanvas.height;
  wCtx.clearRect(0, 0, W, H);
  const midY = H / 2;

  // Subtle background grid
  wCtx.strokeStyle = 'rgba(255,255,255,0.15)';
  wCtx.lineWidth = 1;
  wCtx.beginPath();
  wCtx.moveTo(0, midY); wCtx.lineTo(W, midY);
  wCtx.stroke();

  const wave = DATA.ctrl_wave || [];
  if (wave.length > 1) {{
    // Shaded area under curve
    const grad = wCtx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, 'rgba(0, 255, 133, 0.38)');
    grad.addColorStop(0.5, 'rgba(4, 245, 255, 0.05)');
    grad.addColorStop(1, 'rgba(255, 40, 130, 0.38)');

    wCtx.beginPath();
    wCtx.moveTo(0, midY);
    wave.forEach(pt => {{
      const x = (pt[0] / DATA.max_frame) * W;
      const y = midY - (pt[1] / 100.0) * (midY - 6);
      wCtx.lineTo(x, y);
    }});
    wCtx.lineTo(W, midY);
    wCtx.closePath();
    wCtx.fillStyle = grad;
    wCtx.fill();

    // Waveform stroke line
    wCtx.beginPath();
    wave.forEach((pt, idx) => {{
      const x = (pt[0] / DATA.max_frame) * W;
      const y = midY - (pt[1] / 100.0) * (midY - 6);
      if (idx === 0) wCtx.moveTo(x, y);
      else wCtx.lineTo(x, y);
    }});
    wCtx.strokeStyle = '#00ff85';
    wCtx.lineWidth = 2.2;
    wCtx.stroke();
  }}

  // Pass event markers on waveform
  DATA.events.forEach(ev => {{
    const ex = (ev.frame / DATA.max_frame) * W;
    wCtx.fillStyle = ev.outcome.includes('completed') ? '#04f5ff' : '#ff2882';
    wCtx.beginPath();
    wCtx.arc(ex, midY, 3.6, 0, Math.PI * 2);
    wCtx.fill();
    wCtx.strokeStyle = '#000';
    wCtx.lineWidth = 1;
    wCtx.stroke();
  }});

  // Current playhead bar & glowing knob
  const playX = (currentFrame / DATA.max_frame) * W;
  wCtx.strokeStyle = '#facc15';
  wCtx.lineWidth = 2.5;
  wCtx.beginPath();
  wCtx.moveTo(playX, 0); wCtx.lineTo(playX, H);
  wCtx.stroke();

  wCtx.fillStyle = '#facc15';
  wCtx.beginPath();
  wCtx.arc(playX, midY, 4.5, 0, Math.PI * 2);
  wCtx.fill();
}}

function drawPitchMinimap(frame, sampleKey, activePid, pressObj) {{
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // Lush striped Premier League turf
  for (let i = 0; i < 12; i++) {{
    ctx.fillStyle = i % 2 === 0 ? '#114224' : '#15522d';
    ctx.fillRect((W / 12) * i, 0, W / 12, H);
  }}

  // Tactical Third Divider Lines & Labels
  ctx.strokeStyle = 'rgba(255,255,255,0.24)';
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(W / 3, 12); ctx.lineTo(W / 3, H - 12);
  ctx.moveTo((W * 2) / 3, 12); ctx.lineTo((W * 2) / 3, H - 12);
  ctx.stroke();
  ctx.setLineDash([]);

  // Outer pitch lines, penalty boxes, and center circle
  ctx.strokeStyle = 'rgba(255,255,255,0.7)';
  ctx.lineWidth = 1.6;
  ctx.strokeRect(16, 12, W - 32, H - 24);
  ctx.strokeRect(16, H / 2 - 48, 68, 96);
  ctx.strokeRect(W - 84, H / 2 - 48, 68, 96);
  ctx.beginPath();
  ctx.moveTo(W / 2, 12); ctx.lineTo(W / 2, H - 12);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(W / 2, H / 2, 30, 0, Math.PI * 2);
  ctx.stroke();

  const mapX = x => 16 + (x / 1920.0) * (W - 32);
  const mapY = y => 12 + ((y - 260.0) / 780.0) * (H - 24);

  // 1. Draw Tactical Spotlight / Voronoi Space Glow around Carrier & Best Pass Target
  if (pressObj && pressObj.carrier_xy) {{
    const cx = mapX(pressObj.carrier_xy[0]);
    const cy = mapY(pressObj.carrier_xy[1]);
    const radGrad = ctx.createRadialGradient(cx, cy, 2, cx, cy, 38);
    radGrad.addColorStop(0, 'rgba(250, 204, 21, 0.35)');
    radGrad.addColorStop(1, 'rgba(250, 204, 21, 0)');
    ctx.fillStyle = radGrad;
    ctx.beginPath();
    ctx.arc(cx, cy, 38, 0, Math.PI * 2);
    ctx.fill();
  }}

  // 2. Draw Red Pressing Threat Lines from opposing defenders within 12m
  if (pressObj && pressObj.carrier_xy && pressObj.press_lines) {{
    const cx = mapX(pressObj.carrier_xy[0]);
    const cy = mapY(pressObj.carrier_xy[1]);
    pressObj.press_lines.forEach(opp => {{
      const ox = mapX(opp[0]), oy = mapY(opp[1]);
      ctx.strokeStyle = 'rgba(255, 40, 130, 0.9)';
      ctx.lineWidth = 2.2;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(ox, oy); ctx.lineTo(cx, cy);
      ctx.stroke();
      ctx.setLineDash([]);
    }});
  }}

  // 3. Draw Counterfactual AI Best Pass Option vector in glowing Cyan dashed line
  if (pressObj && pressObj.carrier_xy && pressObj.best_opt) {{
    const cx = mapX(pressObj.carrier_xy[0]);
    const cy = mapY(pressObj.carrier_xy[1]);
    const tx = mapX(pressObj.best_opt.x);
    const ty = mapY(pressObj.best_opt.y);

    const optGrad = ctx.createRadialGradient(tx, ty, 2, tx, ty, 24);
    optGrad.addColorStop(0, 'rgba(4, 245, 255, 0.38)');
    optGrad.addColorStop(1, 'rgba(4, 245, 255, 0)');
    ctx.fillStyle = optGrad;
    ctx.beginPath();
    ctx.arc(tx, ty, 24, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = '#04f5ff';
    ctx.lineWidth = 2.4;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(cx, cy); ctx.lineTo(tx, ty);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.beginPath();
    ctx.arc(tx, ty, 10, 0, Math.PI * 2);
    ctx.strokeStyle = '#04f5ff';
    ctx.lineWidth = 1.8;
    ctx.stroke();
  }}

  // 4. Draw actual pass vectors up to current frame
  DATA.events.filter(e => e.frame <= frame).forEach(ev => {{
    const x1 = mapX(ev.px), y1 = mapY(ev.py);
    const x2 = mapX(ev.rx), y2 = mapY(ev.ry);
    ctx.strokeStyle = ev.outcome.includes('completed') ? 'rgba(0, 255, 133, 0.58)' : 'rgba(255, 40, 130, 0.68)';
    ctx.lineWidth = 1.9;
    ctx.beginPath();
    ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
    ctx.stroke();
  }});

  // 5. Draw active player's movement trail in Gold
  const trail = (DATA.player_trails[String(activePid)] || []).filter(pt => pt[0] <= frame);
  if (trail.length > 1) {{
    ctx.strokeStyle = '#facc15';
    ctx.lineWidth = 2.8;
    ctx.beginPath();
    trail.forEach((pt, idx) => {{
      const tx = mapX(pt[1]), ty = mapY(pt[2]);
      if (idx === 0) ctx.moveTo(tx, ty);
      else ctx.lineTo(tx, ty);
    }});
    ctx.stroke();
  }}

  // 6. Draw all tracked players & store coordinates for click-to-inspect
  currentRenderedPlayers = [];
  const rFrame = DATA.radar_frames[sampleKey];
  if (rFrame && rFrame.p) {{
    rFrame.p.forEach(p => {{
      const pid = p[0], px = mapX(p[1]), py = mapY(p[2]), team = p[3];
      const isHero = pid === activePid;
      currentRenderedPlayers.push({{ pid, cx: px, cy: py, team }});

      if (isHero) {{
        ctx.beginPath();
        ctx.arc(px, py, 13, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(250, 204, 21, 0.35)';
        ctx.fill();
        ctx.strokeStyle = '#facc15';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }}

      ctx.beginPath();
      ctx.arc(px, py, isHero ? 6.8 : 4.8, 0, Math.PI * 2);
      ctx.fillStyle = isHero ? '#facc15' : (team === 0 ? '#04f5ff' : '#ff2882');
      ctx.fill();
      ctx.strokeStyle = '#09000c';
      ctx.lineWidth = 1.4;
      ctx.stroke();

      if (isHero || DATA.active_ids.includes(pid)) {{
        ctx.fillStyle = isHero ? '#facc15' : '#ffffff';
        ctx.font = isHero ? '900 10.5px Inter, sans-serif' : '700 8.5px Inter, sans-serif';
        ctx.fillText(`#${{pid}}`, px - 8, py - 9);
      }}
    }});

    if (rFrame.b) {{
      const bx = mapX(rFrame.b[0]), by = mapY(rFrame.b[1]);
      ctx.beginPath();
      ctx.arc(bx, by, 4.2, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
      ctx.strokeStyle = '#facc15';
      ctx.lineWidth = 1.8;
      ctx.stroke();
    }}
  }}
}}

function maybeSpeakLiveEvent(sec, frame) {{
  if (!voiceToggle.checked || video.paused || !('speechSynthesis' in window)) return;
  const hitEv = DATA.events.find(e => Math.abs(e.sec - sec) <= 0.35);
  if (hitEv && hitEv.event_id !== lastSpokenEventId) {{
    lastSpokenEventId = hitEv.event_id;
    window.speechSynthesis.cancel();
    const isComp = hitEv.outcome.includes('completed');
    const txt = isComp
      ? `Pass completed for Team ${{hitEv.passer_team}} from Player I D ${{hitEv.passer}} to I D ${{hitEv.receiver}}, bypassing ${{hitEv.bypassed}} defenders.`
      : `Turnover! Ball switches from Team ${{hitEv.passer_team}} I D ${{hitEv.passer}} to Team ${{hitEv.receiver_team}} I D ${{hitEv.receiver}}.`;
    const utter = new SpeechSynthesisUtterance(txt);
    utter.rate = 1.12;
    window.speechSynthesis.speak(utter);
  }}
}}

function buildNarrative(pid, team, sec, frame, live, totals, pressObj, latestEvStr, teamStats, activeZone) {{
  const ctrl = pressObj.ctrl;
  const stateWord = ctrl < 0
    ? `High-Chaos Pressing / Transition (${{pressObj.pressers}} Opponents within 12m, Nearest: ${{pressObj.nearest_m}}m)`
    : `Unpressed Tactical Control (Nearest Opponent: ${{pressObj.nearest_m}}m away, ${{pressObj.pressers}} Pressing)`;
  const teamPoss = team === 0 ? teamStats.t0_poss : teamStats.t1_poss;
  const teamPassStr = team === 0 ? teamStats.t0_passes : teamStats.t1_passes;
  const optStr = pressObj.best_opt
    ? `AI Best Pass Option: <b>Player ID #${{pressObj.best_opt.id}}</b> (${{pressObj.best_opt.dist_m}}m, open by ${{pressObj.best_opt.space_m}}m, <b>${{pressObj.best_opt.xp}}% xP</b>)`
    : `Progressive Carry Lane Open`;

  if (LANG === "Swahili") {{
    return `• <b>Hali ya Mchezo (${{sec.toFixed(1)}}s):</b> <b>Player ID #${{pid}} (Timu ${{team}})</b> yuko kwenye <i>${{activeZone}}</i> — <b>${{pressObj.label}}</b> (Index: <b>${{ctrl >= 0 ? '+' : ''}}${{ctrl.toFixed(1)}}</b>).<br>` +
           `• <b>Takwimu & AI Chaguo Bora:</b> Umbali <b>${{live.dist_m.toFixed(1)}}m</b>, kukokota mpira <b>${{live.drib}} (${{live.carry_m.toFixed(2)}}m)</b>, xT <b>+${{live.xt.toFixed(3)}}</b> | ${{optStr}}.<br>` +
           `• <b>Tukio la Karibuni:</b> ${{latestEvStr}}`;
  }}
  if (LANG === "Spanish") {{
    return `• <b>Estado Espacial (${{sec.toFixed(1)}}s):</b> <b>Player ID #${{pid}} (Equipo ${{team}})</b> en <i>${{activeZone}}</i> — <b>${{pressObj.label}}</b> (Índice: <b>${{ctrl >= 0 ? '+' : ''}}${{ctrl.toFixed(1)}}</b>).<br>` +
           `• <b>Telemetría & IA Opción Óptima:</b> Distancia <b>${{live.dist_m.toFixed(1)}}m</b>, <b>${{live.drib}} regates (${{live.carry_m.toFixed(2)}}m)</b>, xT <b>+${{live.xt.toFixed(3)}}</b> | ${{optStr}}.<br>` +
           `• <b>Última Acción:</b> ${{latestEvStr}}`;
  }}
  if (LANG === "French") {{
    return `• <b>État Spatial (${{sec.toFixed(1)}}s):</b> <b>Player ID #${{pid}} (Équipe ${{team}})</b> dans <i>${{activeZone}}</i> — <b>${{pressObj.label}}</b> (Indice: <b>${{ctrl >= 0 ? '+' : ''}}${{ctrl.toFixed(1)}}</b>).<br>` +
           `• <b>Télémétrie & IA Meilleure Passe:</b> Distance <b>${{live.dist_m.toFixed(1)}}m</b>, <b>${{live.drib}} dribbles (${{live.carry_m.toFixed(2)}}m)</b>, xT <b>+${{live.xt.toFixed(3)}}</b> | ${{optStr}}.<br>` +
           `• <b>Dernière Action:</b> ${{latestEvStr}}`;
  }}

  if (PERSONA === "Fan Mode") {{
    return `• <b>🔥 Spotlight on Player ID #${{pid}} (${{sec.toFixed(1)}}s):</b> Operating in <b>${{activeZone}}</b> under <b>${{pressObj.label}}</b> (Team ${{team}} Possession: <b>${{teamPoss.toFixed(1)}}%</b>)!<br>` +
           `• <b>⚡ Dribble Impact & AI Vision:</b> Covered <b>${{live.dist_m.toFixed(1)}} m</b> with <b>${{live.drib}} dribble(s) (${{live.carry_m.toFixed(2)}} m)</b> and <b>+${{live.xt.toFixed(3)}} xT</b>! ${{optStr}}.<br>` +
           `• <b>🎯 Latest Pitch Action:</b> ${{latestEvStr}}`;
  }}
  if (PERSONA === "Broadcast Mode") {{
    return `• <b>🎙️ [LOWER-THIRD BROADCAST FEED]:</b> <code>ID #${{pid}} (TEAM ${{team}}) | ${{activeZone.toUpperCase()}} | SPACE: ${{pressObj.nearest_m}}m | CTRL: ${{ctrl >= 0 ? '+' : ''}}${{ctrl.toFixed(1)}}</code><br>` +
           `• <b>📺 Commentator Cue (${{sec.toFixed(1)}}s):</b> <i>"Player ID #${{pid}} on the ball in ${{activeZone}} with ${{pressObj.nearest_m}}m of separation—${{optStr}}."</i><br>` +
           `• <b>⚡ Team Rhythm:</b> Team ${{team}} at ${{teamPassStr}} completed passes (${{teamPoss.toFixed(1)}}% possession). ${{latestEvStr}}`;
  }}
  return `• <b>🧠 Spatial Control vs. Chaos (Frame ${{frame}} / ${{sec.toFixed(1)}}s):</b> <b>${{stateWord}}</b> (Control Index: <b>${{ctrl >= 0 ? '+' : ''}}${{ctrl.toFixed(1)}}</b>) with <b>Player ID #${{pid}}</b> in <b>${{activeZone}}</b>.<br>` +
         `• <b>🎯 Counterfactual AI Vision & Telemetry:</b> ${{optStr}} | Covered <b>${{live.dist_m.toFixed(1)}} m</b>, logging <b>${{live.drib}} dribble(s) (${{live.carry_m.toFixed(2)}} m)</b> and <b>+${{live.xt.toFixed(3)}} xT</b>.<br>` +
         `• <b>🔗 Pass Progression:</b> Completed <b>${{live.comp}}/${{live.att}} passes</b> and <b>${{live.rec}} reception(s)</b>. ${{latestEvStr}}`;
}}

function updateUI() {{
  const sec = video.currentTime || 0.0;
  const frame = Math.min(DATA.max_frame, Math.max(1, Math.round(sec * DATA.fps) + 1));
  const sampleKey = getNearestSampleFrame(frame);

  let activePid = DATA.carrier_by_frame[String(frame)] || 8;
  if (!DATA.active_ids.includes(activePid)) activePid = DATA.active_ids[0];

  if (autoChk.checked) {{
    manualSelect.value = activePid;
  }} else {{
    activePid = parseInt(manualSelect.value, 10);
  }}

  const pidKey = String(activePid);
  const totals = DATA.player_totals[pidKey];
  const live = (DATA.player_timelines[pidKey] && DATA.player_timelines[pidKey][sampleKey]) || {{
    att: 0, comp: 0, rec: 0, drib: 0, carry_m: 0.0, touches: 0, dist_m: 0.0, cur_spd: 0.0, max_spd: 0.0, xt: 0.0, zone: `Middle Third (Team ${{totals.team}})`
  }};

  const activeZone = autoChk.checked
    ? (DATA.carrier_zone_by_frame[String(frame)] || live.zone)
    : live.zone;

  const mins = String(Math.floor(sec / 60)).padStart(2, '0');
  const remS = (sec % 60).toFixed(1).padStart(4, '0');
  document.getElementById('topClock').textContent = `${{mins}}:${{remS}} • F${{frame}}`;

  const teamStats = DATA.team_stats_by_sample[sampleKey] || {{
    t0_poss: 84.0, t1_poss: 16.0, t0_passes: "0/0", t1_passes: "0/0", t0_att_f: 0, t1_att_f: 0
  }};
  document.getElementById('possT0Lbl').textContent = `TEAM 0 (CYAN): ${{teamStats.t0_poss.toFixed(1)}}% POSS (${{teamStats.t0_passes}} PASS)`;
  document.getElementById('possT1Lbl').textContent = `TEAM 1 (PINK): ${{teamStats.t1_poss.toFixed(1)}}% POSS (${{teamStats.t1_passes}} PASS)`;
  document.getElementById('possBarT0').style.width = `${{teamStats.t0_poss}}%`;

  const pressObj = DATA.pressure_by_sample[sampleKey] || {{
    ctrl: 55.0, press: 22.0, nearest_m: 11.2, pressers: 0, label: "Unpressed Control (11.2m Space)", carrier_xy: [960, 540], press_lines: [], best_opt: null
  }};

  // Highlight active event button if near timestamp
  DATA.events.forEach(ev => {{
    const b = document.getElementById(`evBtn_${{ev.event_id}}`);
    if (b) b.classList.toggle('active-ev', Math.abs(ev.sec - sec) <= 0.6);
  }});

  document.getElementById('dossierStatus').innerHTML = autoChk.checked
    ? `<span class="pulse-dot"></span>AUTO-FOLLOWING VIDEO BALL CARRIER @ ${{sec.toFixed(1)}}s (FRAME ${{frame}})`
    : `📌 MANUALLY INSPECTING PLAYER ID @ ${{sec.toFixed(1)}}s (FRAME ${{frame}})`;

  document.getElementById('dossierPlayerId').textContent = `Player ID #${{activePid}}`;
  const pill = document.getElementById('dossierTeamPill');
  pill.textContent = `Team ${{totals.team}}`;
  pill.style.background = totals.team === 0 ? '#04f5ff' : '#ff2882';
  pill.style.color = totals.team === 0 ? '#000' : '#fff';
  document.getElementById('dossierZone').textContent = `📍 ${{activeZone}}`;

  document.getElementById('ltHeader').textContent = `LIVE BROADCAST TELEMETRY • FRAME ${{frame}} (${{sec.toFixed(2)}}s)`;
  document.getElementById('ltMain').textContent = `ACTIVE CARRIER: PLAYER ID #${{activePid}} (TEAM ${{totals.team}})`;
  document.getElementById('ltSub').textContent = `Zone: ${{activeZone}} | Nearest Opp: ${{pressObj.nearest_m}}m (${{pressObj.pressers}} Pressing) | Dribbles: ${{live.drib}} (${{live.carry_m.toFixed(2)}}m)`;

  document.getElementById('ltPressState').textContent = pressObj.label.toUpperCase();
  const ltCtrl = document.getElementById('ltCtrl');
  ltCtrl.textContent = `${{pressObj.ctrl >= 0 ? '+' : ''}}${{pressObj.ctrl.toFixed(1)}}`;
  ltCtrl.style.color = pressObj.ctrl >= 0 ? '#00ff85' : '#ff2882';

  document.getElementById('mDist').textContent = `${{live.dist_m.toFixed(1)}} m`;
  document.getElementById('mDistSub').textContent = `Match Total: ${{totals.total_dist_m.toFixed(1)}} m`;
  document.getElementById('barDist').style.width = `${{Math.min(100, (live.dist_m / Math.max(1, totals.total_dist_m)) * 100)}}%`;

  document.getElementById('mDrib').textContent = `${{live.drib}} (${{live.carry_m.toFixed(2)}}m)`;
  document.getElementById('mDribSub').textContent = `Match Total: ${{totals.total_drib}} (${{totals.total_carry_m.toFixed(2)}}m)`;
  document.getElementById('barDrib').style.width = `${{Math.min(100, (live.carry_m / Math.max(0.5, totals.total_carry_m)) * 100)}}%`;

  document.getElementById('mPass').textContent = `${{live.comp}} / ${{live.att}}`;
  document.getElementById('mPassSub').textContent = `Match Total: ${{totals.total_comp}} / ${{totals.total_att}}`;
  document.getElementById('barPass').style.width = `${{totals.total_att > 0 ? Math.min(100, (live.comp / totals.total_att) * 100) : 100}}%`;

  document.getElementById('mPress').textContent = `${{pressObj.nearest_m.toFixed(1)}}m Space`;
  document.getElementById('mPressSub').textContent = pressObj.best_opt
    ? `AI Best Target: ID #${{pressObj.best_opt.id}} (${{pressObj.best_opt.xp}}% xP)`
    : `${{pressObj.pressers}} Opponent(s) within 12m`;
  document.getElementById('barPress').style.width = `${{Math.min(100, pressObj.press)}}%`;
  document.getElementById('barPress').style.background = pressObj.pressers >= 2 ? '#ff2882' : (pressObj.pressers === 1 ? '#facc15' : '#00ff85');

  document.getElementById('mXt').textContent = `+${{live.xt.toFixed(3)}} xT`;
  document.getElementById('mXtSub').textContent = `Press Resistance: ${{totals.press_res}}/100`;
  document.getElementById('barXt').style.width = `${{Math.min(100, (live.xt / Math.max(0.05, totals.xt)) * 100)}}%`;

  document.getElementById('mSpd').textContent = `${{live.cur_spd.toFixed(1)}} km/h`;
  document.getElementById('mSpdSub').textContent = `Peak Speed: ${{totals.peak_spd.toFixed(1)}} km/h`;
  document.getElementById('barSpd').style.width = `${{Math.min(100, (live.cur_spd / 34.5) * 100)}}%`;

  drawAttributePentagon(totals.radar);
  drawPitchMinimap(frame, sampleKey, activePid, pressObj);
  drawControlWaveform(frame);
  maybeSpeakLiveEvent(sec, frame);

  const pastEvents = DATA.events.filter(e => e.frame <= frame);
  let latestEvStr = "Initiating progressive ball carry.";
  if (pastEvents.length > 0) {{
    const lastEv = pastEvents[pastEvents.length - 1];
    latestEvStr = `Event #${{lastEv.event_id}} (${{lastEv.sec.toFixed(1)}}s): ID #${{lastEv.passer}} (T${{lastEv.passer_team}}) → ID #${{lastEv.receiver}} (T${{lastEv.receiver_team}}) (${{lastEv.travel_m}}m, bypassed ${{lastEv.bypassed}} opp, xP: ${{lastEv.xp}}%).`;
  }}

  document.getElementById('agentTitle').textContent = `🤖 ${{PERSONA}} (${{LANG}}) — Player ID #${{activePid}} (Team ${{totals.team}})`;
  document.getElementById('agentBody').innerHTML = buildNarrative(
    activePid, totals.team, sec, frame, live, totals, pressObj, latestEvStr, teamStats, activeZone
  );

  document.getElementById('tableHeader').textContent = `📋 Pass & Reception Events Involving Player ID #${{activePid}} (Team ${{totals.team}}):`;
  const pEvents = DATA.events.filter(e => e.passer === activePid || e.receiver === activePid);
  const tbody = document.getElementById('passTableBody');
  tbody.innerHTML = '';
  pEvents.forEach(ev => {{
    const tr = document.createElement('tr');
    const isPast = ev.frame <= frame;
    const isCurrentEv = Math.abs(ev.sec - sec) <= 0.65;
    if (isCurrentEv) tr.className = 'active-pass-row';
    tr.style.opacity = isPast ? '1.0' : '0.45';
    tr.onclick = () => {{
      video.currentTime = Math.max(0, ev.sec - 0.12);
      updateUI();
    }};
    const isComp = ev.outcome.includes('completed');
    const badgeClass = isComp ? 'tag-ok' : 'tag-bad';
    const badgeText = isComp ? 'Completed' : 'Turnover';
    tr.innerHTML = `
      <td>#${{ev.event_id}} ${{isPast ? '✅' : '⏳'}}</td>
      <td style="font-family:'JetBrains Mono',monospace;">${{ev.sec.toFixed(2)}}s</td>
      <td><b>ID #${{ev.passer}}</b> <span style="font-size:10px; color:#94a3b8;">(T${{ev.passer_team}})</span></td>
      <td><b>ID #${{ev.receiver}}</b> <span style="font-size:10px; color:#94a3b8;">(T${{ev.receiver_team}})</span></td>
      <td>${{ev.travel_m}}m (${{ev.speed_kmh}} km/h)</td>
      <td style="color:var(--pl-gold); font-weight:800;">${{ev.bypassed}} Opp</td>
      <td>${{ev.diff}} (${{ev.xp}}% xP)</td>
      <td><span class="${{badgeClass}}">${{badgeText}}</span></td>
    `;
    tbody.appendChild(tr);
  }});
}}

updateUI();
</script>
</body>
</html>
"""

components.html(html_component, height=960, scrolling=True)