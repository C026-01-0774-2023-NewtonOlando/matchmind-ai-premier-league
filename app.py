import base64
import json
import math
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="MatchMind AI — Premier League Broadcast Hub",
    layout="wide",
    initial_sidebar_state="expanded",
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


def get_pitch_zone(x_px: float, width_px: float = 1920.0) -> str:
    ratio = x_px / max(1.0, width_px)
    if ratio < 0.33:
        return "Defensive Third"
    elif ratio < 0.66:
        return "Middle Third"
    return "Attacking Third"


def compute_xt_value(x_px: float, y_px: float) -> float:
    """Computes a Premier League-style Expected Threat (xT) zone value (0.01 to 0.35)."""
    nx = max(0.0, min(1.0, x_px / 1920.0))
    ny = max(0.0, min(1.0, (y_px - 280.0) / 720.0))
    centrality = 1.0 - abs(ny - 0.5) * 1.4
    progression = abs(nx - 0.5) * 1.8
    return round(float(np.clip(0.02 + 0.18 * progression + 0.12 * centrality, 0.01, 0.36)), 3)


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
def build_premier_league_telemetry():
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

    team_lookup = {
        int(r["player_track_id"]): (int(r["team_id"]) if pd.notna(r["team_id"]) else 0)
        for _, r in dossiers.iterrows()
    }
    player_rows["team_id"] = player_rows["track_id"].map(
        lambda x: team_lookup.get(int(x), 0)
    )

    # Replicate exact event_detector.py ball carrier HUD logic + store 2D radar coordinates
    last_known_ball = None
    candidate_holder_id = None
    candidate_holder_streak = 0
    confirmed_holder_id = None
    last_confirmed_id = None

    hud_carrier_by_frame = {}
    raw_carrier_by_frame = {}
    radar_frames = {}
    team0_poss_frames = 0
    team1_poss_frames = 0
    poss_by_sample = {}

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

        c_team = team_lookup.get(cur_c, 0)
        if c_team == 0:
            team0_poss_frames += 1
        else:
            team1_poss_frames += 1

        # Sample 2D radar positions every 4 frames for smooth HTML5 canvas rendering
        if f % 4 == 1 or f == max_frame:
            tot_p = max(1, team0_poss_frames + team1_poss_frames)
            poss_by_sample[str(f)] = round(100.0 * team0_poss_frames / tot_p, 1)

            pts = []
            for _, pr in p_sub.iterrows():
                ptid = int(pr["track_id"])
                pts.append(
                    [
                        ptid,
                        round(float((pr["x1"] + pr["x2"]) / 2.0), 1),
                        round(float(pr["y2"]), 1),
                        team_lookup.get(ptid, 0),
                    ]
                )
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

        total_dist_m = round(float(p_df["clean_step_px"].sum() * PX_TO_METERS), 1)
        total_carry_m = round(float(p_df["carry_step_px"].sum() * PX_TO_METERS), 2)
        total_touches = int(p_df["is_carrier"].sum())
        peak_spd = round(float(p_df["speed_kmh"].max()), 1) if not p_df.empty else 0.0
        total_att = int((passes["passer_id"] == pid).sum())
        total_comp = int(
            (
                (passes["passer_id"] == pid)
                & (passes["outcome_candidate"].str.contains("completed", na=False))
            ).sum()
        )
        total_rec = int((passes["receiver_id"] == pid).sum())

        # Compute Expected Threat (xT) generated & Press Resistance (0-100)
        xt_total = round(0.045 * total_comp + 0.025 * max_drib_cap + 0.008 * total_carry_m, 3)
        press_res = int(
            np.clip(
                68 + (12 if total_comp == total_att and total_att > 0 else -10) + max_drib_cap * 6,
                45,
                98,
            )
        )

        # 5-Axis Radar Ratings (0-99 Premier League / EA FC style)
        prog_rating = int(np.clip(62 + total_comp * 14 + total_carry_m * 3.5, 50, 97))
        ret_rating = int(95 if (total_att > 0 and total_comp == total_att) else (84 if total_rec > 0 else 64))
        work_rating = int(np.clip(55 + (total_dist_m / 95.0) * 42, 52, 98))
        drib_rating = int(np.clip(58 + max_drib_cap * 11 + total_carry_m * 3.2, 50, 96))
        pace_rating = int(np.clip((peak_spd / 35.0) * 96, 55, 97))

        player_totals[str(pid)] = {
            "pid": pid,
            "team": team_lookup.get(pid, 0),
            "total_dist_m": max(0.1, total_dist_m),
            "total_carry_m": max(0.0, total_carry_m),
            "total_drib": max_drib_cap,
            "total_touches": total_touches,
            "peak_spd": peak_spd,
            "total_att": total_att,
            "total_comp": total_comp,
            "total_rec": total_rec,
            "xt": xt_total,
            "press_res": press_res,
            "radar": [prog_rating, ret_rating, work_rating, drib_rating, pace_rating],
        }

        # Downsampled trail points for 2D canvas drawing
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

            att = int((sub_ev["passer_id"] == pid).sum())
            comp = int(
                (
                    (sub_ev["passer_id"] == pid)
                    & (sub_ev["outcome_candidate"].str.contains("completed", na=False))
                ).sum()
            )
            rec = int((sub_ev["receiver_id"] == pid).sum())

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
                    "zone": "Middle Third",
                }
            else:
                touches = int(sub_p["is_carrier"].sum())
                carry_m = round(float(sub_p["carry_step_px"].sum() * PX_TO_METERS), 2)
                dist_m = round(float(sub_p["clean_step_px"].sum() * PX_TO_METERS), 1)
                max_spd = round(float(sub_p["speed_kmh"].max()), 1)
                cur_spd = round(float(sub_p.iloc[-1]["speed_kmh"]), 1)
                zone = get_pitch_zone(float(sub_p.iloc[-1]["foot_x"]), WIDTH)

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

                live_xt = round(0.045 * comp + 0.025 * drib + 0.008 * carry_m, 3)

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

        events_list.append(
            {
                "event_id": int(ev["event_id"]),
                "sec": round(float(ev["timestamp_seconds"]), 2),
                "frame": f_ev,
                "passer": p_id,
                "receiver": r_id,
                "px": px,
                "py": py,
                "rx": rx,
                "ry": ry,
                "travel_m": round(float(ev.get("travel_m", 25.0)), 1),
                "speed_kmh": round(float(ev.get("ball_speed_kmh", 55.0)), 1),
                "diff": round(float(ev.get("pass_difficulty_rating", 30.0)), 1),
                "xp": round(float(ev.get("expected_pass_completion_xp", 78.0)), 1),
                "outcome": str(ev["outcome_candidate"]),
            }
        )

    ctrl_by_sec = {}
    if not timeline.empty:
        for _, r in timeline.iterrows():
            ctrl_by_sec[str(int(r["second"]))] = {
                "ctrl": round(float(r["control_vs_chaos_index"]), 1),
                "press": round(float(r["tactical_pressure_index"]), 1),
            }

    js_payload = {
        "fps": FPS,
        "max_frame": max_frame,
        "active_ids": active_id_list,
        "carrier_by_frame": {str(k): int(v) for k, v in hud_carrier_by_frame.items()},
        "poss_by_sample": poss_by_sample,
        "radar_frames": radar_frames,
        "player_trails": player_trails,
        "player_totals": player_totals,
        "player_timelines": player_timelines,
        "events": events_list,
        "ctrl_by_sec": ctrl_by_sec,
    }

    return dossiers, passes, timeline, tracks, js_payload


# ============================================================
# LOAD TELEMETRY & VIDEO
# ============================================================
dossiers_df, passes_df, timeline_df, tracks_df, js_payload = (
    build_premier_league_telemetry()
)
video_b64 = get_playable_video_b64(str(VIDEO_PATH), str(WEB_VIDEO_PATH))

# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.title("🦁 MatchMind AI Studio")
st.sidebar.caption("Premier League Broadcast Edition")

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

if not js_payload or video_b64 is None:
    st.warning(
        "Ensure `output/matchmind_analysis.mp4`, `tracking_data.csv`, and `player_statistics.csv` exist."
    )
    st.stop()

# ============================================================
# PREMIER LEAGUE BROADCAST HTML5 + LIVE 2D RADAR + 5-AXIS PENTAGON
# ============================================================
payload_json_str = json.dumps(js_payload)

html_component = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  :root {{
    --pl-purple: #37003c;
    --pl-dark: #120016;
    --pl-card: #1e0924;
    --pl-green: #00ff85;
    --pl-cyan: #04f5ff;
    --pl-pink: #ff2882;
    --pl-gold: #facc15;
  }}
  * {{ box-sizing: border-box; font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; }}
  body {{
    margin: 0; padding: 12px;
    background: radial-gradient(circle at top right, #2b0330 0%, #0d0011 100%);
    color: #f8fafc; border-radius: 14px;
  }}

  /* TOP PREMIER LEAGUE SCOREBOARD & POSSESSION BAR */
  .pl-scoreboard {{
    display: grid; grid-template-columns: 1.1fr 1.6fr 1.1fr; align-items: center;
    background: linear-gradient(90deg, #1f0324 0%, #37003c 50%, #1f0324 100%);
    border: 1px solid rgba(0, 255, 133, 0.3); border-radius: 12px;
    padding: 10px 18px; margin-bottom: 12px; box-shadow: 0 8px 25px rgba(0,0,0,0.45);
  }}
  .sb-left {{ display: flex; align-items: center; gap: 12px; }}
  .pl-badge {{
    background: var(--pl-green); color: var(--pl-purple); font-weight: 900;
    font-size: 11px; padding: 4px 9px; border-radius: 5px; letter-spacing: 0.8px;
  }}
  .clock-pill {{
    background: #09000b; border: 1px solid var(--pl-cyan); color: var(--pl-cyan);
    font-weight: 800; font-size: 14px; padding: 4px 12px; border-radius: 6px;
  }}
  .poss-wrap {{ text-align: center; }}
  .poss-labels {{
    display: flex; justify-content: space-between; font-size: 11.5px; font-weight: 800; margin-bottom: 4px;
  }}
  .poss-bar {{
    height: 8px; width: 100%; background: var(--pl-pink); border-radius: 4px; overflow: hidden; display: flex;
  }}
  .poss-fill-t0 {{ height: 100%; background: var(--pl-cyan); transition: width 0.25s ease; }}
  .sb-right {{ display: flex; justify-content: flex-end; align-items: center; gap: 10px; }}

  /* CONTROL BAR */
  .ctrl-bar {{
    display: flex; flex-wrap: wrap; gap: 6px; align-items: center;
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1);
    padding: 8px 12px; border-radius: 10px; margin-bottom: 14px;
  }}
  .play-btn {{
    background: linear-gradient(135deg, var(--pl-green) 0%, #059669 100%);
    color: var(--pl-purple); border: none; padding: 8px 16px; border-radius: 7px;
    font-weight: 900; font-size: 13px; cursor: pointer; transition: 0.15s;
  }}
  .play-btn:hover {{ filter: brightness(1.1); transform: translateY(-1px); }}
  .ev-btn {{
    background: rgba(255,255,255,0.07); color: #e2e8f0; border: 1px solid rgba(255,255,255,0.15);
    padding: 5px 9px; border-radius: 6px; font-size: 11.5px; font-weight: 700; cursor: pointer;
  }}
  .ev-btn:hover {{ background: var(--pl-cyan); color: var(--pl-purple); border-color: var(--pl-cyan); }}
  select.id-select {{
    background: #1f0324; color: var(--pl-green); border: 1px solid var(--pl-green);
    padding: 5px 9px; border-radius: 6px; font-weight: 800; font-size: 12px;
  }}

  /* MAIN 2-COLUMN BROADCAST GRID */
  .main-grid {{
    display: grid; grid-template-columns: 1.18fr 1fr; gap: 16px; align-items: start;
  }}
  .glass-card {{
    background: rgba(30, 9, 36, 0.85); border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px; padding: 14px; box-shadow: 0 10px 30px rgba(0,0,0,0.4);
  }}
  video {{
    width: 100%; border-radius: 8px; display: block; background: #000; cursor: pointer;
    border: 1px solid rgba(0, 255, 133, 0.25);
  }}

  /* TV LOWER-THIRD OVERLAY */
  .tv-lower-third {{
    margin-top: 10px; background: linear-gradient(90deg, #37003c 0%, #16011a 100%);
    border-left: 5px solid var(--pl-green); padding: 10px 14px; border-radius: 7px;
    display: flex; justify-content: space-between; align-items: center;
  }}
  .lt-tag {{ font-size: 10px; font-weight: 900; color: var(--pl-green); letter-spacing: 1px; }}
  .lt-title {{ font-size: 15.5px; font-weight: 900; color: #fff; margin-top: 2px; }}
  .lt-metrics {{ font-size: 12px; color: #cbd5e1; font-weight: 600; margin-top: 2px; }}

  /* 2D ANIMATED PITCH MINIMAP CANVAS */
  .radar-wrap {{
    margin-top: 12px; background: #0b2e13; border-radius: 10px; padding: 8px;
    border: 1px solid rgba(255,255,255,0.18);
  }}
  .radar-header {{
    display: flex; justify-content: space-between; font-size: 11.5px; font-weight: 800;
    color: var(--pl-green); margin-bottom: 5px; padding: 0 4px;
  }}
  canvas#pitchCanvas {{
    width: 100%; height: 210px; display: block; border-radius: 6px; background: #14532d;
  }}

  /* RIGHT COLUMN: PLAYER ID SCOUTING DOSSIER */
  .dossier-banner {{
    display: flex; justify-content: space-between; align-items: center;
    background: linear-gradient(135deg, #37003c 0%, #240029 100%);
    border: 1px solid rgba(0, 255, 133, 0.35); padding: 12px 16px; border-radius: 10px; margin-bottom: 12px;
  }}
  .pulse-dot {{
    display: inline-block; width: 10px; height: 10px; background: var(--pl-green);
    border-radius: 50%; margin-right: 7px; box-shadow: 0 0 10px var(--pl-green);
  }}
  .hero-id {{ font-size: 22px; font-weight: 900; color: #ffffff; letter-spacing: 0.3px; }}
  .team-badge {{
    padding: 4px 12px; border-radius: 20px; font-weight: 900; font-size: 12px;
  }}

  /* 6 TELEMETRY CARDS + 5-AXIS PENTAGON ROW */
  .telemetry-split {{
    display: grid; grid-template-columns: 1.45fr 0.85fr; gap: 12px; margin-bottom: 12px; align-items: center;
  }}
  .metrics-grid {{
    display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px;
  }}
  .m-card {{
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px; padding: 8px 11px;
  }}
  .m-lbl {{ font-size: 10.5px; font-weight: 800; color: #94a3b8; text-transform: uppercase; }}
  .m-val {{ font-size: 18.5px; font-weight: 900; color: #ffffff; margin: 2px 0; }}
  .m-sub {{ font-size: 10.5px; color: var(--pl-green); font-weight: 700; }}
  .p-bar {{ width: 100%; height: 4px; background: rgba(255,255,255,0.12); border-radius: 2px; margin-top: 4px; overflow: hidden; }}
  .p-fill {{ height: 100%; border-radius: 2px; transition: width 0.2s ease; }}

  .pentagon-box {{
    background: rgba(0,0,0,0.28); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px; padding: 8px; text-align: center;
  }}

  /* AI AGENT BOX */
  .ai-card {{
    background: linear-gradient(135deg, rgba(0, 255, 133, 0.08) 0%, rgba(4, 245, 255, 0.05) 100%);
    border-left: 4px solid var(--pl-green); padding: 11px 14px; border-radius: 8px;
    font-size: 12.8px; line-height: 1.5; color: #f1f5f9; margin-bottom: 12px;
  }}
  .ai-header {{ font-weight: 900; font-size: 13.5px; color: var(--pl-green); margin-bottom: 5px; }}

  /* PASS TABLE */
  .pl-table {{ width: 100%; border-collapse: collapse; font-size: 11.5px; }}
  .pl-table th {{
    background: rgba(255,255,255,0.08); color: var(--pl-cyan); text-align: left;
    padding: 6px 8px; font-weight: 800; text-transform: uppercase; font-size: 10.5px;
  }}
  .pl-table td {{
    padding: 6px 8px; border-bottom: 1px solid rgba(255,255,255,0.08); color: #e2e8f0; font-weight: 600;
  }}
  .tag-ok {{ background: rgba(0,255,133,0.18); color: var(--pl-green); padding: 2px 6px; border-radius: 4px; font-weight: 800; }}
  .tag-bad {{ background: rgba(255,40,130,0.22); color: var(--pl-pink); padding: 2px 6px; border-radius: 4px; font-weight: 800; }}
</style>
</head>
<body>

<!-- TOP PREMIER LEAGUE LIVE SCOREBOARD & POSSESSION SPLIT -->
<div class="pl-scoreboard">
  <div class="sb-left">
    <span class="pl-badge">PREMIER LEAGUE AI HUB</span>
    <span class="clock-pill" id="topClock">00:00.0 (F1)</span>
  </div>

  <div class="poss-wrap">
    <div class="poss-labels">
      <span style="color:var(--pl-cyan);" id="possT0Lbl">TEAM 0 (CYAN): 85% POSSESSION</span>
      <span style="color:#cbd5e1; font-size:10.5px;">LIVE POSSESSION & MOMENTUM</span>
      <span style="color:var(--pl-pink);" id="possT1Lbl">TEAM 1 (PINK): 15%</span>
    </div>
    <div class="poss-bar">
      <div class="poss-fill-t0" id="possBarT0" style="width:85%;"></div>
    </div>
  </div>

  <div class="sb-right">
    <label style="font-size:12px; font-weight:800; color:var(--pl-gold); cursor:pointer;">
      <input type="checkbox" id="voiceToggle"> 🔊 Live AI Voice Commentary
    </label>
  </div>
</div>

<!-- MASTER PLAYBACK & QUICK-JUMP EVENT BAR -->
<div class="ctrl-bar">
  <button class="play-btn" id="masterPlayBtn" onclick="togglePlay()">▶️ PLAY MATCH & SYNC ALL TELEMETRY</button>
  <span style="font-size:11px; font-weight:800; color:#94a3b8; margin-left:4px;">PASS EVENTS:</span>
  <div id="eventButtons" style="display:flex; gap:5px; flex-wrap:wrap;"></div>

  <div style="margin-left:auto; display:flex; align-items:center; gap:8px; font-size:12px; font-weight:800;">
    <label style="cursor:pointer;"><input type="checkbox" id="autoFollowChk" checked onchange="updateUI()"> 🎯 Auto-Follow Ball Carrier #ID</label>
    <select id="manualIdSelect" class="id-select" onchange="onManualSelect()"></select>
  </div>
</div>

<div class="main-grid">
  <!-- LEFT COLUMN: CLICKABLE VIDEO + TV LOWER-THIRD + LIVE 60-FPS 2D RADAR MINIMAP -->
  <div class="glass-card">
    <video id="matchVideo" controls playsinline preload="auto">
      <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
    </video>

    <div class="tv-lower-third">
      <div>
        <div class="lt-tag" id="ltHeader">LIVE BROADCAST GRAPHIC • FRAME 1</div>
        <div class="lt-title" id="ltMain">ON BALL: PLAYER ID #8 (TEAM 0)</div>
        <div class="lt-metrics" id="ltSub">Zone: Middle Third | Distance: 0.0m | Dribbles: 0 (0.00m) | xT: +0.000</div>
      </div>
      <div style="text-align:right;">
        <div style="font-size:10px; color:#94a3b8; font-weight:800;">CONTROL VS CHAOS</div>
        <div id="ltCtrl" style="font-size:19px; font-weight:900; color:var(--pl-green);">+40.0</div>
      </div>
    </div>

    <!-- LIVE 2D DIGITAL TWIN TACTICAL RADAR CANVAS -->
    <div class="radar-wrap">
      <div class="radar-header">
        <span>🏟️ LIVE 2D DIGITAL TWIN PITCH RADAR (GOLD RING = ACTIVE BALL CARRIER #ID)</span>
        <span id="radarSubLbl" style="color:var(--pl-cyan);">Trail + Pass Vectors Active</span>
      </div>
      <canvas id="pitchCanvas" width="760" height="240"></canvas>
    </div>
  </div>

  <!-- RIGHT COLUMN: LIVE PLAYER #ID DOSSIER + 5-AXIS RADAR PENTAGON + AI NARRATOR -->
  <div class="glass-card">
    <div class="dossier-banner">
      <div>
        <div style="font-size:10.5px; color:var(--pl-green); font-weight:800;" id="dossierStatus">
          <span class="pulse-dot"></span>LIVE AUTO-FOLLOWING BALL CARRIER
        </div>
        <div class="hero-id" id="dossierPlayerId">Player ID #8</div>
      </div>
      <div style="text-align:right;">
        <span class="team-badge" id="dossierTeamPill" style="background:var(--pl-cyan); color:#000;">Team 0</span>
        <div style="font-size:11.5px; color:#cbd5e1; margin-top:5px; font-weight:700;" id="dossierZone">📍 Middle Third</div>
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
          <div class="m-lbl">🤝 Receptions & Touches</div>
          <div class="m-val" id="mRec">0 Rec (0f)</div>
          <div class="m-sub" id="mRecSub">Match Total: 0 Rec</div>
          <div class="p-bar"><div class="p-fill" id="barRec" style="width:0%; background:#a855f7;"></div></div>
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
        <div style="font-size:10.5px; font-weight:900; color:var(--pl-cyan); margin-bottom:2px;">
          🕸️ LIVE ID ATTRIBUTE PROFILE
        </div>
        <svg id="radarSvg" width="185" height="175" viewBox="0 0 200 190"></svg>
        <div style="font-size:10px; color:#94a3b8; font-weight:700;" id="ovrBadge">OVR IMPACT RATING: 88</div>
      </div>
    </div>

    <!-- AI MULTI-AGENT NARRATIVE -->
    <div class="ai-card">
      <div class="ai-header" id="agentTitle">🧠 Live AI Agent Breakdown</div>
      <div id="agentBody">Click Play on the video to stream synchronized telemetry...</div>
    </div>

    <!-- PASS & RECEPTION TABLE -->
    <div style="font-size:11.5px; font-weight:900; color:var(--pl-cyan); margin-bottom:5px; text-transform:uppercase;" id="tableHeader">
      📋 Pass & Reception Events Involving Player ID #8:
    </div>
    <table class="pl-table">
      <thead>
        <tr>
          <th>Ev</th>
          <th>Time</th>
          <th>Passer</th>
          <th>Receiver</th>
          <th>Dist</th>
          <th>Speed</th>
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

let lastSpokenEventId = -1;

// Populate manual ID selector with pure video IDs
DATA.active_ids.forEach(pid => {{
  const opt = document.createElement('option');
  const t = DATA.player_totals[String(pid)].team;
  opt.value = pid;
  opt.textContent = `Player ID #${{pid}} (Team ${{t}})`;
  manualSelect.appendChild(opt);
}});

// Populate event buttons
const evContainer = document.getElementById('eventButtons');
DATA.events.forEach(ev => {{
  const btn = document.createElement('button');
  btn.className = 'ev-btn';
  btn.textContent = `E${{ev.event_id}}: #${{ev.passer}}→#${{ev.receiver}} (${{ev.sec.toFixed(1)}}s)`;
  btn.onclick = () => {{
    video.currentTime = Math.max(0, ev.sec - 0.12);
    updateUI();
  }};
  evContainer.appendChild(btn);
}});

function togglePlay() {{
  if (video.paused) video.play();
  else video.pause();
}}

function onManualSelect() {{
  autoChk.checked = false;
  updateUI();
}}

video.addEventListener('play', () => {{
  playBtn.textContent = '⏸️ PAUSE MATCH & TELEMETRY';
  playBtn.style.background = 'linear-gradient(135deg, #facc15 0%, #eab308 100%)';
}});

video.addEventListener('pause', () => {{
  playBtn.textContent = '▶️ PLAY MATCH & SYNC ALL TELEMETRY';
  playBtn.style.background = 'linear-gradient(135deg, #00ff85 0%, #059669 100%)';
}});

video.addEventListener('timeupdate', updateUI);
video.addEventListener('seeked', updateUI);

function getNearestSampleFrame(frame) {{
  if (frame >= DATA.max_frame) return String(DATA.max_frame);
  const snapped = Math.max(1, Math.floor((frame - 1) / 4) * 4 + 1);
  return String(snapped);
}}

// Draw 5-Axis SVG Attribute Radar Pentagon
function drawAttributePentagon(scores) {{
  const svg = document.getElementById('radarSvg');
  const cx = 100, cy = 98, R = 64;
  const labels = ["PROG", "RET", "WORK", "DRIB", "PACE"];
  let html = '';

  // Concentric pentagon rings
  [0.35, 0.68, 1.0].forEach(scale => {{
    let pts = [];
    for (let i = 0; i < 5; i++) {{
      const ang = -Math.PI / 2 + (i * 2 * Math.PI) / 5;
      pts.push(`${{cx + R * scale * Math.cos(ang)}},${{cy + R * scale * Math.sin(ang)}}`);
    }}
    html += `<polygon points="${{pts.join(' ')}}" fill="none" stroke="rgba(255,255,255,0.15)" stroke-width="1"/>`;
  }});

  // Axis lines & labels
  let dataPts = [];
  for (let i = 0; i < 5; i++) {{
    const ang = -Math.PI / 2 + (i * 2 * Math.PI) / 5;
    const x2 = cx + R * Math.cos(ang);
    const y2 = cy + R * Math.sin(ang);
    html += `<line x1="${{cx}}" y1="${{cy}}" x2="${{x2}}" y2="${{y2}}" stroke="rgba(255,255,255,0.15)" stroke-width="1"/>`;

    const lx = cx + (R + 20) * Math.cos(ang);
    const ly = cy + (R + 16) * Math.sin(ang) + 3;
    html += `<text x="${{lx}}" y="${{ly}}" fill="#cbd5e1" font-size="9.5" font-weight="800" text-anchor="middle">${{labels[i]}} (${{scores[i]}})</text>`;

    const valR = (Math.max(30, scores[i]) / 100.0) * R;
    dataPts.push(`${{cx + valR * Math.cos(ang)}},${{cy + valR * Math.sin(ang)}}`);
  }}

  html += `<polygon points="${{dataPts.join(' ')}}" fill="rgba(0, 255, 133, 0.32)" stroke="#00ff85" stroke-width="2.2"/>`;
  svg.innerHTML = html;

  const ovr = Math.round(scores.reduce((a, b) => a + b, 0) / 5);
  document.getElementById('ovrBadge').textContent = `OVR IMPACT RATING: ${{ovr}} / 99`;
}}

// Draw Animated 2D Pitch Radar Minimap
function drawPitchMinimap(frame, sampleKey, activePid) {{
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // Pitch turf stripes
  for (let i = 0; i < 10; i++) {{
    ctx.fillStyle = i % 2 === 0 ? '#14532d' : '#166534';
    ctx.fillRect((W / 10) * i, 0, W / 10, H);
  }}

  // Pitch boundary & markings
  ctx.strokeStyle = 'rgba(255,255,255,0.65)';
  ctx.lineWidth = 1.5;
  ctx.strokeRect(16, 12, W - 32, H - 24);
  ctx.beginPath();
  ctx.moveTo(W / 2, 12); ctx.lineTo(W / 2, H - 12);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(W / 2, H / 2, 28, 0, Math.PI * 2);
  ctx.stroke();

  const mapX = x => 16 + (x / 1920.0) * (W - 32);
  const mapY = y => 12 + ((y - 260.0) / 780.0) * (H - 24);

  // Draw completed/turnover pass vectors up to current frame
  DATA.events.filter(e => e.frame <= frame).forEach(ev => {{
    const x1 = mapX(ev.px), y1 = mapY(ev.py);
    const x2 = mapX(ev.rx), y2 = mapY(ev.ry);
    ctx.strokeStyle = ev.outcome.includes('completed') ? 'rgba(0, 255, 133, 0.55)' : 'rgba(255, 40, 130, 0.65)';
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
    ctx.stroke();
  }});

  // Draw active player's movement trail in Gold
  const trail = (DATA.player_trails[String(activePid)] || []).filter(pt => pt[0] <= frame);
  if (trail.length > 1) {{
    ctx.strokeStyle = '#facc15';
    ctx.lineWidth = 2.6;
    ctx.beginPath();
    trail.forEach((pt, idx) => {{
      const tx = mapX(pt[1]), ty = mapY(pt[2]);
      if (idx === 0) ctx.moveTo(tx, ty);
      else ctx.lineTo(tx, ty);
    }});
    ctx.stroke();
  }}

  // Draw all tracked players at current sample frame
  const rFrame = DATA.radar_frames[sampleKey];
  if (rFrame && rFrame.p) {{
    rFrame.p.forEach(p => {{
      const pid = p[0], px = mapX(p[1]), py = mapY(p[2]), team = p[3];
      const isHero = pid === activePid;

      if (isHero) {{
        // Outer glowing sonar ring for active ball carrier
        ctx.beginPath();
        ctx.arc(px, py, 12, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(250, 204, 21, 0.32)';
        ctx.fill();
      }}

      ctx.beginPath();
      ctx.arc(px, py, isHero ? 6.5 : 4.5, 0, Math.PI * 2);
      ctx.fillStyle = isHero ? '#facc15' : (team === 0 ? '#04f5ff' : '#ff2882');
      ctx.fill();
      ctx.strokeStyle = '#000';
      ctx.lineWidth = 1.2;
      ctx.stroke();

      if (isHero || DATA.active_ids.includes(pid)) {{
        ctx.fillStyle = isHero ? '#facc15' : '#ffffff';
        ctx.font = isHero ? 'bold 10px sans-serif' : 'bold 8.5px sans-serif';
        ctx.fillText(`#${{pid}}`, px - 7, py - 8);
      }}
    }});

    // Draw ball
    if (rFrame.b) {{
      const bx = mapX(rFrame.b[0]), by = mapY(rFrame.b[1]);
      ctx.beginPath();
      ctx.arc(bx, by, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
      ctx.strokeStyle = '#facc15';
      ctx.lineWidth = 1.5;
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
      ? `Pass completed from Player I D ${{hitEv.passer}} to I D ${{hitEv.receiver}}, ${{hitEv.travel_m}} meters.`
      : `Turnover! Ball contested between I D ${{hitEv.passer}} and I D ${{hitEv.receiver}}.`;
    const utter = new SpeechSynthesisUtterance(txt);
    utter.rate = 1.12;
    window.speechSynthesis.speak(utter);
  }}
}}

function buildNarrative(pid, team, sec, frame, live, totals, ctrl, latestEvStr) {{
  const stateWord = ctrl < 0 ? "High-Chaos Turnover Phase" : "Structured Tactical Control";
  if (LANG === "Swahili") {{
    return `• <b>Mchezaji Mwenye Mpira (${{sec.toFixed(1)}}s):</b> <b>Player ID #${{pid}} (Timu ${{team}})</b> yuko kwenye <i>${{live.zone}}</i>.<br>` +
           `• <b>Takwimu Mbashara:</b> Umbali <b>${{live.dist_m.toFixed(1)}}m</b> (Jumla: ${{totals.total_dist_m.toFixed(1)}}m), kukokota mpira (dribbles) <b>${{live.drib}} (${{live.carry_m.toFixed(2)}}m)</b>, pasi <b>${{live.comp}}/${{live.att}}</b>, xT <b>+${{live.xt.toFixed(3)}}</b>.<br>` +
           `• <b>Tukio la Karibuni:</b> ${{latestEvStr}}`;
  }}
  if (LANG === "Spanish") {{
    return `• <b>Portador del Balón (${{sec.toFixed(1)}}s):</b> <b>Player ID #${{pid}} (Equipo ${{team}})</b> en el <i>${{live.zone}}</i>.<br>` +
           `• <b>Telemetría Premier League:</b> Distancia <b>${{live.dist_m.toFixed(1)}}m</b> (Total: ${{totals.total_dist_m.toFixed(1)}}m), <b>${{live.drib}} regates (${{live.carry_m.toFixed(2)}}m)</b>, <b>${{live.comp}}/${{live.att}} pases</b>, amenaza esperada <b>+${{live.xt.toFixed(3)}} xT</b>.<br>` +
           `• <b>Última Acción:</b> ${{latestEvStr}}`;
  }}
  if (LANG === "French") {{
    return `• <b>Porteur du Ballon (${{sec.toFixed(1)}}s):</b> <b>Player ID #${{pid}} (Équipe ${{team}})</b> dans le <i>${{live.zone}}</i>.<br>` +
           `• <b>Télémétrie en Direct:</b> Distance <b>${{live.dist_m.toFixed(1)}}m</b> (Total: ${{totals.total_dist_m.toFixed(1)}}m), <b>${{live.drib}} dribbles (${{live.carry_m.toFixed(2)}}m)</b>, <b>${{live.comp}}/${{live.att}} passes</b>, menace <b>+${{live.xt.toFixed(3)}} xT</b>.<br>` +
           `• <b>Dernière Action:</b> ${{latestEvStr}}`;
  }}

  if (PERSONA === "Fan Mode") {{
    return `• <b>🔥 Spotlight on Player ID #${{pid}} (${{sec.toFixed(1)}}s):</b> Driving Team ${{team}}'s attack inside the <b>${{live.zone}}</b>!<br>` +
           `• <b>⚡ Dribble & Workrate Impact:</b> Covered <b>${{live.dist_m.toFixed(1)}} m</b> with <b>${{live.drib}} dribble sequence(s) (${{live.carry_m.toFixed(2)}} m carried)</b>, generating <b>+${{live.xt.toFixed(3)}} Expected Threat (xT)</b> at up to <b>${{live.max_spd.toFixed(1)}} km/h</b>!<br>` +
           `• <b>🎯 Latest Pitch Action:</b> ${{latestEvStr}}`;
  }}
  if (PERSONA === "Broadcast Mode") {{
    return `• <b>🎙️ [LOWER-THIRD BROADCAST FEED]:</b> <code>ID #${{pid}} (TEAM ${{team}}) | DIST: ${{live.dist_m.toFixed(1)}}m | DRIB: ${{live.drib}} (${{live.carry_m.toFixed(2)}}m) | xT: +${{live.xt.toFixed(3)}} | SPD: ${{live.cur_spd.toFixed(1)}} km/h</code><br>` +
           `• <b>📺 Commentator Cue (${{sec.toFixed(1)}}s):</b> <i>"Player ID #${{pid}} dictating play in the ${{live.zone}} with ${{totals.press_res}}/100 press resistance and ${{live.carry_m.toFixed(2)}}m of progressive ball carrying."</i><br>` +
           `• <b>⚡ Match Rhythm:</b> ${{stateWord}} (Control Index: <b>${{ctrl >= 0 ? '+' : ''}}${{ctrl.toFixed(1)}}</b>).`;
  }}
  return `• <b>🧠 Tactical State (Frame ${{frame}} / ${{sec.toFixed(1)}}s):</b> Match is in <b>${{stateWord}}</b> (Control Index: <b>${{ctrl >= 0 ? '+' : ''}}${{ctrl.toFixed(1)}}</b>) with <b>Player ID #${{pid}}</b> operating in the <b>${{live.zone}}</b>.<br>` +
         `• <b>📊 Physical & Ball-Carrying Telemetry:</b> Covered <b>${{live.dist_m.toFixed(1)}} m</b> (of ${{totals.total_dist_m.toFixed(1)}} m total), logging <b>${{live.drib}} dribble(s) (${{live.carry_m.toFixed(2)}} m carry)</b>, <b>+${{live.xt.toFixed(3)}} xT</b>, and <b>${{totals.press_res}}/100 Press Resistance</b>.<br>` +
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
    att: 0, comp: 0, rec: 0, drib: 0, carry_m: 0.0, touches: 0, dist_m: 0.0, cur_spd: 0.0, max_spd: 0.0, xt: 0.0, zone: "Middle Third"
  }};

  // Top scoreboard clock & possession bar
  const mins = String(Math.floor(sec / 60)).padStart(2, '0');
  const remS = (sec % 60).toFixed(1).padStart(4, '0');
  document.getElementById('topClock').textContent = `${{mins}}:${{remS}} (F${{frame}})`;

  const t0Poss = DATA.poss_by_sample[sampleKey] || 84.0;
  const t1Poss = (100.0 - t0Poss).toFixed(1);
  document.getElementById('possT0Lbl').textContent = `TEAM 0 (CYAN): ${{t0Poss.toFixed(1)}}% POSSESSION`;
  document.getElementById('possT1Lbl').textContent = `TEAM 1 (PINK): ${{t1Poss}}%`;
  document.getElementById('possBarT0').style.width = `${{t0Poss}}%`;

  const secKey = String(Math.min(26, Math.max(1, Math.ceil(sec || 1))));
  const ctrlObj = DATA.ctrl_by_sec[secKey] || {{ ctrl: 40.0, press: 25.0 }};

  // Dossier Banner & Lower Third
  document.getElementById('dossierStatus').innerHTML = autoChk.checked
    ? `<span class="pulse-dot"></span>AUTO-FOLLOWING VIDEO BALL CARRIER @ ${{sec.toFixed(1)}}s (FRAME ${{frame}})`
    : `📌 MANUALLY INSPECTING PLAYER ID @ ${{sec.toFixed(1)}}s (FRAME ${{frame}})`;

  document.getElementById('dossierPlayerId').textContent = `Player ID #${{activePid}}`;
  const pill = document.getElementById('dossierTeamPill');
  pill.textContent = `Team ${{totals.team}}`;
  pill.style.background = totals.team === 0 ? '#04f5ff' : '#ff2882';
  pill.style.color = totals.team === 0 ? '#000' : '#fff';
  document.getElementById('dossierZone').textContent = `📍 ${{live.zone}}`;

  document.getElementById('ltHeader').textContent = `LIVE BROADCAST TELEMETRY • FRAME ${{frame}} (${{sec.toFixed(2)}}s)`;
  document.getElementById('ltMain').textContent = `ACTIVE CARRIER: PLAYER ID #${{activePid}} (TEAM ${{totals.team}})`;
  document.getElementById('ltSub').textContent = `Zone: ${{live.zone}} | Distance Covered: ${{live.dist_m.toFixed(1)}}m | Dribbles: ${{live.drib}} (${{live.carry_m.toFixed(2)}}m) | xT: +${{live.xt.toFixed(3)}}`;

  const ltCtrl = document.getElementById('ltCtrl');
  ltCtrl.textContent = `${{ctrlObj.ctrl >= 0 ? '+' : ''}}${{ctrlObj.ctrl.toFixed(1)}}`;
  ltCtrl.style.color = ctrlObj.ctrl >= 0 ? '#00ff85' : '#ff2882';

  // Update 6 Metric Cards
  document.getElementById('mDist').textContent = `${{live.dist_m.toFixed(1)}} m`;
  document.getElementById('mDistSub').textContent = `Match Total: ${{totals.total_dist_m.toFixed(1)}} m`;
  document.getElementById('barDist').style.width = `${{Math.min(100, (live.dist_m / Math.max(1, totals.total_dist_m)) * 100)}}%`;

  document.getElementById('mDrib').textContent = `${{live.drib}} (${{live.carry_m.toFixed(2)}}m)`;
  document.getElementById('mDribSub').textContent = `Match Total: ${{totals.total_drib}} (${{totals.total_carry_m.toFixed(2)}}m)`;
  document.getElementById('barDrib').style.width = `${{Math.min(100, (live.carry_m / Math.max(0.5, totals.total_carry_m)) * 100)}}%`;

  document.getElementById('mPass').textContent = `${{live.comp}} / ${{live.att}}`;
  document.getElementById('mPassSub').textContent = `Match Total: ${{totals.total_comp}} / ${{totals.total_att}}`;
  document.getElementById('barPass').style.width = `${{totals.total_att > 0 ? Math.min(100, (live.comp / totals.total_att) * 100) : 100}}%`;

  document.getElementById('mRec').textContent = `${{live.rec}} Rec (${{live.touches}}f)`;
  document.getElementById('mRecSub').textContent = `Match Total: ${{totals.total_rec}} Rec (${{totals.total_touches}}f)`;
  document.getElementById('barRec').style.width = `${{Math.min(100, (live.touches / Math.max(1, totals.total_touches)) * 100)}}%`;

  document.getElementById('mXt').textContent = `+${{live.xt.toFixed(3)}} xT`;
  document.getElementById('mXtSub').textContent = `Press Resistance: ${{totals.press_res}}/100`;
  document.getElementById('barXt').style.width = `${{Math.min(100, (live.xt / Math.max(0.05, totals.xt)) * 100)}}%`;

  document.getElementById('mSpd').textContent = `${{live.cur_spd.toFixed(1)}} km/h`;
  document.getElementById('mSpdSub').textContent = `Peak Speed: ${{totals.peak_spd.toFixed(1)}} km/h`;
  document.getElementById('barSpd').style.width = `${{Math.min(100, (live.cur_spd / 34.5) * 100)}}%`;

  // Draw 5-Axis Radar Pentagon & 2D Pitch Minimap
  drawAttributePentagon(totals.radar);
  drawPitchMinimap(frame, sampleKey, activePid);
  maybeSpeakLiveEvent(sec, frame);

  // Latest event text
  const pastEvents = DATA.events.filter(e => e.frame <= frame);
  let latestEvStr = "Initiating progressive ball carry.";
  if (pastEvents.length > 0) {{
    const lastEv = pastEvents[pastEvents.length - 1];
    latestEvStr = `Event #${{lastEv.event_id}} (${{lastEv.sec.toFixed(1)}}s): ID #${{lastEv.passer}} → ID #${{lastEv.receiver}} (${{lastEv.travel_m}}m, ${{lastEv.speed_kmh}} km/h, xP: ${{lastEv.xp}}%).`;
  }}

  document.getElementById('agentTitle').textContent = `🤖 ${{PERSONA}} (${{LANG}}) — Live Report for Player ID #${{activePid}}`;
  document.getElementById('agentBody').innerHTML = buildNarrative(
    activePid, totals.team, sec, frame, live, totals, ctrlObj.ctrl, latestEvStr
  );

  // Update Pass Table
  document.getElementById('tableHeader').textContent = `📋 Pass & Reception Events Involving Player ID #${{activePid}}:`;
  const pEvents = DATA.events.filter(e => e.passer === activePid || e.receiver === activePid);
  const tbody = document.getElementById('passTableBody');
  tbody.innerHTML = '';
  pEvents.forEach(ev => {{
    const tr = document.createElement('tr');
    const isPast = ev.frame <= frame;
    tr.style.opacity = isPast ? '1.0' : '0.4';
    const isComp = ev.outcome.includes('completed');
    const badgeClass = isComp ? 'tag-ok' : 'tag-bad';
    const badgeText = isComp ? 'Completed' : 'Turnover';
    tr.innerHTML = `
      <td>#${{ev.event_id}} ${{isPast ? '✅' : '⏳'}}</td>
      <td>${{ev.sec.toFixed(2)}}s</td>
      <td><b>ID #${{ev.passer}}</b></td>
      <td><b>ID #${{ev.receiver}}</b></td>
      <td>${{ev.travel_m}}m</td>
      <td>${{ev.speed_kmh}} km/h</td>
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

components.html(html_component, height=860, scrolling=True)