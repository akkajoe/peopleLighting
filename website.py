import os
import json
import pickle
import argparse
import csv
import re
import math
import shutil
from collections import Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _to_json_safe(obj):
    try:
        import numpy as _np

        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, (_np.ndarray,)):
            return obj.tolist()
    except Exception:
        pass
    return str(obj)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_filename(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")
    return s if s else "untitled"


def copy_file_if_needed(src: str, dst: str):
    ensure_dir(os.path.dirname(dst))
    if os.path.exists(dst):
        return
    shutil.copy2(src, dst)


def build_basename_index(root_dir: str):
    """Map filename basename -> full path (first occurrence wins). Handles nested folders."""
    idx = {}
    for r, _dirs, files in os.walk(root_dir):
        for fn in files:
            if fn not in idx:
                idx[fn] = os.path.join(r, fn)
    return idx


def _extract_ev_tag(s: str) -> str:
    if not s:
        return ""
    s = str(s)
    for ev in ("ev-00", "ev-25", "ev-50"):
        if ev in s:
            return ev
    return ""


def _normalize_id_and_ev(s: str):
    s = str(s).strip()
    s = os.path.basename(s)
    ev = _extract_ev_tag(s)

    low = s.lower()
    if low.endswith(".gz"):
        s = s[:-3]
        low = s.lower()
    if low.endswith(".json"):
        s = s[:-5]
        low = s.lower()
    if low.endswith("_points"):
        s = s[:-7]
        low = s.lower()

    for tag in ("_ev-00", "_ev-25", "_ev-50"):
        if low.endswith(tag):
            s = s[: -len(tag)]
            low = s.lower()
            break

    s = re.sub(r"^(?:rank\s*)?r[123]\s*[:_\-]+", "", s, flags=re.IGNORECASE)
    s = s.replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    s = s.strip("_")
    return s, ev


def confidence_from_row(row, E_good=1.0, E_bad=8.0):
    try:
        frac = float(row.get("inlier_weight_frac", 0.0))
    except Exception:
        frac = 0.0
    try:
        mae = float(row.get("mae_in_deg_weighted", 1e9))
    except Exception:
        mae = 1e9

    if E_bad <= E_good:
        mae_conf = 1.0 if mae <= E_good else 0.0
    else:
        t = (mae - E_good) / (E_bad - E_good)
        if t < 0.0:
            mae_conf = 1.0
        elif t > 1.0:
            mae_conf = 0.0
        else:
            mae_conf = 1.0 - t

    conf = frac * mae_conf
    return max(0.0, min(1.0, conf))


def load_confidence_map_by_id_and_rank(csv_path: str, desired_ev: str, E_good=1.0, E_bad=8.0):
    conf_map = {}
    if not csv_path:
        return conf_map
    if not os.path.exists(csv_path):
        print(f"WARNING: CSV not found: {csv_path}")
        return conf_map

    kept = 0
    skipped_ev = 0
    skipped_rank = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                rk = int(float(row.get("rank", "0")))
            except Exception:
                skipped_rank += 1
                continue

            base_id, ev = _normalize_id_and_ev(row.get("file", ""))

            if desired_ev and ev != desired_ev:
                skipped_ev += 1
                continue

            conf_map[(base_id, rk)] = confidence_from_row(row, E_good=E_good, E_bad=E_bad)
            kept += 1

    print(f"[CSV] Loaded confidence rows kept={kept} | skipped_ev={skipped_ev} | skipped_rank={skipped_rank}")
    return conf_map

DIR_KEY_CANDIDATES = [
    ("dir_x", "dir_y", "dir_z"),
    ("dir_vx", "dir_vy", "dir_vz"),
    ("dirvx", "dirvy", "dirvz"),
    ("light_dir_x", "light_dir_y", "light_dir_z"),
    ("direction_x", "direction_y", "direction_z"),
    ("vx", "vy", "vz"),
    ("Lx", "Ly", "Lz"),
    ("L_x", "L_y", "L_z"),
]
SINGLE_VEC_CANDIDATES = ["dir", "dir_v", "light_dir", "direction", "dir_vec", "dirv", "L", "light_direction"]


def _as_float(x):
    try:
        return float(x)
    except Exception:
        return None


def extract_direction_components(d: dict):
    for kx, ky, kz in DIR_KEY_CANDIDATES:
        if kx in d and ky in d and kz in d:
            x = _as_float(d.get(kx))
            y = _as_float(d.get(ky))
            z = _as_float(d.get(kz))
            if x is not None and y is not None and z is not None:
                return x, y, z, f"{kx},{ky},{kz}"

    for vk in SINGLE_VEC_CANDIDATES:
        if vk in d:
            v = d.get(vk)
            if isinstance(v, (list, tuple)) and len(v) >= 3:
                x = _as_float(v[0])
                y = _as_float(v[1])
                z = _as_float(v[2])
                if x is not None and y is not None and z is not None:
                    return x, y, z, vk

    for root in ("dir", "direction", "light_dir", "L"):
        obj = d.get(root)
        if isinstance(obj, dict):
            for cand in [("x", "y", "z"), ("vx", "vy", "vz"), ("dir_x", "dir_y", "dir_z")]:
                if all(k in obj for k in cand):
                    x = _as_float(obj.get(cand[0]))
                    y = _as_float(obj.get(cand[1]))
                    z = _as_float(obj.get(cand[2]))
                    if x is not None and y is not None and z is not None:
                        return (
                            x,
                            y,
                            z,
                            f"{root}.{cand[0]},{root}.{cand[1]},{root}.{cand[2]}",
                        )

    return None, None, None, "no_direction_keys_found"


def normalize_vec(x, y, z, eps=1e-12):
    n = math.sqrt(x * x + y * y + z * z)
    if not math.isfinite(n) or n < eps:
        return None, None, None, n
    return x / n, y / n, z / n, n


def _arrow_safe_filename(name: str) -> str:
    return safe_filename(name)


def _arrow_base_ev(fname: str):
    base = re.sub(r"_points\.json(\.gz)?$", "", os.path.basename(str(fname)))
    m = re.search(r"^(.*)_ev-(\d{2})$", base)
    return (m.group(1), m.group(2)) if m else (base, None)


def _arrow_unit(v):
    v = np.array(v, float)
    n = np.linalg.norm(v)
    return v / (n if n else 1.0)


def _arrow_s2l_from_csv_frame(v_l2s):
    v = -np.array(v_l2s, float)
    v[1] *= -1.0
    return _arrow_unit(v)


def _arrow_vec_from_row_basic(r: dict):
    if all(k in r and pd.notna(r[k]) for k in ("dir_vx", "dir_vy", "dir_vz")):
        v = np.array([float(r["dir_vx"]), float(r["dir_vy"]), float(r["dir_vz"])], float)
    else:
        az, el = float(r["dir_az_deg"]), float(r["dir_el_deg"])
        azr, elr = math.radians(az), math.radians(el)
        v = np.array(
            [math.cos(elr) * math.cos(azr), math.cos(elr) * math.sin(azr), math.sin(elr)],
            float,
        )
    return _arrow_unit(v)


def _arrow_weighted_mean_dir(V, W):
    v = (V * W).sum(axis=0) / (W.sum(axis=0) + 1e-12)
    return _arrow_unit(v)


_ARROW_RANK_COLORS = {1: "red", 2: "orange", 3: "purple"}
_ARROW_RANK_LINE_WIDTH = 5
_ARROW_CONE_SIZEREF = 0.18


def _arrow_add_rank_arrow(fig: go.Figure, rk: int, v: np.ndarray):
    color = _ARROW_RANK_COLORS.get(rk, "black")
    name = f"Rank {rk}"

    fig.add_trace(
        go.Scatter3d(
            x=[0, float(v[0])],
            y=[0, float(v[1])],
            z=[0, float(v[2])],
            mode="lines",
            line=dict(color=color, width=_ARROW_RANK_LINE_WIDTH),
            name=name,
            showlegend=True,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Cone(
            x=[float(v[0])],
            y=[float(v[1])],
            z=[float(v[2])],
            u=[float(v[0])],
            v=[float(v[1])],
            w=[float(v[2])],
            anchor="tip",
            colorscale=[[0, color], [1, color]],
            showscale=False,
            name=name,
            showlegend=False,
            hoverinfo="skip",
            sizemode="absolute",
            sizeref=_ARROW_CONE_SIZEREF,
        )
    )


def _arrow_make_arrow_plot_json(rows_for_base, top_k=3, ev_only=None, flip_s2l=False):
    by_rank = {}
    for r in rows_for_base:
        try:
            rk = int(float(r.get("rank", 1)))
        except Exception:
            rk = 1
        rk = max(1, min(top_k, rk))
        by_rank.setdefault(rk, []).append(r)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode="markers",
            marker=dict(size=4, color="black"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    base_title = rows_for_base[0].get("_base") if rows_for_base else "Arrow plot"
    for rk in sorted(by_rank.keys()):
        rows_rk = by_rank[rk]
        if ev_only is not None:
            rows_rk = [r for r in rows_rk if str(r.get("_ev")) == str(ev_only)]
            if not rows_rk:
                continue

        v_list = []
        w_list = []
        for r in rows_rk:
            v = _arrow_vec_from_row_basic(r)
            if flip_s2l:
                v = _arrow_s2l_from_csv_frame(v)
            v_list.append(v)

            try:
                w = float(r.get("score", 1.0))
                if not np.isfinite(w) or w <= 0:
                    w = 1.0
            except Exception:
                w = 1.0
            w_list.append(w)

        if not v_list:
            continue

        V = np.vstack(v_list)
        W = np.asarray(w_list, float).reshape(-1, 1)
        v_rank = _arrow_weighted_mean_dir(V, W)
        _arrow_add_rank_arrow(fig, rk, v_rank)

    R = 1.0
    cameraYZ = dict(
        eye=dict(x=2.6, y=0.0, z=0.0),
        center=dict(x=0.0, y=0.0, z=0.0),
        up=dict(x=0.0, y=0.0, z=1.0),
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="dir_x", range=[-R, R], showgrid=True, zeroline=False, autorange=False),
            yaxis=dict(title="dir_y", range=[-R, R], showgrid=True, zeroline=False, autorange=False),
            zaxis=dict(title="dir_z", range=[-R, R], showgrid=True, zeroline=False, autorange=False),
            aspectmode="cube",
            camera=cameraYZ,
        ),
        margin=dict(l=10, r=10, t=60, b=10),
        height=650,
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=True,
        title=dict(text=f"{base_title}"),
    )
    return fig.to_plotly_json()


def _arrow_generate_plot_jsons_from_csv(csv_path: str, out_dir: str, ev_only=None, flip_s2l=False):
    df = pd.read_csv(csv_path)

    recs = []
    for _, row in df.iterrows():
        r = dict(row)
        b, ev = _arrow_base_ev(r.get("file", ""))
        r["_base"] = b
        r["_ev"] = ev
        recs.append(r)

    by_base = {}
    for r in recs:
        by_base.setdefault(r["_base"], []).append(r)

    ensure_dir(out_dir)

    wrote = 0
    for base in sorted(by_base.keys()):
        payload = _arrow_make_arrow_plot_json(by_base[base], top_k=3, ev_only=ev_only, flip_s2l=flip_s2l)
        out_json = os.path.join(out_dir, _arrow_safe_filename(base) + ".plot.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        wrote += 1

    print(f"[ARROWS] Generated {wrote} plot json files into: {out_dir}")


def stage_painting_images(points_min, img_src_dir: str, out_dir: str, img_prefix: str, debug=False):
    if not img_src_dir:
        print("[ASSETS] img-src not provided; images will not be copied.")
        return
    img_src_dir = os.path.normpath(img_src_dir)
    if not os.path.exists(img_src_dir):
        print(f"[ASSETS] WARNING: img-src not found: {img_src_dir}")
        return

    dst_root = os.path.join(out_dir, img_prefix)
    ensure_dir(dst_root)

    idx = build_basename_index(img_src_dir)

    needed = set()
    for d in points_min:
        base = d.get("img_file", "")
        if base:
            needed.add(base)

    copied = 0
    missing = 0
    for base in sorted(needed):
        src = idx.get(base)
        dst = os.path.join(dst_root, base)
        if src and os.path.exists(src):
            copy_file_if_needed(src, dst)
            copied += 1
        else:
            missing += 1
            if debug:
                print(f"[ASSETS] Missing painting image basename='{base}' under img-src")

    print(f"[ASSETS] Painting images staged: copied={copied}, missing={missing}, dst='{dst_root}'")


def stage_plots(plots_src: str, out_dir: str, plots_prefix: str, desired_ev_tag: str, flip_s2l: bool):
    if not plots_src:
        print("[ASSETS] plots-src not provided; arrow JSON will not be copied/generated.")
        return

    plots_src = os.path.normpath(plots_src)
    dst_root = os.path.join(out_dir, plots_prefix)
    ensure_dir(dst_root)

    # If plots-src is a CSV, generate arrow plots
    if os.path.isfile(plots_src) and plots_src.lower().endswith(".csv"):
        ev_only = None
        if desired_ev_tag:
            m = re.search(r"ev-(\d{2})", desired_ev_tag.lower())
            if m:
                ev_only = m.group(1)
        print(f"[ASSETS] plots-src is CSV; generating arrow plots from: {plots_src}")
        _arrow_generate_plot_jsons_from_csv(plots_src, dst_root, ev_only=ev_only, flip_s2l=flip_s2l)
        return

    # Copy plot jsons from a directory
    if not os.path.exists(plots_src):
        print(f"[ASSETS] WARNING: plots-src not found: {plots_src}")
        return

    copied = 0
    for r, _dirs, files in os.walk(plots_src):
        for fn in files:
            if fn.lower().endswith(".plot.json"):
                src = os.path.join(r, fn)
                dst = os.path.join(dst_root, fn)
                copy_file_if_needed(src, dst)
                copied += 1
    print(f"[ASSETS] Plot JSON staged: copied={copied}, dst='{dst_root}'")


def stage_balls(balls_src: str, out_dir: str, balls_prefix: str):
    if not balls_src:
        print("[ASSETS] balls-src not provided; chrome balls will not be copied.")
        return
    balls_src = os.path.normpath(balls_src)
    if not os.path.exists(balls_src):
        print(f"[ASSETS] WARNING: balls-src not found: {balls_src}")
        return

    dst_root = os.path.join(out_dir, balls_prefix)
    ensure_dir(dst_root)

    copied = 0
    for r, _dirs, files in os.walk(balls_src):
        parts = [p.lower() for p in os.path.normpath(r).split(os.sep)]
        if "square" not in parts:
            continue

        for fn in files:
            low = fn.lower()
            if low.endswith(".png") and ("_ev-00" in low or "_ev-25" in low or "_ev-50" in low):
                src = os.path.join(r, fn)
                dst = os.path.join(dst_root, fn)
                shutil.copy2(src, dst)   # overwrite
                copied += 1

    print(f"[ASSETS] Chrome balls staged: copied={copied}, dst='{dst_root}'")


def generate_html_light(site_data_prefix: str, img_prefix: str, plots_prefix: str, balls_prefix: str, flip_s2l: bool, flip_dir: bool):
    """
    Full 2D + 3D panel + arrows + balls + neighbors.
    The only functional change: Plotly traces are configured to remain clickable without creating large hover/text arrays.
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Interactive Light Direction</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Helvetica Neue', Arial, sans-serif;
         margin:0; padding:0; background:#f2f2f7; color:#1d1d1f; }}
  .container {{ max-width:1480px; margin:32px auto; padding:28px;
               background:rgba(255,255,255,0.86);
               border:0.5px solid rgba(0,0,0,0.06);
               border-radius:14px;
               box-shadow:0 8px 32px rgba(0,0,0,0.06);
               backdrop-filter: blur(18px); }}
  .title {{ text-align:center; font-size:30px; font-weight:650; margin:0 0 10px; letter-spacing:-0.02em; }}
  .row {{ display:flex; gap:18px; align-items:flex-start; }}
  .left {{ flex:1; min-width:720px; background:rgba(255,255,255,0.92);
          border:0.5px solid rgba(0,0,0,0.06); border-radius:14px; padding:18px;
          box-shadow:0 4px 16px rgba(0,0,0,0.06); }}
  .right {{ width:440px; background:rgba(255,255,255,0.92);
           border:0.5px solid rgba(0,0,0,0.06); border-radius:14px; padding:18px;
           box-shadow:0 4px 16px rgba(0,0,0,0.06); position:sticky; top:18px; }}
  #plot {{ width:100%; height:680px; border-radius:12px; }}

  .controls {{ display:flex; flex-wrap:wrap; gap:10px; margin-bottom:12px; justify-content:space-between; }}
  .control-box {{ flex:1; min-width:240px; background:rgba(0,0,0,0.03);
                 border:1px solid rgba(0,0,0,0.05); border-radius:12px; padding:12px; }}
  .control-title {{ font-size:13px; font-weight:650; color:#374151; margin-bottom:8px; }}
  select, input[type="text"] {{ width:100%; padding:10px 12px; border-radius:10px;
                               border:1px solid rgba(0,0,0,0.10); background:white; outline:none; font-size:14px; }}
  .hint {{ font-size:12px; color:#6b7280; margin-top:6px; }}

  .pillbar {{ display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }}
  .pill {{ padding:6px 10px; border-radius:999px; background:rgba(0,0,0,0.04);
          border:1px solid rgba(0,0,0,0.06); font-size:12px; color:#374151; }}

  .btnrow {{ display:flex; gap:10px; margin-top:10px; flex-wrap:wrap; }}
  .btn {{ flex:1; min-width:160px; padding:10px 12px; border-radius:999px; border:0; cursor:pointer;
         background:rgba(0,0,0,0.04); color:#1d1d1f; font-weight:600; }}
  .btn:hover {{ background:rgba(0,122,255,0.12); color:#007aff; }}
  .btn.active {{ background:#007aff; color:#ffffff; }}

  .rankbar {{ display:flex; gap:10px; margin-top:10px; flex-wrap:wrap; }}
  .rankbtn {{
    padding:9px 12px;
    border-radius:999px;
    border:1px solid rgba(0,0,0,0.10);
    background:white;
    cursor:pointer;
    font-weight:750;
    font-size:12px;
    color:#111827;
    box-shadow: 0 2px 0 rgba(0,0,0,0.18), 0 8px 18px rgba(0,0,0,0.08);
    user-select:none;
    opacity: 1.0;
  }}
  .rankbtn.off {{ opacity:0.35; }}

  .panel-title {{ font-size:18px; font-weight:700; margin:0 0 10px; }}
  .empty {{ padding:34px 14px; text-align:center; color:#9CA3AF;
           border:2px dashed rgba(0,0,0,0.10); border-radius:14px; background:rgba(248,249,250,0.6); }}

  .kv {{ margin-top:12px; border:1px solid rgba(0,0,0,0.06); border-radius:12px; overflow:hidden; }}
  .kvrow {{ display:flex; justify-content:space-between; padding:10px 12px;
           border-bottom:1px solid rgba(0,0,0,0.06); background:rgba(255,255,255,0.9); }}
  .kvrow:last-child {{ border-bottom:0; }}
  .k {{ color:#6b7280; font-size:13px; font-weight:600; }}
  .v {{ color:#111827; font-size:13px; font-weight:650; text-align:right;
       max-width:260px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}

  table.ranktbl {{ width:100%; border-collapse:collapse; font-size:12px; margin-top:8px; }}
  table.ranktbl th, table.ranktbl td {{ padding:6px 6px; border-bottom:1px solid rgba(0,0,0,0.07); vertical-align:top; }}
  table.ranktbl th {{ text-align:left; color:#374151; font-weight:700; background:rgba(0,0,0,0.02); }}

  .badge {{ display:inline-block; padding:3px 8px; border-radius:999px;
           background:rgba(0,0,0,0.05); border:1px solid rgba(0,0,0,0.07);
           font-size:12px; font-weight:650; color:#374151; }}

  .imgwrap {{ margin-top:12px; }}
  .imgwrap img {{ width:100%; border-radius:12px; box-shadow:0 10px 26px rgba(0,0,0,0.18); }}

  .plotwrap {{ margin-top:12px; border:1px solid rgba(0,0,0,0.06); border-radius:12px; overflow:hidden; background:white; }}
  #arrowPlotWrapper {{ height: 260px; max-height: 260px; width: 100%; overflow: hidden; border-radius: 12px; background: #f8fafc; }}
  #arrowPlot {{ width:100%; height:100%; }}
  .smallhint {{ font-size:12px; color:#6b7280; margin-top:8px; }}

  .ball-stack {{ display:flex; flex-direction:column; gap:12px; margin-top:12px; }}
  .ball-card {{ background:white; border:1px solid rgba(0,0,0,0.08); border-radius:14px; padding:10px; overflow:hidden; }}
  .ball-card img {{ width:100%; height:auto; max-height:520px; object-fit:contain; background:#fff; border-radius:12px; display:block; }}
  .ball-label {{ margin-top:8px; font-weight:800; color:#111827; }}
  .ball-fn {{ margin-top:4px; font-size:12px; color:#6b7280; word-break:break-all; }}
</style>
</head>
<body>
  <div class="container">
    <h1 class="title">Interactive 3D Light Direction</h1>

    <div class="controls">
      <div class="control-box">
        <div class="control-title">Artist filter</div>
        <select id="artistSelect"></select>
      </div>

      <div class="control-box">
        <div class="control-title">Genre filter</div>
        <select id="genreSelect"></select>
      </div>

      <div class="control-box">
        <div class="control-title">Search (painting id)</div>
        <input id="searchBox" type="text" placeholder="Type to find..." />
        <div class="hint">Click a point to load details.</div>
      </div>
    </div>

    <div class="pillbar" id="statsPills"></div>

    <div class="row">
      <div class="left">
        <div class="btnrow">
          <button class="btn" id="resetBtn">Reset View</button>
          <button class="btn" id="clearBtn">Clear Selection (ESC)</button>
        </div>

        <div class="rankbar">
          <button id="rk1" class="rankbtn">Rank 1</button>
          <button id="rk2" class="rankbtn">Rank 2</button>
          <button id="rk3" class="rankbtn">Rank 3</button>
        </div>

        <div style="margin-top:12px;">
          <div id="plot"></div>
        </div>
      </div>

      <div class="right">
        <div class="panel-title">Sample Details</div>

        <div class="btnrow" style="margin-top: 0;">
          <button id="view2dBtn" class="btn active">2D View</button>
          <button id="view3dBtn" class="btn">3D View</button>
        </div>

        <div id="details2d">
          <div id="details"><div class="empty">Click a point to view details.</div></div>
        </div>

        <div id="details3d" style="display:none;">
          <div id="details3dInner"><div class="empty">Click a point to view details.</div></div>
        </div>
      </div>
    </div>
  </div>

<script>
const SITE_DATA_PREFIX = {json.dumps(site_data_prefix)};
const IMG_PREFIX = {json.dumps(img_prefix)};
const PLOTS_PREFIX = {json.dumps(plots_prefix)};
const BALLS_PREFIX = {json.dumps(balls_prefix)};

const FLIP_S2L_FOR_SPHERE = {str(bool(flip_s2l)).lower()};
const FLIP_DIR_FOR_SPHERE = {str(bool(flip_dir and not flip_s2l)).lower()};

function safeFilename(name) {{
  const s = String(name ?? "").replace(/[^A-Za-z0-9._-]+/g, "_").replace(/^_+|_+$/g, "");
  return s ? s : "untitled";
}}
function escapeHtml(s){{
  return String(s ?? "")
    .replaceAll("&","&amp;")
    .replaceAll("<","&lt;")
    .replaceAll(">","&gt;")
    .replaceAll('"',"&quot;")
    .replaceAll("'","&#039;");
}}
function fmtPct(x) {{
  if (x == null || Number.isNaN(x)) return '—';
  return (x*100).toFixed(0) + '%';
}}
function clamp(x, lo, hi){{ return Math.max(lo, Math.min(hi, x)); }}
function angleDegBetween(a, b){{
  const dot = clamp(a[0]*b[0] + a[1]*b[1] + a[2]*b[2], -1.0, 1.0);
  return Math.acos(dot) * 180.0 / Math.PI;
}}

// deterministic color
function hashString(str) {{
  let h = 2166136261;
  for (let i=0;i<str.length;i++){{ h ^= str.charCodeAt(i); h = Math.imul(h, 16777619); }}
  return h >>> 0;
}}
const MUTED = ["#5B8E7D","#F2D0A4","#B5838D","#6D597A","#A8C686","#E6B8A2","#7EA8BE","#CDB4DB","#9A8C98","#DDBEA9"];
const artistColorMap = new Map();
function getArtistColor(artist) {{
  if (artistColorMap.has(artist)) return artistColorMap.get(artist);
  const idx = hashString(String(artist)) % MUTED.length;
  const c = MUTED[idx];
  artistColorMap.set(artist, c);
  return c;
}}

let dataAll = [];
let dataFiltered = [];
let enabledRanks = new Set([1,2,3]);
let byPainting = new Map();
let repByPainting = new Map();
let lastSelectedPainting = null;

const artistSelect = document.getElementById('artistSelect');
const genreSelect  = document.getElementById('genreSelect');
const searchBox    = document.getElementById('searchBox');
const plotDiv      = document.getElementById('plot');

const defaultRanges = {{ x: [-1.05, 1.05], y: [-1.05, 1.05], z: [-1.05, 1.05] }};
const cameraYZ = {{ eye: {{x: 2.6, y: 0.0, z: 0.0}}, center: {{x:0,y:0,z:0}}, up: {{x:0,y:0,z:1}} }};
let currentSphereCamera = cameraYZ;

const baseLayout = {{
  scene: {{
    xaxis: {{ title: 'dir_x', range: defaultRanges.x, autorange:false, fixedrange:true }},
    yaxis: {{ title: 'dir_y', range: defaultRanges.y, autorange:false, fixedrange:true }},
    zaxis: {{ title: 'dir_z', range: defaultRanges.z, autorange:false, fixedrange:true }},
    camera: cameraYZ,
    aspectmode: 'cube',
    bgcolor: '#ffffff'
  }},
  margin: {{t:10, b:10, l:10, r:10}},
  paper_bgcolor: '#ffffff',
  plot_bgcolor: '#ffffff',
  showlegend: false,
  // helps reduce hover work
  hovermode: false
}};
const config = {{ responsive: true, displayModeBar: false, displaylogo: false }};

function pointSize() {{ return 3.6; }} // unchanged

// highlight
const HILITE_NAME = '__selected__';
function highlightTrace() {{
  return {{
    type: 'scatter3d',
    mode: 'markers',
    x: [null], y: [null], z: [null],
    marker: {{
      size: pointSize() * 2.2,
      color: '#FF2D55',
      opacity: 1.0,
      line: {{ width: 3, color: 'rgba(255,255,255,0.95)' }}
    }},
    hoverinfo: 'skip',
    showlegend: false,
    name: HILITE_NAME
  }};
}}
function getHighlightIndex() {{
  try {{
    const data = plotDiv?.data;
    if (!data || !Array.isArray(data)) return -1;
    for (let i = data.length - 1; i >= 0; i--) {{
      if (data[i]?.name === HILITE_NAME) return i;
    }}
  }} catch (e) {{}}
  return -1;
}}
function setHighlightPoint(x, y, z) {{
  const idx = getHighlightIndex();
  if (idx < 0) return;
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return;
  Plotly.restyle(plotDiv, {{ x: [[x]], y: [[y]], z: [[z]] }}, [idx]);
}}
function clearHighlightPoint() {{
  const idx = getHighlightIndex();
  if (idx < 0) return;
  Plotly.restyle(plotDiv, {{ x: [[null]], y: [[null]], z: [[null]] }}, [idx]);
}}

function syncRankButtons(){{
  [1,2,3].forEach(rk => {{
    const el = document.getElementById(`rk${{rk}}`);
    if (!el) return;
    if (enabledRanks.has(rk)) el.classList.remove("off");
    else el.classList.add("off");
  }});
}}

function makeTrace(points, rank, opacity) {{
  if (!enabledRanks.has(rank)) {{
    return {{
      x: [], y: [], z: [],
      mode: 'markers',
      type: 'scatter3d',
      marker: {{ size: pointSize(), opacity: 0.0 }},
      hoverinfo: 'skip',
      showlegend: false,
      customdata: []
    }};
  }}
  const pts = points.filter(d => d.rank === rank);
  return {{
    x: pts.map(d => d.x),
    y: pts.map(d => d.y),
    z: pts.map(d => d.z),
    mode: 'markers',
    type: 'scatter3d',
    marker: {{
      size: pointSize(),
      symbol: 'circle',
      color: pts.map(d => getArtistColor(d.artist)),
      opacity: opacity,
      line: {{ color: 'rgba(255,255,255,0.6)', width: 0.5 }}
    }},
    customdata: pts.map(d => d.idx),

    // key part:
    hoverinfo: 'none',
    hovertemplate: '<extra></extra>',

    showlegend: false
  }};
}}

function makeTraces(points) {{
  return [
    makeTrace(points, 3, 0.25),
    makeTrace(points, 2, 0.50),
    makeTrace(points, 1, 1.00),
    highlightTrace()
  ];
}}

function updateStatsPills() {{
  const pillbar = document.getElementById('statsPills');
  pillbar.innerHTML = '';
  const totalPts = dataFiltered.length;
  const totalAll = dataAll.length;
  const nPaintings = new Set(dataFiltered.map(d => d.painting_id)).size;
  const nArtists = new Set(dataFiltered.map(d => d.artist)).size;
  const nGenres  = new Set(dataFiltered.map(d => d.genre)).size;

  const mk = (txt) => {{ const p=document.createElement('div'); p.className='pill'; p.textContent=txt; pillbar.appendChild(p); }};
  mk(`Showing: ${{totalPts}} / ${{totalAll}} points`);
  mk(`Paintings: ${{nPaintings}}`);
  mk(`Artists: ${{nArtists}} | Genres: ${{nGenres}}`);

  let conv = "as-is";
  if (FLIP_S2L_FOR_SPHERE) conv = "match --flip-s2l (negate xyz, then flip y)";
  else if (FLIP_DIR_FOR_SPHERE) conv = "surface→light (negate xyz)";
}}

function resetView() {{
  Plotly.relayout(plotDiv, {{
    'scene.xaxis.range': defaultRanges.x,
    'scene.yaxis.range': defaultRanges.y,
    'scene.zaxis.range': defaultRanges.z,
    'scene.camera': currentSphereCamera
  }});
}}

function clearSelection() {{
  document.getElementById('details').innerHTML =
    '<div class="empty">Click a point to view details.</div>';
  document.getElementById('details3dInner').innerHTML =
    '<div class="empty">Click a point to view details.</div>';
  lastSelectedPainting = null;
  clearHighlightPoint();
  resetView();
}}
document.addEventListener('keydown', (e) => {{ if (e.key === 'Escape') clearSelection(); }});

function showRightPanel(mode){{
  const rightPanelMode = (mode === '3d') ? '3d' : '2d';
  const d2 = document.getElementById('details2d');
  const d3 = document.getElementById('details3d');
  const b2 = document.getElementById('view2dBtn');
  const b3 = document.getElementById('view3dBtn');
  d2.style.display = (rightPanelMode === '2d') ? 'block' : 'none';
  d3.style.display = (rightPanelMode === '3d') ? 'block' : 'none';
  b2.classList.toggle('active', rightPanelMode === '2d');
  b3.classList.toggle('active', rightPanelMode === '3d');
}}

async function tryFetchJson(urls){{
  for (const url of urls){{
    try {{
      const resp = await fetch(url);
      if (resp && resp.ok) return {{ url, payload: await resp.json() }};
    }} catch(e) {{}}
  }}
  return null;
}}

// robust basename candidates so plots show even if IDs differ slightly
function arrowCandidates(d){{
  const out = [];
  const pid = d.painting_id ?? "";
  const rid = d.rank_id ?? "";
  const img = d.img_file ?? "";

  const baseImg = img.replace(/\\.[^.]+$/, ""); // strip ext
  out.push(safeFilename(pid));
  out.push(safeFilename(rid));
  out.push(safeFilename(baseImg));
  out.push(safeFilename(String(pid).replace(/^R\\d+[_:\\-]+/i, "")));
  return Array.from(new Set(out.filter(s => s && s !== "untitled")));
}}

async function loadArrowPlot3D(d) {{
  const plotDiv2 = document.getElementById('arrowPlot');
  if (!plotDiv2) return;

  const baseUrl = PLOTS_PREFIX.replaceAll('\\\\','/');
  const candidates = arrowCandidates(d);

  const urls = [];
  for (const c of candidates){{
    urls.push(encodeURI(baseUrl + "/" + c + ".plot.json"));
  }}

  plotDiv2.innerHTML = "";
  const got = await tryFetchJson(urls);

  if (!got) {{
    plotDiv2.innerHTML = `<div style="padding:12px; color:#b91c1c; font-weight:700;">
      Arrow plot not found
      <div style="margin-top:6px; font-weight:600; color:#6b7280;">
        Tried:<br>${{urls.map(u=>escapeHtml(u)).join("<br>")}}
      </div>
    </div>`;
    return;
  }}

  const payload = got.payload;
  const layout = Object.assign({{}}, payload.layout || {{}});
  layout.height = 260;
  layout.margin = layout.margin || {{l:10, r:10, t:40, b:10}};
  layout.scene = layout.scene || {{}};
  layout.scene.camera = cameraYZ;
  layout.scene.aspectmode = layout.scene.aspectmode || 'cube';

  try {{ Plotly.purge(plotDiv2); }} catch (e) {{}}
  Plotly.react(plotDiv2, payload.data, layout, {{displaylogo:false, displayModeBar:true, responsive:true}});
}}

function render3DPanel(d){{
  const wrap = document.getElementById('details3dInner');
  const pid = String(d.painting_id ?? "");
  const artist = String(d.artist ?? "Unknown");
  const genre = String(d.genre ?? "Unknown");

  const baseCandidates = Array.from(new Set([
    pid,
    safeFilename(pid)
  ].filter(Boolean)));

  wrap.innerHTML = `
    <div class="kv">
      <div class="kvrow"><div class="k">Painting ID</div><div class="v" title="${{escapeHtml(pid)}}">${{escapeHtml(pid)}}</div></div>
      <div class="kvrow"><div class="k">Artist</div><div class="v" title="${{escapeHtml(artist)}}">${{escapeHtml(artist)}}</div></div>
      <div class="kvrow"><div class="k">Genre</div><div class="v" title="${{escapeHtml(genre)}}">${{escapeHtml(genre)}}</div></div>
    </div>

    <div style="margin-top:14px; font-weight:800; color:#111827; font-size:20px;">Chrome balls (ev-00 / ev-25 / ev-50)</div>
    <div style="font-size:12px; color:#6b7280; margin-top:6px;">
      Loaded from <b>${{escapeHtml(BALLS_PREFIX)}}/</b> trying raw and sanitized filenames.
    </div>

    <div class="ball-stack" id="ballStack"></div>
  `;

  const stack = document.getElementById("ballStack");
  const evTags = ["ev-00", "ev-25", "ev-50"];

  evTags.forEach(tag => {{
    const card = document.createElement("div");
    card.className = "ball-card";

    const img = document.createElement("img");
    img.alt = tag;

    const label = document.createElement("div");
    label.className = "ball-label";
    label.textContent = tag;

    const fn = document.createElement("div");
    fn.className = "ball-fn";
    fn.textContent = "Trying...";

    const candidates = baseCandidates.map(base => ({{
      fn: `${{base}}_${{tag}}.png`,
      url: encodeURI(BALLS_PREFIX.replaceAll('\\\\','/') + "/" + `${{base}}_${{tag}}.png`)
    }}));

    let idx = 0;

    function tryNext() {{
      if (idx >= candidates.length) {{
        img.removeAttribute("src");
        fn.textContent = "Not found";
        return;
      }}
      const cand = candidates[idx++];
      fn.textContent = cand.fn;
      img.src = cand.url;
    }}

    img.onerror = tryNext;
    img.onload = () => {{
      fn.textContent = candidates[Math.max(0, idx - 1)].fn;
    }};

    tryNext();

    card.appendChild(img);
    card.appendChild(label);
    card.appendChild(fn);
    stack.appendChild(card);
  }});
}}

function getTop3Neighbors(selectedPaintingId){{
  const selRep = repByPainting.get(selectedPaintingId);
  if (!selRep) return [];
  const a = [selRep.x, selRep.y, selRep.z];

  const visiblePaintings = new Set(dataFiltered.map(d => d.painting_id));
  const cand = [];
  for (const pid of visiblePaintings){{
    if (pid === selectedPaintingId) continue;
    const rep = repByPainting.get(pid);
    if (!rep) continue;
    const b = [rep.x, rep.y, rep.z];
    const ang = angleDegBetween(a, b);
    cand.push({{ pid, rep, ang }});
  }}
  cand.sort((u,v) => u.ang - v.ang);
  return cand.slice(0, 3);
}}

function showDetailsBundle(d) {{
  const details = document.getElementById('details');

  const siblings = byPainting.get(d.painting_id) ?? [d];
  lastSelectedPainting = d.painting_id;

  const dom = (d.dominance == null || Number.isNaN(d.dominance))
    ? '—'
    : (Number(d.dominance)*100).toFixed(1) + '%';

  const rows = siblings.map(s => {{
    const rk = (s.rank == null) ? '—' : 'R' + String(s.rank);
    const confText = fmtPct(s.confidence);
    const dirText  = `(${{Number(s.x).toFixed(3)}}, ${{Number(s.y).toFixed(3)}}, ${{Number(s.z).toFixed(3)}})`;
    return `
      <tr>
        <td><span class="badge">${{rk}}</span></td>
        <td>${{confText}}</td>
        <td style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;">
          ${{dirText}}
        </td>
      </tr>
    `;
  }}).join('');

  const neigh = getTop3Neighbors(d.painting_id);
  const neighHTML = (neigh.length === 0) ? `<div class="k" style="margin-top:8px;">—</div>` : `
    <div style="margin-top:10px; display:grid; grid-template-columns: 1fr 1fr 1fr; gap:10px;">
      ${{neigh.map(n => {{
        const imgUrl = encodeURI(IMG_PREFIX.replaceAll('\\\\','/') + '/' + (n.rep.img_file ?? ""));
        return `
          <div style="background:white; border:1px solid rgba(0,0,0,0.08); border-radius:12px; padding:8px; cursor:pointer;"
               onclick="window.jumpToPainting('${{String(n.pid).replaceAll("'", "\\\\'")}}')">
            <img src="${{imgUrl}}" style="width:100%; height:100px; object-fit:cover; border-radius:10px;">
            <div style="margin-top:6px; font-size:11px; color:#374151; line-height:1.25;">
              <div style="font-weight:750; color:#111827;" title="${{escapeHtml(n.pid)}}">${{escapeHtml(n.pid)}}</div>
              <div>${{escapeHtml(n.rep.artist ?? "Unknown")}}</div>
              <div style="color:#6b7280;">Δ angle: ${{n.ang.toFixed(2)}}°</div>
            </div>
          </div>
        `;
      }}).join('')}}
    </div>
  `;

  const imgUrl = encodeURI(IMG_PREFIX.replaceAll('\\\\','/') + '/' + (d.img_file ?? ""));

  details.innerHTML = `
    <div class="kv">
      <div class="kvrow"><div class="k">Painting ID</div><div class="v" title="${{escapeHtml(d.painting_id)}}">${{escapeHtml(d.painting_id)}}</div></div>
      <div class="kvrow"><div class="k">Artist</div><div class="v" title="${{escapeHtml(d.artist)}}">${{escapeHtml(d.artist)}}</div></div>
      <div class="kvrow"><div class="k">Genre</div><div class="v" title="${{escapeHtml(d.genre)}}">${{escapeHtml(d.genre)}}</div></div>
      <div class="kvrow"><div class="k">Dominance</div><div class="v">${{escapeHtml(dom)}}</div></div>
    </div>

    <div class="imgwrap"><img src="${{imgUrl}}" alt="painting" /></div>

    <div class="plotwrap">
      <div id="arrowPlotWrapper"><div id="arrowPlot"></div></div>
    </div>
    <div class="smallhint">
      Loading arrow from <b>${{escapeHtml(PLOTS_PREFIX)}}/</b> using one of:
      <code>${{arrowCandidates(d).map(c => escapeHtml(c + ".plot.json")).join("</code>, <code>")}}</code>
    </div>

    <div style="margin-top:12px; font-weight:800; color:#111827;">All ranks for this painting</div>
    <table class="ranktbl">
      <thead><tr><th>Rank</th><th>Confidence</th><th>dir</th></tr></thead>
      <tbody>${{rows}}</tbody>
    </table>

    <div style="margin-top:12px; font-weight:800; color:#111827;">Similar paintings</div>
    ${{neighHTML}}
  `;

  loadArrowPlot3D(d);
  render3DPanel(d);
}}

window.jumpToPainting = function(pid){{
  const rep = repByPainting.get(pid);
  if (!rep) return;
  setHighlightPoint(rep.x, rep.y, rep.z);
  showDetailsBundle(rep);
}};

// Plotly click binding (with tiny guard to avoid double-fire)
let clickBusy = false;
function bindPlotClickHandler(){{
  try {{ plotDiv.removeAllListeners('plotly_click'); }} catch(e) {{}}
  plotDiv.on('plotly_click', (evt) => {{
    if (clickBusy) return;
    clickBusy = true;
    requestAnimationFrame(() => {{
      try {{
        const pt = evt.points?.[0];
        const idx = pt?.customdata;
        const nidx = Number(idx);
        if (!Number.isFinite(nidx)) return;

        const d = dataAll[nidx];
        if (!d) return;

        setHighlightPoint(d.x, d.y, d.z);
        showDetailsBundle(d);
        showRightPanel('2d');
      }} finally {{
        clickBusy = false;
      }}
    }});
  }});
}}

function applyFilters() {{
  const a = artistSelect.value;
  const g = genreSelect.value;
  const q = searchBox.value.trim().toLowerCase();

  dataFiltered = dataAll.filter(d => {{
    const okA = (a === '__ALL__') ? true : d.artist === a;
    const okG = (g === '__ALL__') ? true : d.genre === g;
    const okQ = (!q) ? true : String(d.painting_id ?? "").toLowerCase().includes(q);
    return okA && okG && okQ;
  }});

  updateStatsPills();
  syncRankButtons();

  Plotly.react(plotDiv, makeTraces(dataFiltered), baseLayout, config).then(() => {{
    bindPlotClickHandler();
    resetView();
  }});
}}

function initFilters() {{
  const artists = Array.from(new Set(dataAll.map(d => d.artist))).sort((a,b)=>a.localeCompare(b));
  const genres  = Array.from(new Set(dataAll.map(d => d.genre))).sort((a,b)=>a.localeCompare(b));

  artistSelect.innerHTML = '<option value="__ALL__">All artists</option>' + artists.map(a => `<option value="${{escapeHtml(a)}}">${{escapeHtml(a)}}</option>`).join('');
  genreSelect.innerHTML  = '<option value="__ALL__">All genres</option>' + genres.map(g => `<option value="${{escapeHtml(g)}}">${{escapeHtml(g)}}</option>`).join('');

  artistSelect.onchange = applyFilters;
  genreSelect.onchange = applyFilters;
  searchBox.oninput = applyFilters;

  document.getElementById('rk1').onclick = () => toggleRank(1);
  document.getElementById('rk2').onclick = () => toggleRank(2);
  document.getElementById('rk3').onclick = () => toggleRank(3);

  document.getElementById('resetBtn').onclick = resetView;
  document.getElementById('clearBtn').onclick = clearSelection;

  document.getElementById('view2dBtn').onclick = () => showRightPanel('2d');
  document.getElementById('view3dBtn').onclick = () => showRightPanel('3d');
}}

function toggleRank(rk){{
  if (enabledRanks.has(rk)) enabledRanks.delete(rk);
  else enabledRanks.add(rk);
  if (enabledRanks.size === 0) enabledRanks.add(rk);
  syncRankButtons();
  applyFilters();
}}

async function loadPoints() {{
  const url = encodeURI(SITE_DATA_PREFIX + "/points.json");
  const resp = await fetch(url);
  if (!resp.ok) {{
    throw new Error("Failed to fetch " + url + " (" + resp.status + ")");
  }}
  const payload = await resp.json();
  return payload.points || [];
}}

function buildIndices() {{
  byPainting = new Map();
  for (const d of dataAll) {{
    const key = d.painting_id;
    if (!byPainting.has(key)) byPainting.set(key, []);
    byPainting.get(key).push(d);
  }}
  byPainting.forEach(arr => arr.sort((a,b) => ((a.rank ?? 99) - (b.rank ?? 99))));

  repByPainting = new Map();
  byPainting.forEach((arr, key) => {{
    let rep = arr.find(x => x.rank === 1) ?? arr[0];
    repByPainting.set(key, rep);
  }});
}}

function normalizeAndFlip() {{
  // ensure numeric rank for filtering + numeric idx for click lookup
  dataAll.forEach(d => {{
    d.rank = (d.rank == null) ? null : Number(d.rank);
    d.idx  = Number(d.idx);
    let x = Number(d.x), y = Number(d.y), z = Number(d.z);

    if (FLIP_S2L_FOR_SPHERE) {{
      x = -x; y = -y; z = -z;
      y = -y;
    }} else if (FLIP_DIR_FOR_SPHERE) {{
      x = -x; y = -y; z = -z;
    }}

    const n = Math.sqrt(x*x + y*y + z*z);
    if (n > 1e-12) {{ x/=n; y/=n; z/=n; }}
    d.x = x; d.y = y; d.z = z;
  }});
}}

(async function main(){{
  try {{
    dataAll = await loadPoints();
    normalizeAndFlip();
    buildIndices();
    initFilters();

    dataFiltered = dataAll.slice();
    updateStatsPills();
    syncRankButtons();

    await Plotly.newPlot(plotDiv, makeTraces(dataFiltered), baseLayout, config);
    Plotly.relayout(plotDiv, {{ 'scene.camera': cameraYZ }});
    bindPlotClickHandler();
    showRightPanel('2d');

  }} catch (e) {{
    console.error(e);
    document.getElementById('details').innerHTML =
      `<div class="empty" style="color:#b91c1c; border-color:rgba(185,28,28,0.35);">
        Failed to load points.json<br><br>
        <div style="font-size:12px; color:#6b7280;">${{escapeHtml(String(e))}}</div>
      </div>`;
  }}
}})();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--file-name", type=str, required=True)

    parser.add_argument("--img-src", type=str, required=True)
    parser.add_argument("--plots-src", type=str, default="")
    parser.add_argument("--balls-src", type=str, default="")

    parser.add_argument("--out-dir", type=str, required=True)

    parser.add_argument("--img-prefix", type=str, default="data")
    parser.add_argument("--plots-prefix", type=str, default="plots_embedded")
    parser.add_argument("--balls-prefix", type=str, default="balls")

    parser.add_argument("--site-data-prefix", type=str, default="site_data", help="Where points.json is written")

    parser.add_argument("--csv-path", type=str, default="")
    parser.add_argument("--conf-mae-good", type=float, default=1.0)
    parser.add_argument("--conf-mae-bad", type=float, default=8.0)

    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--flip-s2l", action="store_true")
    parser.add_argument("--flip-dir", action="store_true")

    args = parser.parse_args()

    pkl_path = os.path.normpath(os.path.join(args.data_path, args.file_name))
    if not os.path.exists(pkl_path):
        print(f"Pickle not found: {pkl_path}")
        return

    out_dir = os.path.normpath(args.out_dir)
    ensure_dir(out_dir)

    print(f"Loading data from: {pkl_path}")
    with open(pkl_path, "rb") as f:
        all_data = pickle.load(f)

    if not isinstance(all_data, list) or (all_data and not isinstance(all_data[0], dict)):
        print("Pickle isn't list[dict].")
        return

    desired_ev = _extract_ev_tag(args.file_name)

    # Optional confidence
    if args.csv_path:
        conf_map = load_confidence_map_by_id_and_rank(
            args.csv_path,
            desired_ev=desired_ev,
            E_good=args.conf_mae_good,
            E_bad=args.conf_mae_bad,
        )
        hit = 0
        miss = 0
        for d in all_data:
            try:
                rk = int(float(d.get("rank", 1)))
            except Exception:
                rk = 1

            candidates = [
                d.get("painting_info", ""),
                d.get("painting_rank_id", ""),
                d.get("pose_info", ""),
                d.get("img_path", ""),
                d.get("file", ""),
            ]

            conf = None
            for cand in candidates:
                if not cand:
                    continue
                base_id, _ = _normalize_id_and_ev(cand)
                if not base_id:
                    continue
                v = conf_map.get((base_id, rk))
                if v is not None:
                    conf = float(v)
                    break

            d["confidence"] = conf
            if conf is None:
                miss += 1
            else:
                hit += 1
        print(f"confidence matched={hit} | missing={miss} | total={len(all_data)}")
    else:
        print("csv-path not provided; confidence disabled (will show '—').")

    # Build lightweight points list
    used = []
    skipped = 0
    key_counter = Counter()

    for _i, d in enumerate(all_data):
        x, y, z, src = extract_direction_components(d)
        if x is None:
            skipped += 1
            continue

        if args.no_normalize:
            ux, uy, uz = x, y, z
            nraw = math.sqrt(x * x + y * y + z * z)
        else:
            ux, uy, uz, nraw = normalize_vec(x, y, z)
            if ux is None:
                skipped += 1
                continue

        # IMPORTANT: do NOT flip here; JS applies flip consistently.
        img_path = str(d.get("img_path", "") or "")
        img_file = os.path.basename(img_path.replace("\\", "/"))

        try:
            rank_val = None if d.get("rank", None) is None else int(float(d.get("rank", 0) or 0))
        except Exception:
            rank_val = None

        used.append(
            {
                "idx": len(used),  # stable key for click lookup
                "painting_id": str(d.get("painting_info", "") or d.get("pose_info", "") or ""),
                "rank_id": str(d.get("painting_rank_id", "") or ""),
                "artist": str(d.get("dataset", "") or "Unknown"),
                "genre": str(d.get("genre_info", "") or "Unknown"),
                "rank": rank_val,
                "x": float(ux),
                "y": float(uy),
                "z": float(uz),
                "dominance": None if d.get("dominance", None) is None else float(d.get("dominance")),
                "confidence": None if d.get("confidence", None) is None else float(d.get("confidence")),
                "img_file": img_file,
                "dir_src_keys": src,
                "dir_norm_raw": None if (nraw is None or not math.isfinite(nraw)) else float(nraw),
            }
        )
        key_counter[src] += 1

    print(f"Physical xyz applied to: {len(used)} points | skipped(no physical dir): {skipped}")
    if len(used) == 0:
        print("ERROR: 0 points had physical direction keys.")
        return

    # Write site data (separate file -> faster / lighter HTML)
    site_data_dir = os.path.join(out_dir, args.site_data_prefix)
    ensure_dir(site_data_dir)
    points_path = os.path.join(site_data_dir, "points.json")
    with open(points_path, "w", encoding="utf-8") as f:
        json.dump({"points": used}, f, ensure_ascii=False, separators=(",", ":"), default=_to_json_safe)
    print(f"[SITE_DATA] Wrote: {points_path}")

    # Stage assets
    stage_painting_images(used, args.img_src, out_dir, args.img_prefix, debug=args.debug)

    if args.plots_src:
        stage_plots(args.plots_src, out_dir, args.plots_prefix, desired_ev_tag=desired_ev, flip_s2l=args.flip_s2l)

    if args.balls_src:
        stage_balls(args.balls_src, out_dir, args.balls_prefix)

    # Write HTML
    html_content = generate_html_light(
        site_data_prefix=args.site_data_prefix,
        img_prefix=args.img_prefix,
        plots_prefix=args.plots_prefix,
        balls_prefix=args.balls_prefix,
        flip_s2l=args.flip_s2l,
        flip_dir=args.flip_dir,
    )
    out_html = os.path.join(out_dir, "index.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    print("\n" + "=" * 44)
    print(f"FINISH! Output file: {out_html}")
    print("Test locally from out-dir:")
    print(f'  cd "{out_dir}"')
    print("  python -m http.server 8000")
    print("  http://localhost:8000/index.html")
    print(f"\nArrow plots folder expected at: {os.path.join(out_dir, args.plots_prefix)}")
    print(f"Points JSON expected at: {points_path}")

    if args.flip_s2l:
        print("\nNOTE: --flip-s2l enabled (match create_3d_plots.py: negate xyz, then flip y).")
    elif args.flip_dir:
        print("\nNOTE: --flip-dir enabled (legacy: negate xyz only).")


if __name__ == "__main__":
    main()