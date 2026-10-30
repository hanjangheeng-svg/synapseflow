import html
import json
import os
import textwrap
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.autocircuit_core import (
    parse_circuit_spec_v2,
    run_physics_guided_agent,
)
from src.diff_pair_core import (
    run_nominal_agent,
    run_vcm_robust_agent,
    run_pvt_aware_agent,
)
from src.auto_surrogate import DEFAULT_DIFF_PAIR_NETLIST, parse_netlist
from src.efficient_optimizer import STRATEGY_LABELS, run_efficient_optimizer

st.set_page_config(
    page_title="SynapseFlow · Analog Design Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

LOCAL_WINDOWS_NGSPICE = r"C:\Users\'장희\Downloads\ngspice-47_64\Spice64\bin\ngspice_con.exe"
DEFAULT_NGSPICE = os.getenv("NGSPICE_PATH", "").strip()
if not DEFAULT_NGSPICE and os.name == "nt" and Path(LOCAL_WINDOWS_NGSPICE).exists():
    DEFAULT_NGSPICE = LOCAL_WINDOWS_NGSPICE

# Cloud deployments normally install ngspice into PATH via packages.txt.
# An Ollama endpoint is optional: when blank, the parser uses the deterministic
# value/unit fallback so the public demo does not depend on the developer PC.
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "").strip()
if not DEFAULT_OLLAMA_URL and os.name == "nt":
    DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b").strip() or "qwen3:4b"

# Theme is controlled inside the app so the demo behaves consistently
# regardless of the browser / Streamlit global theme.
dark_mode = bool(st.session_state.get("dark_mode", False))

if dark_mode:
    THEME = {
        "ink": "#F7FAFD", "sub": "#D0DCE6", "line": "#3B5367",
        "bg": "#07111B", "bg2": "#0A1622", "card": "#0F1D2A",
        "surface": "#132433", "soft": "#102232", "input": "#0C1925",
        "blue": "#8CC7F0", "blue_deep": "#83C4F1", "teal": "#87D8C5",
        "teal_soft": "#12312D", "warn": "#F0C36E", "warn_soft": "#332A17",
        "shadow": "rgba(0,0,0,.30)", "muted": "#B9C9D6",
    }
else:
    THEME = {
        "ink": "#0B1B2A", "sub": "#344E64", "line": "#C7D5E0",
        "bg": "#F4F7FA", "bg2": "#FBFDFE", "card": "#FFFFFF",
        "surface": "#F4F8FB", "soft": "#EAF3F8", "input": "#F7F9FB",
        "blue": "#236B9F", "blue_deep": "#173F60", "teal": "#256F62",
        "teal_soft": "#EAF7F3", "warn": "#8A5A10", "warn_soft": "#FFF5DC",
        "shadow": "rgba(25,55,80,.06)", "muted": "#49647A",
    }

st.markdown(
    f"""
    <style>
    :root {{
        --ink:{THEME['ink']}; --sub:{THEME['sub']}; --line:{THEME['line']};
        --bg:{THEME['bg']}; --bg2:{THEME['bg2']}; --card:{THEME['card']};
        --surface:{THEME['surface']}; --soft:{THEME['soft']}; --input:{THEME['input']};
        --blue:{THEME['blue']}; --blue-deep:{THEME['blue_deep']};
        --teal:{THEME['teal']}; --teal-soft:{THEME['teal_soft']};
        --warn:{THEME['warn']}; --warn-soft:{THEME['warn_soft']};
        --shadow:{THEME['shadow']}; --muted:{THEME['muted']};
    }}
    html, body, [class*="css"] {{ color:var(--ink); }}
    .stApp {{ background:linear-gradient(180deg,var(--bg2) 0,var(--bg) 300px,var(--bg) 100%); color:var(--ink); }}
    .block-container {{ max-width:1380px; padding-top:1.0rem; padding-bottom:4rem; }}
    header[data-testid="stHeader"] {{ display:none !important; }}
    div[data-testid="stToolbar"] {{ display:none !important; }}
    #MainMenu, footer {{ visibility:hidden !important; }}
    h1,h2,h3,h4,label,p {{ color:var(--ink); }}
    /* Cross-theme readability: never rely on Streamlit's default muted colors. */
    [data-testid="stMarkdownContainer"] {{ color:var(--ink); }}
    [data-testid="stMarkdownContainer"] small,
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p,
    [data-testid="stWidgetLabel"] p,
    div[role="radiogroup"] label,
    div[data-testid="stToggle"] label p {{ color:var(--sub) !important; opacity:1 !important; }}
    [data-testid="stWidgetLabel"] p {{ font-weight:650 !important; }}

    .ac-topbar {{ display:flex; align-items:center; justify-content:space-between; gap:1rem;
        padding:.55rem .1rem .85rem; border-bottom:1px solid var(--line); margin-bottom:.3rem; }}
    .ac-brand {{ display:flex; align-items:center; gap:.7rem; min-width:0; }}
    .ac-mark {{ width:38px;height:38px;border-radius:12px;display:grid;place-items:center;
        background:var(--blue-deep);color:{'#07111B' if dark_mode else '#FFFFFF'};font-weight:900;font-size:1.2rem; }}
    .ac-name {{ font-size:1.55rem;font-weight:850;letter-spacing:-.045em;color:var(--ink); }}
    .ac-tag {{ color:var(--sub);font-size:.82rem;border-left:1px solid var(--line);padding-left:.75rem; }}
    .ac-badge {{ padding:.4rem .7rem;border:1px solid var(--line);border-radius:999px;background:var(--surface);
        color:var(--sub);font-size:.72rem;font-weight:760;white-space:nowrap; }}
    .theme-row {{ margin:-.15rem 0 .55rem; }}

    .hero {{ background:var(--card);border:1px solid var(--line);border-radius:22px;padding:1.35rem 1.5rem;
        box-shadow:0 8px 25px var(--shadow); margin-bottom:1rem; }}
    .hero-title {{ font-size:2rem;font-weight:860;letter-spacing:-.045em;color:var(--ink); }}
    .hero-sub {{ color:var(--sub);font-size:.98rem;line-height:1.7;margin-top:.45rem;max-width:1040px; }}
    .proof-grid {{ display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.7rem;margin:0 0 1rem; }}
    .proof-card {{ background:var(--card);border:1px solid var(--line);border-radius:16px;padding:.9rem 1rem; }}
    .proof-value {{ font-size:1.35rem;font-weight:860;color:var(--blue-deep);letter-spacing:-.03em; }}
    .proof-label {{ color:var(--sub);font-size:.76rem;line-height:1.45;margin-top:.18rem; }}
    .mode-note {{ color:var(--sub);font-size:.78rem;line-height:1.55;margin:.35rem 0 .45rem; }}
    .scope-strip {{ display:flex;align-items:center;gap:.55rem;flex-wrap:wrap;margin:.15rem 0 .9rem; }}
    .scope-badge {{ display:inline-flex;align-items:center;padding:.28rem .52rem;border-radius:999px;border:1px solid var(--line);background:var(--surface);color:var(--ink);font-size:.69rem;font-weight:780; }}
    .scope-note {{ color:var(--sub);font-size:.72rem;line-height:1.45; }}
    .screening-callout {{ display:flex;align-items:center;justify-content:space-between;gap:1rem;flex-wrap:wrap;background:var(--surface);border:1px solid var(--line);border-radius:15px;padding:.82rem .95rem;margin:.65rem 0 .95rem; }}
    .screening-main {{ color:var(--ink);font-size:1.05rem;font-weight:850;letter-spacing:-.02em; }}
    .screening-main b {{ color:var(--blue-deep); }}
    .screening-sub {{ color:var(--sub);font-size:.73rem;line-height:1.45;max-width:620px; }}
    .diag-grid {{ display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.55rem;margin:.35rem 0 .65rem; }}
    .diag-card {{ background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:.68rem .75rem; }}
    .diag-label {{ color:var(--sub);font-size:.68rem;font-weight:720; }}
    .diag-value {{ color:var(--ink);font-size:.92rem;font-weight:830;margin-top:.12rem; }}
    .flow {{ display:flex;flex-wrap:wrap;gap:.45rem;margin-top:1rem;align-items:center; }}
    .flow span {{ padding:.35rem .58rem;background:var(--surface);border:1px solid var(--line);border-radius:9px;
        color:var(--sub);font-size:.76rem;font-weight:680; }}
    .flow b {{ color:var(--muted);font-weight:700; }}

    /* Animated design context */
    .design-context {{ margin:1.05rem 0 1.25rem; }}
    .context-head {{ display:flex;justify-content:space-between;align-items:flex-end;gap:1rem;margin-bottom:.6rem; }}
    .context-title {{ font-size:1.08rem;font-weight:840;color:var(--ink); }}
    .context-sub {{ color:var(--sub);font-size:.76rem;line-height:1.45;max-width:720px; }}
    .anim-grid {{ display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.75rem; }}
    .anim-card {{ position:relative;overflow:hidden;background:var(--card);border:1px solid var(--line);
        border-radius:18px;padding:.85rem .95rem .7rem;min-height:300px;box-shadow:0 8px 22px var(--shadow); }}
    .anim-card::after {{ content:"";position:absolute;inset:auto -80px -100px auto;width:220px;height:220px;
        border-radius:50%;background:radial-gradient(circle,rgba(75,157,210,.10),rgba(75,157,210,0) 70%);pointer-events:none; }}
    .anim-label {{ display:flex;justify-content:space-between;align-items:center;gap:.6rem;margin-bottom:.35rem; }}
    .anim-title {{ font-size:.92rem;font-weight:820;color:var(--ink); }}
    .anim-pill {{ font-size:.67rem;font-weight:760;color:var(--sub);background:var(--surface);border:1px solid var(--line);
        border-radius:999px;padding:.25rem .48rem; }}
    .anim-svg {{ width:100%;height:225px;display:block; }}
    .anim-caption {{ color:var(--sub);font-size:.72rem;line-height:1.45;margin-top:.1rem; }}
    .flow-line {{ stroke:var(--blue);stroke-width:2.2;stroke-dasharray:8 7;animation:acDash 1.15s linear infinite; }}
    .wire {{ stroke:var(--sub);stroke-width:2;fill:none;stroke-linecap:round;stroke-linejoin:round;opacity:.88; }}
    .wire-live {{ stroke:var(--teal);stroke-width:2.6;fill:none;stroke-linecap:round;stroke-dasharray:12 8;animation:acDash .9s linear infinite; }}
    .node-box {{ fill:var(--surface);stroke:var(--line);stroke-width:1.2; }}
    .node-text {{ fill:var(--ink);font-size:12px;font-weight:760;text-anchor:middle;dominant-baseline:middle; }}
    .node-sub {{ fill:var(--sub);font-size:9px;text-anchor:middle; }}
    .pulse1 {{ animation:nodePulse 5s ease-in-out infinite;transform-box:fill-box;transform-origin:center; }}
    .pulse2 {{ animation:nodePulse 5s 1s ease-in-out infinite;transform-box:fill-box;transform-origin:center; }}
    .pulse3 {{ animation:nodePulse 5s 2s ease-in-out infinite;transform-box:fill-box;transform-origin:center; }}
    .pulse4 {{ animation:nodePulse 5s 3s ease-in-out infinite;transform-box:fill-box;transform-origin:center; }}
    .pulse5 {{ animation:nodePulse 5s 4s ease-in-out infinite;transform-box:fill-box;transform-origin:center; }}
    .signal-dot {{ fill:var(--teal);filter:drop-shadow(0 0 5px rgba(70,190,160,.55)); }}
    .wave {{ fill:none;stroke:var(--blue);stroke-width:2.2;stroke-dasharray:220;stroke-dashoffset:220;animation:drawWave 3.2s ease-in-out infinite; }}
    .size80,.size100,.size140 {{ fill:var(--ink);font-size:11px;font-weight:800;text-anchor:middle;opacity:0; }}
    .size80 {{ animation:sizeSwap 6s 0s infinite; }} .size100 {{ animation:sizeSwap 6s 2s infinite; }} .size140 {{ animation:sizeSwap 6s 4s infinite; }}
    .cell {{ fill:var(--soft);stroke:var(--line);stroke-width:.9;animation:cellBlink 3.2s ease-in-out infinite; }}
    .cell:nth-of-type(2n) {{ animation-delay:.35s; }} .cell:nth-of-type(3n) {{ animation-delay:.7s; }}
    @keyframes acDash {{ to {{ stroke-dashoffset:-30; }} }}
    @keyframes nodePulse {{ 0%,16%,100% {{ transform:scale(1); }} 7% {{ transform:scale(1.055); }} }}
    @keyframes drawWave {{ 0% {{ stroke-dashoffset:220;opacity:.45; }} 45%,75% {{ stroke-dashoffset:0;opacity:1; }} 100% {{ stroke-dashoffset:-220;opacity:.4; }} }}
    @keyframes sizeSwap {{ 0%,28% {{ opacity:1; }} 33%,100% {{ opacity:0; }} }}
    @keyframes cellBlink {{ 0%,100% {{ opacity:.45; }} 50% {{ opacity:1; }} }}

    .section-title {{ font-size:1.1rem;font-weight:820;color:var(--ink);margin:.3rem 0 .75rem; }}
    .method-card {{ background:var(--card);border:1px solid var(--line);border-radius:12px;padding:.72rem .82rem;min-height:74px; }}
    .method-kicker {{ color:var(--sub);font-size:.69rem;font-weight:760;text-transform:uppercase;letter-spacing:.05em; }}
    .method-name {{ color:var(--ink);font-size:.92rem;font-weight:820;margin-top:.16rem; }}
    .method-note {{ color:var(--sub);font-size:.71rem;line-height:1.4;margin-top:.12rem; }}
    .result-shell {{ background:var(--card);border:1px solid var(--line);border-radius:18px;padding:1rem 1.15rem;margin:.8rem 0; }}
    .status-pass {{ padding:.85rem 1rem;border-radius:14px;background:var(--teal-soft);border:1px solid var(--line);
        color:var(--teal);font-weight:800; }}
    .status-fail {{ padding:.85rem 1rem;border-radius:14px;background:var(--warn-soft);border:1px solid var(--line);
        color:var(--warn);font-weight:800; }}
    .result-head {{ display:flex;align-items:center;justify-content:space-between;gap:1rem;flex-wrap:wrap;margin:.25rem 0 .7rem; }}
    .result-kicker {{ color:var(--sub);font-size:.75rem;font-weight:760;text-transform:uppercase;letter-spacing:.06em; }}
    .result-title {{ color:var(--ink);font-size:1.35rem;font-weight:860;letter-spacing:-.03em;margin-top:.08rem; }}
    .result-chip {{ padding:.42rem .68rem;border:1px solid var(--line);border-radius:999px;background:var(--teal-soft);
        color:var(--teal);font-size:.76rem;font-weight:800;white-space:nowrap; }}
    .margin-grid {{ display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.65rem;margin:.7rem 0 1rem; }}
    .margin-card {{ background:var(--card);border:1px solid var(--line);border-radius:15px;padding:.82rem .9rem; }}
    .margin-label {{ color:var(--sub);font-size:.73rem;font-weight:720; }}
    .margin-main {{ color:var(--blue-deep);font-size:1.16rem;font-weight:860;margin-top:.15rem; }}
    .margin-sub {{ color:var(--sub);font-size:.72rem;margin-top:.13rem; }}
    .trace-note {{ padding:.72rem .85rem;border-radius:12px;background:var(--surface);border:1px solid var(--line);
        color:var(--sub);font-size:.78rem;line-height:1.55;margin:.45rem 0 .75rem; }}
    .tiny {{ color:var(--muted);font-size:.76rem;line-height:1.6; }}

    div[data-testid="stMetric"] {{ background:var(--card);border:1px solid var(--line);padding:.8rem .9rem;border-radius:14px; }}
    div[data-testid="stMetric"] label, div[data-testid="stMetric"] label p, div[data-testid="stMetricValue"] {{ color:var(--ink) !important;opacity:1 !important; }}
    div[data-testid="stMetric"] label p {{ font-weight:650 !important; }}
    div[data-testid="stDataFrame"] {{ border:1px solid var(--line);border-radius:12px;overflow:hidden;background:var(--card); }}
    .ac-table-wrap {{ width:100%; overflow:auto; border:1px solid var(--line); border-radius:12px; background:var(--card); margin:.15rem 0 .65rem; }}
    .ac-table {{ width:100%; min-width:760px; border-collapse:separate; border-spacing:0; background:var(--card); color:var(--ink); font-size:.78rem; }}
    .ac-table th {{ position:sticky; top:0; z-index:1; padding:.62rem .68rem; text-align:left; background:var(--surface); color:var(--ink); font-weight:760; border-bottom:1px solid var(--line); white-space:nowrap; }}
    .ac-table td {{ padding:.58rem .68rem; color:var(--ink); border-bottom:1px solid var(--line); border-right:1px solid var(--line); white-space:nowrap; }}
    .ac-table th:not(:last-child) {{ border-right:1px solid var(--line); }}
    .ac-table td:last-child {{ border-right:0; }}
    .ac-table tbody tr:last-child td {{ border-bottom:0; }}
    .ac-table tbody tr:hover td {{ background:var(--surface); }}
    .ac-check {{ display:inline-grid; place-items:center; width:18px; height:18px; border-radius:5px; border:1px solid var(--line); font-size:.75rem; font-weight:900; }}
    .ac-check.yes {{ color:var(--teal); background:var(--teal-soft); }}
    .ac-check.no {{ color:var(--muted); background:var(--surface); }}
    div[data-baseweb="select"] > div, textarea, input {{ background:var(--input) !important;color:var(--ink) !important;border-color:var(--line) !important; }}
    div[data-baseweb="select"] span, div[data-baseweb="select"] svg {{ color:var(--ink) !important;fill:var(--ink) !important; }}
    textarea::placeholder, input::placeholder {{ color:var(--muted) !important;opacity:1 !important; }}
    div[data-testid="stExpander"], details[data-testid="stExpander"] {{ background:var(--card) !important;border-color:var(--line) !important;border-radius:12px !important; }}
    div[data-testid="stExpander"] summary, details[data-testid="stExpander"] summary {{ background:var(--card) !important;color:var(--ink) !important;border-radius:11px !important; }}
    div[data-testid="stExpander"] summary *, details[data-testid="stExpander"] summary *, div[data-testid="stExpander"] p, details[data-testid="stExpander"] p {{ color:var(--ink) !important;opacity:1 !important; }}
    div[data-testid="stExpander"] summary svg, details[data-testid="stExpander"] summary svg {{ fill:var(--ink) !important;color:var(--ink) !important; }}
    div[data-testid="stAlert"] {{ background:var(--surface) !important;border:1px solid var(--line) !important;color:var(--ink) !important; }}
    div[data-testid="stAlert"] * {{ color:var(--ink) !important;opacity:1 !important; }}
    input[type="radio"] {{ accent-color:var(--blue-deep) !important; }}
    div[data-testid="stButton"] > button[kind="primary"] {{ background:var(--blue-deep) !important;border-color:var(--blue-deep) !important;
        color:{'#07111B' if dark_mode else '#FFFFFF'} !important;border-radius:12px !important;min-height:48px;font-weight:800; }}
    div[data-testid="stButton"] > button[kind="primary"] p {{ color:{'#07111B' if dark_mode else '#FFFFFF'} !important;opacity:1 !important;font-weight:800 !important; }}
    div[data-testid="stButton"] > button[kind="primary"]:hover {{ filter:brightness(1.08); }}
    div[data-testid="stDownloadButton"] > button {{ background:var(--card) !important;border:1px solid var(--line) !important;color:var(--ink) !important;border-radius:10px !important; }}
    div[data-testid="stDownloadButton"] > button p {{ color:var(--ink) !important;opacity:1 !important;font-weight:700 !important; }}
    div[data-testid="stToggle"] label, div[data-testid="stToggle"] label p {{ color:var(--ink) !important;opacity:1 !important; }}

    @media (max-width:900px) {{
        .proof-grid,.margin-grid,.anim-grid,.diag-grid {{ grid-template-columns:1fr; }}
        .ac-tag {{ display:none; }}
        .anim-card {{ min-height:285px; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Header + in-app theme switch
header_left, header_right = st.columns([8.6, 1.4], vertical_alignment="center")
with header_left:
    st.markdown(
        """
        <div class="ac-topbar">
          <div class="ac-brand">
            <div class="ac-mark">S</div>
            <div class="ac-name">SynapseFlow</div>
            <div class="ac-tag">Netlist-aware adaptive analog design assistant</div>
          </div>
          <div class="ac-badge">Specification parser → Automatic surrogate → ngspice</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with header_right:
    st.toggle("Dark mode", key="dark_mode")

st.markdown(
    textwrap.dedent("""
    <div class="hero">
      <div class="hero-title">From specification to SPICE-verified sizing</div>
      <div class="hero-sub">SynapseFlow turns a requested analog specification into SPICE-verified sizing. It builds a local small-signal surrogate from netlist connectivity and ngspice operating-point physics, searches only inside a validated trust region, and updates the local model from verified simulator results. An Ollama endpoint can be used for specification extraction; when none is configured, explicit value/unit tokens are parsed deterministically.</div>
      <div class="flow"><span>Specification</span><b>→</b><span>Netlist + OP physics</span><b>→</b><span>Automatic surrogate</span><b>→</b><span>Trust-region search</span><b>→</b><span>SPICE verification</span><b>→</b><span>Jacobian update</span></div>
    </div>
    <div class="proof-grid">
      <div class="proof-card"><div class="proof-value">Reference engine · 29.40× fewer</div><div class="proof-label">Physics-guided SPICE evaluations vs. random search on the real-tail differential-pair held-out benchmark</div></div>
      <div class="proof-card"><div class="proof-value">Reference engine · 88/88</div><div class="proof-label">Nominal held-out specifications solved with SPICE verification</div></div>
      <div class="proof-card"><div class="proof-value">Automatic surrogate · 12/12 · 61% fewer OP</div><div class="proof-label">Balanced benchmark: Broyden solved all 12 specs with 61.0% fewer average OP simulations than full recalibration</div></div>
      <div class="proof-card"><div class="proof-value">27/27</div><div class="proof-label">Illustrative PVT-like conditions passed after physics-guided robust redesign</div></div>
    </div>
    """),
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="margin-top:1.25rem;margin-bottom:.55rem;">
      <div style="font-size:1.27rem;font-weight:800;letter-spacing:-.02em;">Where SynapseFlow fits in the IC design flow</div>
      <div style="margin-top:.2rem;font-size:.92rem;color:var(--sub);line-height:1.55;">Digital implementation and analog/mixed-signal sizing use different toolchains. The diagrams provide design-flow context; SynapseFlow currently targets the analog/mixed-signal sizing and verification loop.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

_anim_bg = THEME["card"]
_anim_surface = THEME["surface"]
_anim_line = THEME["line"]
_anim_ink = THEME["ink"]
_anim_sub = THEME["sub"]
_anim_blue = THEME["blue"]
_anim_teal = THEME["teal"]
_anim_soft = THEME["soft"]

animation_html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
* {{ box-sizing:border-box; }}
html,body {{ margin:0; padding:0; background:transparent; font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:{_anim_ink}; }}
.grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
.card {{ position:relative; overflow:hidden; min-height:330px; padding:18px 18px 14px; border:1px solid {_anim_line}; border-radius:20px; background:{_anim_bg}; }}
.card:after {{ content:""; position:absolute; width:180px; height:180px; border-radius:50%; right:-80px; bottom:-110px; background:{_anim_soft}; opacity:.65; pointer-events:none; }}
.head {{ display:flex; align-items:center; justify-content:space-between; gap:10px; margin-bottom:7px; }}
.title {{ font-size:15px; font-weight:800; }}
.pill {{ font-size:11px; font-weight:700; padding:5px 8px; border:1px solid {_anim_line}; border-radius:999px; color:{_anim_sub}; background:{_anim_surface}; }}
.caption {{ margin-top:3px; color:{_anim_sub}; font-size:12px; line-height:1.48; }}
svg {{ width:100%; height:245px; display:block; overflow:visible; }}
.node {{ fill:{_anim_surface}; stroke:{_anim_line}; stroke-width:1.5; }}
.nodeText {{ fill:{_anim_ink}; font-size:14px; font-weight:800; text-anchor:middle; dominant-baseline:middle; }}
.sub {{ fill:{_anim_sub}; font-size:10.5px; text-anchor:middle; }}
.wire {{ fill:none; stroke:{_anim_sub}; stroke-width:2; stroke-linecap:round; stroke-linejoin:round; opacity:.9; }}
.live {{ fill:none; stroke:{_anim_teal}; stroke-width:3; stroke-linecap:round; filter:drop-shadow(0 0 3px {_anim_teal}); animation:glow 1.7s ease-in-out infinite alternate; }}
.flow {{ fill:none; stroke:{_anim_blue}; stroke-width:2.5; stroke-linecap:round; }}
.dot {{ fill:{_anim_blue}; filter:drop-shadow(0 0 5px {_anim_blue}); }}
.cell {{ fill:{_anim_blue}; opacity:.32; animation:cellPulse 2.4s ease-in-out infinite; }}
.wave {{ fill:none; stroke:{_anim_teal}; stroke-width:2.8; stroke-linecap:round; stroke-dasharray:7 5; animation:dash 1.7s linear infinite; }}
.p1 {{ animation:pulse 4.0s ease-in-out infinite; }} .p2 {{ animation:pulse 4.0s .7s ease-in-out infinite; }} .p3 {{ animation:pulse 4.0s 1.4s ease-in-out infinite; }} .p4 {{ animation:pulse 4.0s 2.1s ease-in-out infinite; }}
.size80,.size100,.size140 {{ fill:{_anim_ink}; font-size:13px; font-weight:800; text-anchor:middle; opacity:0; }}
.size80 {{ animation:size80 6s infinite; }} .size100 {{ animation:size100 6s infinite; }} .size140 {{ animation:size140 6s infinite; }}
@keyframes pulse {{ 0%,15%,100%{{opacity:.62}} 28%,45%{{opacity:1; filter:drop-shadow(0 0 8px {_anim_blue});}} }}
@keyframes glow {{ from{{opacity:.55}} to{{opacity:1}} }}
@keyframes dash {{ to{{stroke-dashoffset:-24}} }}
@keyframes cellPulse {{ 0%,100%{{opacity:.22}} 50%{{opacity:.58}} }}
@keyframes size80 {{ 0%,29%{{opacity:1}} 34%,100%{{opacity:0}} }}
@keyframes size100 {{ 0%,30%{{opacity:0}} 35%,62%{{opacity:1}} 67%,100%{{opacity:0}} }}
@keyframes size140 {{ 0%,63%{{opacity:0}} 68%,96%{{opacity:1}} 100%{{opacity:0}} }}
@media(max-width:800px) {{ .grid{{grid-template-columns:1fr;}} .card{{min-height:310px;}} }}
</style>
</head>
<body>
<div class="grid">
  <div class="card">
    <div class="head"><div class="title">Digital circuit design · SoC / RTL</div><div class="pill">context flow</div></div>
    <svg viewBox="0 0 620 245" role="img" aria-label="Animated digital SoC RTL design flow">
      <defs><marker id="arrD" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="{_anim_blue}"/></marker></defs>
      <path class="flow" marker-end="url(#arrD)" d="M112 67 H165"/><path class="flow" marker-end="url(#arrD)" d="M265 67 H318"/><path class="flow" marker-end="url(#arrD)" d="M418 67 H470"/>
      <g class="p1"><rect class="node" x="15" y="34" rx="12" width="97" height="66"/><text class="nodeText" x="63" y="59">SPEC</text><text class="sub" x="63" y="82">architecture</text></g>
      <g class="p2"><rect class="node" x="165" y="34" rx="12" width="100" height="66"/><text class="nodeText" x="215" y="59">RTL</text><text class="sub" x="215" y="82">Verilog / SV</text></g>
      <g class="p3"><rect class="node" x="318" y="34" rx="12" width="100" height="66"/><text class="nodeText" x="368" y="59">SYNTH</text><text class="sub" x="368" y="82">logic gates</text></g>
      <g class="p4"><rect class="node" x="470" y="34" rx="12" width="134" height="66"/><text class="nodeText" x="537" y="59">PLACE & ROUTE</text><text class="sub" x="537" y="82">physical layout</text></g>
      <path class="wire" d="M63 100 V149 H537 V100"/><circle class="dot" r="5"><animateMotion dur="4.3s" repeatCount="indefinite" path="M63 100 V149 H537 V100"/></circle>
      <text class="sub" x="63" y="169">simulation</text><text class="sub" x="365" y="169">timing closure</text><text class="sub" x="537" y="169">GDS</text>
      <g transform="translate(453 185)"><rect class="cell" x="0" y="0" width="30" height="18" rx="3"/><rect class="cell" x="35" y="0" width="43" height="18" rx="3"/><rect class="cell" x="83" y="0" width="25" height="18" rx="3"/><rect class="cell" x="113" y="0" width="35" height="18" rx="3"/><rect class="cell" x="5" y="23" width="39" height="18" rx="3"/><rect class="cell" x="49" y="23" width="28" height="18" rx="3"/><rect class="cell" x="82" y="23" width="46" height="18" rx="3"/><rect class="cell" x="133" y="23" width="18" height="18" rx="3"/></g>
      <text class="sub" x="295" y="228">constraints → verification → implementation</text>
    </svg>
    <div class="caption">RTL logic is synthesized into gates, physically placed/routed, and iterated until timing and implementation constraints close.</div>
  </div>

  <div class="card">
    <div class="head"><div class="title">Analog / mixed-signal circuit design</div><div class="pill">SynapseFlow focus</div></div>
    <svg viewBox="0 0 620 245" role="img" aria-label="Animated analog mixed signal sizing and SPICE flow">
      <line class="wire" x1="310" y1="11" x2="310" y2="31"/><text class="sub" x="310" y="9">VDD</text>
      <path class="wire" d="M225 31 v14 l-10 7 l20 10 l-20 10 l20 10 l-10 7 v15"/><path class="wire" d="M395 31 v14 l-10 7 l20 10 l-20 10 l20 10 l-10 7 v15"/><line class="wire" x1="225" y1="31" x2="395" y2="31"/>
      <line class="live" x1="225" y1="104" x2="225" y2="123"/><line class="live" x1="395" y1="104" x2="395" y2="123"/>
      <rect class="node" x="202" y="123" width="46" height="36" rx="8"/><rect class="node" x="372" y="123" width="46" height="36" rx="8"/><text class="nodeText" x="225" y="141">M1</text><text class="nodeText" x="395" y="141">M2</text>
      <line class="wire" x1="132" y1="141" x2="202" y2="141"/><line class="wire" x1="418" y1="141" x2="488" y2="141"/><text class="sub" x="126" y="145">VIN+</text><text class="sub" x="494" y="145">VIN−</text>
      <line class="live" x1="225" y1="159" x2="310" y2="177"/><line class="live" x1="395" y1="159" x2="310" y2="177"/><rect class="node" x="287" y="176" width="46" height="34" rx="8"/><text class="nodeText" x="310" y="193">TAIL</text>
      <text class="sub" x="203" y="112">VOUT+</text><text class="sub" x="416" y="112">VOUT−</text>
      <g transform="translate(24 33)"><rect class="node" x="0" y="0" width="132" height="72" rx="12"/><text class="sub" x="66" y="20">netlist → OP</text><text class="nodeText" x="66" y="43">gm / gds / C</text><text class="sub" x="66" y="63">automatic MNA surrogate</text></g>
      <g transform="translate(458 29)"><rect class="node" x="0" y="0" width="138" height="78" rx="12"/><text class="sub" x="69" y="20">SPICE verification</text><path class="wave" d="M10 51 C26 23,40 76,56 50 S88 23,103 51 S123 76,130 49"/><text class="sub" x="69" y="69">gain · BW · power</text></g>
      <path class="flow" d="M156 69 H190"/><path class="flow" d="M430 69 H458"/>
      <text class="sub" x="310" y="232">build surrogate → rank candidates → SPICE → Broyden update → PASS</text>
    </svg>
    <div class="caption">SynapseFlow builds a local surrogate from netlist connectivity and ngspice operating-point physics, ranks nearby candidates cheaply, and uses SPICE as the final verifier and re-anchoring source.</div>
  </div>
</div>
</body>
</html>
"""

components.html(animation_html, height=395, scrolling=False)

# -----------------------------
# Main controls
# -----------------------------
ENGINE_AUTO = "Automatic surrogate · trust-region + SPICE"
ENGINE_LEGACY = "Topology-specific physics · reference"

c_top, c_mode, c_engine = st.columns([1, 1, 1.15])
with c_top:
    topology = st.selectbox(
        "Circuit topology",
        ["Differential Pair · Real NMOS Tail", "Common-Source NMOS"],
    )
with c_mode:
    if topology.startswith("Differential"):
        design_mode = st.selectbox(
            "Design mode",
            ["Nominal", "VCM Robust", "PVT-like Robust"],
        )
    else:
        design_mode = "Nominal"
        st.selectbox("Design mode", ["Nominal"], disabled=True)
with c_engine:
    if topology.startswith("Differential") and design_mode == "Nominal":
        engine = st.selectbox("Optimization engine", [ENGINE_AUTO, ENGINE_LEGACY])
    else:
        engine = ENGINE_LEGACY
        st.selectbox("Optimization engine", [ENGINE_LEGACY], disabled=True)

if topology.startswith("Differential"):
    if design_mode == "Nominal" and engine == ENGINE_AUTO:
        mode_copy = (
            "Automatic engine: parse the netlist, import device operating-point physics from ngspice, "
            "assemble a local small-signal surrogate, search only inside the validated local radius, "
            "and re-anchor with SPICE when required."
        )
    else:
        mode_copy = {
            "Nominal": "Reference engine: find one SPICE-verified sizing point at VCM = 1.00 V using the hand-crafted topology-specific physics surrogate.",
            "VCM Robust": "Verify and redesign across VCM = 0.95 / 1.00 / 1.05 V.",
            "PVT-like Robust": "Research mode: redesign across 3 process-like × 3 temperature × 3 VCM conditions.",
        }[design_mode]
else:
    mode_copy = "Find a SPICE-verified common-source sizing point with the reference topology-specific engine."
st.markdown(f'<div class="mode-note">{mode_copy}</div>', unsafe_allow_html=True)
if topology.startswith("Differential"):
    if design_mode == "Nominal" and engine == ENGINE_AUTO:
        scope_badge = "Automatic surrogate engine"
        scope_note = "Netlist-derived local model · trust-region search · SPICE final verification"
    elif design_mode == "VCM Robust":
        scope_badge = "Validated robust reference engine"
        scope_note = "Topology-specific physics surrogate · VCM robustness across 0.95 / 1.00 / 1.05 V"
    elif design_mode == "PVT-like Robust":
        scope_badge = "Illustrative PVT-like reference engine"
        scope_note = "Level-1 research study · not foundry-PDK signoff"
    else:
        scope_badge = "Topology-specific reference engine"
        scope_note = "Reference nominal flow retained for comparison with the automatic-surrogate engine"
    st.markdown(
        f'<div class="scope-strip"><span class="scope-badge">{scope_badge}</span><span class="scope-note">{scope_note}</span></div>',
        unsafe_allow_html=True,
    )

# Automatic-surrogate engine defaults. The current optimizer parameterization
# expects .param RD / WIN / WTAIL / VBIAS and differential outp/outn nodes.
auto_strategy = "broyden"
auto_netlist_text = DEFAULT_DIFF_PAIR_NETLIST
auto_output_p, auto_output_n, auto_input_amp = "outp", "outn", 1.0
auto_rd_step, auto_win_step_um, auto_wtail_step_um, auto_vbias_step = 200.0, 2.0, 5.0, 0.01
auto_safe_radius, auto_sat_margin_mv, auto_max_moves, auto_active_k = 3.0, 50.0, 12, 2

if topology.startswith("Differential") and design_mode == "Nominal" and engine == ENGINE_AUTO:
    st.markdown('<div class="section-title">Automatic surrogate engine</div>', unsafe_allow_html=True)
    e1, e2 = st.columns([1, 1])
    with e1:
        strategy_label = st.selectbox(
            "Surrogate update method",
            [STRATEGY_LABELS["broyden"], STRATEGY_LABELS["active"], STRATEGY_LABELS["full"]],
            help="Broyden reuses the previous local Jacobian after each verified operating point. Active-variable recalibrates selected columns. Full finite-difference is the high-cost reference.",
        )
        auto_strategy = {v: k for k, v in STRATEGY_LABELS.items()}[strategy_label]
    with e2:
        st.markdown(
            '<div class="method-card"><div class="method-kicker">Validated search method</div>'
            '<div class="method-name">Trust-region local candidate ranking</div>'
            '<div class="method-note">Candidates are ranked only inside the empirically validated local radius; selected designs are verified by ngspice.</div></div>',
            unsafe_allow_html=True,
        )

input_mode = st.radio(
    "Specification input",
    ["Natural language", "Manual"],
    horizontal=True,
)

DEFAULT_TEXT = {
    "Nominal": "차동 증폭기를 설계해줘. 전압 이득은 최소 18 dB, 대역폭은 4.5 MHz 이상, 소비전력은 0.4 mW 이하로 해줘.",
    "VCM Robust": "차동 증폭기를 설계해줘. 전압 이득은 최소 22 dB, 대역폭은 3.5 MHz 이상, 소비전력은 0.5 mW 이하로 해줘.",
    "PVT-like Robust": "차동 증폭기를 설계해줘. 전압 이득은 최소 22 dB, 대역폭은 3.5 MHz 이상, 소비전력은 0.5 mW 이하로 해줘.",
}

if input_mode == "Natural language":
    default_request = (
        "공통 소스 증폭기를 설계해줘. 전압 이득은 최소 20 dB, 대역폭은 3 MHz 이상, 소비전력은 0.5 mW 이하로 해줘."
        if topology.startswith("Common")
        else DEFAULT_TEXT[design_mode]
    )
    request = st.text_area(
        "Describe the target",
        value=default_request,
        height=105,
    )
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        default_gain = 22.0 if design_mode != "Nominal" else 18.0
        manual_gain = st.number_input("Minimum gain (dB)", 0.0, 60.0, default_gain, 0.5)
    with c2:
        default_bw = 3.5 if design_mode != "Nominal" else 4.5
        manual_bw = st.number_input("Minimum bandwidth (MHz)", 0.1, 1000.0, default_bw, 0.5)
    with c3:
        default_power = 0.5 if design_mode != "Nominal" else 0.4
        manual_power = st.number_input("Maximum power (mW)", 0.001, 100.0, default_power, 0.05)

with st.expander("Advanced engineering & runtime settings", expanded=False):
    if topology.startswith("Differential") and design_mode == "Nominal" and engine == ENGINE_AUTO:
        st.markdown("**Netlist → automatic local surrogate**")
        auto_netlist_text = st.text_area(
            "SPICE netlist",
            value=DEFAULT_DIFF_PAIR_NETLIST,
            height=260,
            help="The builder parses R/C/V/MOS connectivity and imports gm/gds/gmb/capacitance/operating-point quantities from ngspice. The current optimizer expects RD/WIN/WTAIL/VBIAS parameters.",
        )
        n1, n2, n3 = st.columns(3)
        auto_output_p = n1.text_input("Output + node", value="outp")
        auto_output_n = n2.text_input("Output − node", value="outn")
        auto_input_amp = n3.number_input("Differential AC input amplitude (V)", min_value=1e-9, value=1.0, format="%.6f")

        st.markdown("**Trust gate + calibration**")
        t1, t2, t3, t4 = st.columns(4)
        auto_rd_step = t1.number_input("RD perturbation (Ω)", min_value=1.0, value=200.0)
        auto_win_step_um = t2.number_input("WIN perturbation (µm)", min_value=0.1, value=2.0)
        auto_wtail_step_um = t3.number_input("WTAIL perturbation (µm)", min_value=0.1, value=5.0)
        auto_vbias_step = t4.number_input("VBIAS perturbation (V)", min_value=0.001, value=0.01, format="%.3f")
        t5, t6, t7, t8 = st.columns(4)
        auto_safe_radius = t5.number_input("Validated safe radius", min_value=0.5, value=3.0, step=0.25)
        auto_sat_margin_mv = t6.number_input("Required saturation margin (mV)", min_value=0.0, value=50.0, step=5.0)
        auto_max_moves = int(t7.number_input("Maximum verified design moves", min_value=1, max_value=30, value=12, step=1))
        auto_active_k = int(t8.number_input("Active variables (Active method)", min_value=1, max_value=4, value=2, step=1))

    st.markdown("**Local runtime**")
    a1, a2, a3 = st.columns(3)
    with a1:
        ngspice_path = st.text_input("ngspice executable", value=DEFAULT_NGSPICE, placeholder="Blank = auto-detect from PATH", help="Cloud deployment installs ngspice via packages.txt and auto-detects it from PATH.")
    with a2:
        ollama_url = st.text_input("Ollama URL (optional)", value=DEFAULT_OLLAMA_URL, placeholder="Blank = deterministic parser fallback", help="For public/cloud deployment, leave this blank unless you have a remotely reachable Ollama endpoint.")
    with a3:
        model = st.text_input("Parser model", value=DEFAULT_OLLAMA_MODEL)

run = st.button("Run SynapseFlow · Search + SPICE Verify", type="primary", use_container_width=True)


def parse_spec():
    if input_mode == "Manual":
        return None, {
            "min_gain_db": float(manual_gain),
            "min_bandwidth_mhz": float(manual_bw),
            "max_power_mw": float(manual_power),
        }
    raw, common_spec = parse_circuit_spec_v2(request, ollama_url=ollama_url, model=model)
    return raw, {
        "min_gain_db": float(common_spec["min_gain_db"]),
        "min_bandwidth_mhz": float(common_spec["min_bandwidth_hz"]) / 1e6,
        "max_power_mw": float(common_spec["max_power_mw"]),
    }


def _table_cell(value):
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except Exception:
        pass
    if isinstance(value, bool) or str(value) in {"True", "False"}:
        yes = bool(value) if isinstance(value, bool) else str(value) == "True"
        cls = "yes" if yes else "no"
        mark = "✓" if yes else ""
        return f'<span class="ac-check {cls}">{mark}</span>'
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return html.escape(str(value))


DISPLAY_COLUMN_NAMES = {
    "Active_vars": "Active variables",
    "RD_kOhm": "RD (kΩ)",
    "RD_ohm": "RD (Ω)",
    "WIN_um": "Input W (µm)",
    "WTAIL_um": "Tail W (µm)",
    "VBIAS_V": "VBIAS (V)",
    "Pred_Gain_dB": "Pred. Gain (dB)",
    "Pred_BW_MHz": "Pred. BW (MHz)",
    "Pred_Power_mW": "Pred. Power (mW)",
    "Pred_MinSatMargin_mV": "Pred. min sat. margin (mV)",
    "SPICE_Gain_dB": "SPICE Gain (dB)",
    "SPICE_BW_MHz": "SPICE BW (MHz)",
    "SPICE_Power_mW": "SPICE Power (mW)",
    "SPICE_MinSatMargin_mV": "SPICE min sat. margin (mV)",
    "Pred_PASS": "Predicted PASS",
    "Physical_Valid": "Physical valid",
}


def display_table(df, max_height=420):
    if df is None or df.empty:
        return
    shown = df.rename(columns={c: DISPLAY_COLUMN_NAMES.get(c, c) for c in df.columns})
    render_table(shown, max_height=max_height)


def render_table(df, max_height=420):
    if df is None or df.empty:
        return
    cols = list(df.columns)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
    body_rows = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{_table_cell(row[c])}</td>" for c in cols)
        body_rows.append(f"<tr>{cells}</tr>")
    table_html = (
        f'<div class="ac-table-wrap" style="max-height:{int(max_height)}px">'
        f'<table class="ac-table"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody></table></div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)


def show_spec(spec):
    st.markdown('<div class="section-title">Target specification</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Gain ≥", f"{spec['min_gain_db']:.2f} dB")
    m2.metric("Bandwidth ≥", f"{spec['min_bandwidth_mhz']:.3f} MHz")
    m3.metric("Power ≤", f"{spec['max_power_mw']:.3f} mW")


def _fmt_change(prev, curr):
    changes = []
    for key, label, fmt in [
        ("RD_ohm", "RD", lambda x: f"{x/1000:.1f} kΩ"),
        ("WIN_um", "WIN", lambda x: f"{x:.0f} µm"),
        ("WTAIL_um", "WTAIL", lambda x: f"{x:.0f} µm"),
        ("VBIAS_V", "VBIAS", lambda x: f"{x:.2f} V"),
    ]:
        if key in curr and key in prev and curr[key] != prev[key]:
            changes.append(f"{label} {fmt(prev[key])} → {fmt(curr[key])}")
    return ", ".join(changes) if changes else "Initial design"


def make_trace_display(history):
    if history.empty:
        return history
    h = history.reset_index(drop=True).copy()
    rows = []
    for i, row in h.iterrows():
        prev = h.iloc[i - 1] if i > 0 else row
        change = _fmt_change(prev, row) if i > 0 else "Initial design"
        if "PASS" in h.columns:
            passed = bool(row.get("PASS", False))
        elif "All_27_PASS" in h.columns:
            passed = bool(row.get("All_27_PASS", False))
        else:
            passed = bool(row.get("All_Corners_PASS", False))
        gain = row.get("Gain_dB", row.get("Worst_Gain_dB", float("nan")))
        bw = row.get(
            "Bandwidth_MHz",
            row.get("BW_MHz", row.get("Worst_BW_MHz", float("nan"))),
        )
        power = row.get("Power_mW", row.get("Worst_Power_mW", float("nan")))

        # Robust-search history may use a text marker such as "Expanded"
        # after the frozen sizing grid is exhausted. Keep that label instead
        # of forcing every iteration marker through int().
        raw_step = row.get("Iteration", i + 1)
        try:
            step = int(raw_step)
        except (TypeError, ValueError):
            step = str(raw_step)

        rows.append({
            "Step": step,
            "Sizing change": change,
            "Gain (dB)": round(float(gain), 3),
            "Bandwidth (MHz)": round(float(bw), 3),
            "Power (mW)": round(float(power), 3),
            "Status": "PASS" if passed else "Searching",
        })
    return pd.DataFrame(rows)


def show_diff_final(result, mode, spec):
    final = result["final_design"]
    corners = result.get("final_corner_results")
    if corners is not None and not isinstance(corners, pd.DataFrame):
        corners = pd.DataFrame(corners)

    if mode == "Nominal":
        a = result["final_result"]
        gain = float(a["Gain_dB"])
        bw = float(a["Bandwidth_MHz"])
        power = float(a["Power_mW"])
        coverage_text = f"{result['spice_evaluations']} SPICE evaluations"
    else:
        gain = float(corners["Gain_dB"].min())
        bw = float(corners["Bandwidth_MHz"].min())
        power = float(corners["Power_mW"].max())
        coverage_text = (
            f"{int(corners['PASS'].sum())}/{len(corners)} VCM corners"
            if mode == "VCM Robust"
            else f"{int(corners['PASS'].sum())}/27 conditions"
        )

    if result["success"]:
        result_html = (
            '<div class="result-head">'
            '<div><div class="result-kicker">Verified result</div>'
            '<div class="result-title">Requested constraints satisfied</div></div>'
            f'<div class="result-chip">SPICE VERIFIED · {coverage_text}</div>'
            '</div>'
        )
        st.markdown(result_html, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="status-fail">No verified solution found within the active design space / iteration budget.</div>',
            unsafe_allow_html=True,
        )

    gain_margin = gain - spec["min_gain_db"]
    bw_margin = bw - spec["min_bandwidth_mhz"]
    power_headroom = spec["max_power_mw"] - power
    margin_prefix = "" if mode == "Nominal" else "Worst-case "
    margin_html = (
        '<div class="margin-grid">'
        f'<div class="margin-card"><div class="margin-label">{margin_prefix}Gain margin</div><div class="margin-main">+{gain_margin:.3f} dB</div><div class="margin-sub">{gain:.3f} dB achieved · target ≥ {spec["min_gain_db"]:.3f}</div></div>'
        f'<div class="margin-card"><div class="margin-label">{margin_prefix}Bandwidth margin</div><div class="margin-main">+{bw_margin:.3f} MHz</div><div class="margin-sub">{bw:.3f} MHz achieved · target ≥ {spec["min_bandwidth_mhz"]:.3f}</div></div>'
        f'<div class="margin-card"><div class="margin-label">{margin_prefix}Power headroom</div><div class="margin-main">{power_headroom:.3f} mW</div><div class="margin-sub">{power:.3f} mW achieved · limit ≤ {spec["max_power_mw"]:.3f}</div></div>'
        '</div>'
    )
    st.markdown(margin_html, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Final sizing</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RD", f"{final['RD_ohm']/1000:.1f} kΩ")
    c2.metric("Input W", f"{final['WIN_um']:.0f} µm")
    c3.metric("Tail W", f"{final['WTAIL_um']:.0f} µm")
    c4.metric("VBIAS", f"{final['VBIAS_V']:.2f} V")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Gain", f"{gain:.3f} dB")
    r2.metric("Bandwidth", f"{bw:.3f} MHz")
    r3.metric("Power", f"{power:.3f} mW")
    if mode == "Nominal":
        r4.metric("SPICE evaluations", str(result["spice_evaluations"]))
    else:
        r4.metric("Verified coverage", coverage_text)

    if mode == "VCM Robust" and result.get("expanded"):
        st.info(
            "The frozen nominal sizing grid did not provide a verified robust solution. "
            "SynapseFlow relaxed the sizing bounds and verified an expanded-space candidate with SPICE."
        )

    if mode == "PVT-like Robust":
        hist = pd.DataFrame(result.get("history", []))
        if not hist.empty and "PASS_Count" in hist.columns:
            start_pass = int(hist.iloc[0]["PASS_Count"])
            end_pass = int(hist.iloc[-1]["PASS_Count"])
            st.markdown(
                f'<div class="trace-note"><b>Robustness improvement:</b> {start_pass}/27 → {end_pass}/27 conditions '
                f'· {len(hist)} design iterations · {int(hist.iloc[-1]["Total_SPICE_Evals"])} SPICE condition evaluations</div>',
                unsafe_allow_html=True,
            )
        st.warning(
            "Research scope: this is an illustrative PVT-like study using a Level-1 MOS model, "
            "illustrative KP/VTO process variation, and ngspice temperature behavior — not foundry PDK signoff."
        )



def show_automatic_final(result, spec, netlist_text, update_method_label, safe_radius):
    """Render the V5 netlist-derived adaptive optimizer result."""
    m = result.final_metrics
    p = result.final_params
    if result.success:
        st.markdown(
            '<div class="result-head"><div><div class="result-kicker">Automatic surrogate engine</div>'
            '<div class="result-title">Requested constraints satisfied</div></div>'
            f'<div class="result-chip">SPICE VERIFIED · {result.ac_simulations} AC verifications</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-fail">No SPICE-verified solution found within the active trust-region / move budget.</div>',
            unsafe_allow_html=True,
        )

    gain_margin = float(m["gain_db"]) - spec["min_gain_db"]
    bw_margin = float(m["bw_mhz"]) - spec["min_bandwidth_mhz"]
    power_headroom = spec["max_power_mw"] - float(m["power_mw"])
    st.markdown(
        '<div class="margin-grid">'
        f'<div class="margin-card"><div class="margin-label">Gain margin</div><div class="margin-main">{gain_margin:+.3f} dB</div><div class="margin-sub">{m["gain_db"]:.3f} dB achieved · target ≥ {spec["min_gain_db"]:.3f}</div></div>'
        f'<div class="margin-card"><div class="margin-label">Bandwidth margin</div><div class="margin-main">{bw_margin:+.3f} MHz</div><div class="margin-sub">{m["bw_mhz"]:.3f} MHz achieved · target ≥ {spec["min_bandwidth_mhz"]:.3f}</div></div>'
        f'<div class="margin-card"><div class="margin-label">Power headroom</div><div class="margin-main">{power_headroom:.3f} mW</div><div class="margin-sub">{m["power_mw"]:.3f} mW achieved · limit ≤ {spec["max_power_mw"]:.3f}</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="screening-callout"><div><div class="screening-main"><b>{result.surrogate_evaluations:,}</b> surrogate candidates screened → <b>{result.ac_simulations}</b> ngspice AC verifications</div>'
        '<div class="screening-sub">Most sizing candidates are evaluated by the local surrogate. Only selected candidates are sent to ngspice; final acceptance is always simulator-verified.</div></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Final sizing</div>', unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("RD", f"{p['rd']/1e3:.3f} kΩ")
    c2.metric("Input W", f"{p['win']*1e6:.2f} µm")
    c3.metric("Tail W", f"{p['wtail']*1e6:.2f} µm")
    c4.metric("VBIAS", f"{p['vbias']:.3f} V")
    r1,r2,r3,r4=st.columns(4)
    r1.metric("SPICE gain", f"{m['gain_db']:.3f} dB")
    r2.metric("SPICE bandwidth", f"{m['bw_mhz']:.3f} MHz")
    r3.metric("SPICE power", f"{m['power_mw']:.3f} mW")
    r4.metric("Min saturation margin", f"{m['min_sat_margin_mv']:.1f} mV")

    st.markdown(
        '<div class="trace-note"><b>Why this design?</b> The candidate was selected from the physically valid local ranking inside the validated trust region. It was accepted only after ngspice verification satisfied the requested gain, bandwidth, power, and operating-region constraints.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Simulator / surrogate cost accounting</div>', unsafe_allow_html=True)
    q1,q2,q3,q4,q5=st.columns(5)
    q1.metric("Surrogate candidate evaluations", f"{result.surrogate_evaluations:,}")
    q2.metric("ngspice OP total", str(result.op_simulations))
    q3.metric("Perturbation OP", str(result.perturbation_op_simulations))
    q4.metric("ngspice AC verifications", str(result.ac_simulations))
    q5.metric("Local surrogate builds", str(result.anchor_rebuilds))
    st.caption(
        "OP and AC calls are reported separately because they are not equal-cost operations. Final PASS is always determined from ngspice, not from the surrogate prediction."
    )

    trace = pd.DataFrame(result.trace)
    try:
        parsed = parse_netlist(netlist_text)
        element_summary = f"{len(parsed.mosfets)} MOS · {len(parsed.resistors)} R · {len(parsed.capacitors)} C · {len(parsed.nodes)} nodes"
    except Exception:
        element_summary = "Netlist parsed for automatic surrogate"
    if not trace.empty:
        first = trace.iloc[0]
        pred0 = first.get("Pred_Gain_dB", first.get("SPICE_Gain_dB", float("nan")))
        spice0 = first.get("SPICE_Gain_dB", pred0)
        try:
            anchor_agreement = f"{abs(float(pred0)-float(spice0)):.4f} dB gain error"
        except Exception:
            anchor_agreement = "Initial anchor verified"
    else:
        anchor_agreement = "Initial anchor verified"
    with st.expander("Surrogate diagnostics · how the local model was built", expanded=False):
        st.markdown(
            '<div class="diag-grid">'
            f'<div class="diag-card"><div class="diag-label">Netlist parsed</div><div class="diag-value">{element_summary}</div></div>'
            '<div class="diag-card"><div class="diag-label">Device physics source</div><div class="diag-value">ngspice operating point</div></div>'
            f'<div class="diag-card"><div class="diag-label">Validated local radius</div><div class="diag-value">{safe_radius:.2f} calibration-step units</div></div>'
            f'<div class="diag-card"><div class="diag-label">Jacobian update</div><div class="diag-value">{html.escape(update_method_label)}</div></div>'
            '</div>'
            f'<div class="trace-note"><b>Initial anchor agreement:</b> {anchor_agreement} at the calibration point. The local model is used as a search guide, not as the final acceptance criterion.</div>',
            unsafe_allow_html=True,
        )

    if not trace.empty:
        st.markdown('<div class="section-title">Adaptive optimization trace</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="trace-note">The local surrogate ranks many candidates inside the validated radius. Only the selected design is verified by ngspice; failed verification becomes a new anchor and updates the local model.</div>',
            unsafe_allow_html=True,
        )
        display_table(trace, max_height=520)

    board = pd.DataFrame(result.leaderboard)
    if not board.empty:
        with st.expander("What the surrogate considered before each SPICE call", expanded=False):
            display_table(board, max_height=520)

    st.markdown(
        '<div class="trace-note"><b>Verified design loop:</b> netlist → ngspice operating point → automatic local small-signal surrogate → trust-region candidate ranking → physical-validity gate → SPICE verification → Jacobian update.</div>',
        unsafe_allow_html=True,
    )


if run:
    try:
        spinner_copy = (
            "Parsing specification → netlist-derived local surrogate → trust-region search → ngspice verification..."
            if (topology.startswith("Differential") and design_mode == "Nominal" and engine == ENGINE_AUTO)
            else "Parsing specification → physics-guided search → ngspice verification..."
        )
        with st.spinner(spinner_copy):
            raw_spec, spec = parse_spec()
            is_auto = topology.startswith("Differential") and design_mode == "Nominal" and engine == ENGINE_AUTO
            if is_auto:
                efficient_spec = {
                    "min_gain_db": spec["min_gain_db"],
                    "min_bw_mhz": spec["min_bandwidth_mhz"],
                    "max_power_mw": spec["max_power_mw"],
                }
                auto_steps = {
                    "rd": float(auto_rd_step),
                    "win": float(auto_win_step_um) * 1e-6,
                    "wtail": float(auto_wtail_step_um) * 1e-6,
                    "vbias": float(auto_vbias_step),
                }
                result = run_efficient_optimizer(
                    baseline_text=auto_netlist_text,
                    ngspice_path=ngspice_path,
                    spec=efficient_spec,
                    strategy=auto_strategy,
                    active_k=int(auto_active_k),
                    output_p=auto_output_p,
                    output_n=auto_output_n,
                    input_amp=float(auto_input_amp),
                    f_start=10.0,
                    f_stop=1e9,
                    calibration_steps=auto_steps,
                    safe_radius=float(auto_safe_radius),
                    sat_margin_v=float(auto_sat_margin_mv) / 1e3,
                    max_design_moves=int(auto_max_moves),
                )
            elif topology.startswith("Common"):
                cs_spec = {
                    "min_gain_db": spec["min_gain_db"],
                    "min_bandwidth_hz": spec["min_bandwidth_mhz"] * 1e6,
                    "max_power_mw": spec["max_power_mw"],
                }
                result = run_physics_guided_agent(
                    spec=cs_spec,
                    ngspice_path=ngspice_path,
                    start_rd=4700,
                    start_w=20,
                    start_vg=0.80,
                )
            elif design_mode == "Nominal":
                result = run_nominal_agent(spec, ngspice_path)
            elif design_mode == "VCM Robust":
                nominal = run_nominal_agent(spec, ngspice_path)
                result = run_vcm_robust_agent(
                    spec,
                    ngspice_path,
                    start_design=nominal["final_design"],
                    allow_expansion=True,
                )
                result["nominal_seed"] = nominal
            else:
                robust_seed = {
                    "RD_ohm": 9000,
                    "WIN_um": 80,
                    "WTAIL_um": 20,
                    "VBIAS_V": 0.75,
                }
                result = run_pvt_aware_agent(spec, ngspice_path, start_design=robust_seed)

        show_spec(spec)
        if raw_spec is not None:
            parser_source = raw_spec.get("_parser_source", "local_llm")
            if parser_source == "deterministic_fallback":
                st.warning(
                    "Local LLM parser returned an error, so SynapseFlow used its deterministic value/unit fallback for this run. "
                    "Numerical unit normalization and all circuit search/verification remain unchanged."
                )
            with st.expander("Language parser extraction"):
                st.json(raw_spec)

        if is_auto:
            show_automatic_final(result, spec, auto_netlist_text, STRATEGY_LABELS[auto_strategy], float(auto_safe_radius))
            payload = {
                "topology": topology,
                "mode": design_mode,
                "engine": engine,
                "surrogate_update_method": STRATEGY_LABELS[auto_strategy],
                "spec": spec,
                "success": result.success,
                "final_params": result.final_params,
                "final_metrics": result.final_metrics,
                "cost": {
                    "op_total": result.op_simulations,
                    "op_perturbation": result.perturbation_op_simulations,
                    "op_verified_anchors": result.anchor_op_simulations,
                    "ac_verifications": result.ac_simulations,
                    "surrogate_evaluations": result.surrogate_evaluations,
                    "surrogate_builds": result.anchor_rebuilds,
                    "wall_clock_s": result.wall_clock_s,
                },
                "trace": result.trace,
            }
            st.download_button(
                "Download run summary",
                data=json.dumps(payload, indent=2, default=str),
                file_name="synapseflow_automatic_run.json",
                mime="application/json",
            )
        else:
            history = pd.DataFrame(result.get("history", []))
            if topology.startswith("Common"):
                if result.get("success"):
                    final = result["result"]
                    p0, m0, op = final["parameters"], final["metrics"], final["operating_point"]
                    st.markdown('<div class="status-pass">SPICE VERIFIED — requested constraints satisfied.</div>', unsafe_allow_html=True)
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("RD", f"{p0['RD_ohm']/1000:.1f} kΩ")
                    c2.metric("W", f"{p0['W_um']:.0f} µm")
                    c3.metric("VG", f"{p0['VG_V']:.2f} V")
                    c4.metric("SPICE evaluations", str(result["spice_runs"]))
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Gain", f"{m0['Gain_dB']:.3f} dB")
                    r2.metric("Bandwidth", f"{m0['BW_MHz']:.3f} MHz")
                    r3.metric("Power", f"{m0['Power_mW']:.3f} mW")
                    r4.metric("Region", op["region"])
                else:
                    st.markdown('<div class="status-fail">No verified common-source solution found.</div>', unsafe_allow_html=True)
            else:
                show_diff_final(result, design_mode, spec)

            if not history.empty:
                trace_display = make_trace_display(history)
                st.markdown('<div class="section-title">How SynapseFlow converged</div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="trace-note">Each row is one design actually verified by ngspice. Physics evaluates nearby candidates cheaply; only the selected candidate is sent to SPICE.</div>',
                    unsafe_allow_html=True,
                )
                render_table(trace_display, max_height=500)
                with st.expander("Detailed engineering trace"):
                    render_table(history, max_height=520)

            if (not topology.startswith("Common")) and result.get("final_corner_results") is not None:
                corners = result.get("final_corner_results")
                if not isinstance(corners, pd.DataFrame):
                    corners = pd.DataFrame(corners)
                keep = [
                    c for c in [
                        "Process", "Temp_C", "VCM_V", "Gain_dB", "Bandwidth_MHz",
                        "Power_mW", "VTAIL_V", "Physical_Valid", "PASS"
                    ] if c in corners.columns
                ]
                with st.expander("Verified operating conditions", expanded=False):
                    render_table(corners[keep], max_height=520)

            payload = {
                "topology": topology,
                "mode": design_mode,
                "engine": engine,
                "spec": spec,
                "result": {
                    k: v for k, v in result.items()
                    if k not in {"final_corner_results", "history", "final_result", "nominal_seed"}
                },
            }
            st.download_button(
                "Download run summary",
                data=json.dumps(payload, indent=2, default=str),
                file_name="synapseflow_run_summary.json",
                mime="application/json",
            )

    except Exception as exc:
        st.error("SynapseFlow could not complete this run.")
        st.exception(exc)

st.divider()
st.markdown(
    """
    <div class="tiny">
    Research prototype. The V5 automatic engine builds a local small-signal surrogate from netlist connectivity and ngspice operating-point physics; the current optimizer parameterization is validated on the real-tail differential-pair Level-1 study. Robust PVT-like mode remains illustrative, not foundry signoff. ngspice remains the final verifier.
    </div>
    """,
    unsafe_allow_html=True,
)
