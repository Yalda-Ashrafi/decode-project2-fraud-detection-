# app.py — FraudSense AI · Fraud Detection Dashboard
import io
import os
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go          
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score,
                             precision_recall_curve, average_precision_score,
                             precision_score, recall_score, f1_score)

st.set_page_config(page_title="FraudSense · AI Detection Dashboard",
                   page_icon="🛡️", layout="wide",
                   initial_sidebar_state="collapsed")

# ---------------- AURORA PALETTE ----------------
VIOLET  = "#8B5CF6"
INDIGO  = "#6366F1"
BLUE    = "#3B82F6"
CYAN    = "#22D3EE"
TEAL    = "#2DD4BF"
MAGENTA = "#E879A6"
ROSE    = "#FB7185"
AMBER   = "#FBBF24"
LIME    = "#A3E635"
OK, WARN, BAD = TEAL, AMBER, ROSE
ACCENTS = [VIOLET, BLUE, CYAN, TEAL, MAGENTA, INDIGO]

# ---------------- THEME SYSTEM ----------------
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"        # "Light" or "Dark"

THEMES = {
    # ---- soft pastel aurora ----
    "Light": dict(
        bg="#EEF2FF",
        glow1="rgba(139,92,246,.30)", glow2="rgba(34,211,238,.28)",
        glow3="rgba(232,121,166,.24)", glow4="rgba(45,212,191,.26)",
        grid="rgba(49,46,129,.05)",
        card="rgba(255,255,255,.62)", card_border="rgba(79,70,229,.16)",
        card_hover_border="rgba(139,92,246,.55)",
        shadow="0 10px 32px rgba(49,46,129,.10)",
        shadow_hover="0 20px 48px rgba(49,46,129,.18), 0 0 26px rgba(34,211,238,.22)",
        text="#1E1B4B", body="#3F3D6B", muted="#6B7A99",
        input_bg="rgba(255,255,255,.60)", input_border="rgba(79,70,229,.20)",
        nav_bg="rgba(255,255,255,.55)",
        hero_text="linear-gradient(90deg,#4C1D95 0%,#4F46E5 40%,#0891B2 72%,#DB2777 100%)",
        plot_grid="rgba(49,46,129,.10)", plot_font="#3F3D6B",
        hover_bg="#FFFFFF", hover_border="#6366F1", hover_font="#1E1B4B",
        heat=[[0, "#F5F3FF"], [.45, "#A5B4FC"], [1, "#6366F1"]],
        neutral_bar="#C7D2FE",
    ),
    
    "Dark": dict(
        bg="#171034",
        glow1="rgba(139,92,246,.42)", glow2="rgba(34,211,238,.30)",
        glow3="rgba(232,121,166,.26)", glow4="rgba(45,212,191,.24)",
        grid="rgba(196,181,253,.055)",
        card="rgba(255,255,255,.055)", card_border="rgba(196,181,253,.20)",
        card_hover_border="rgba(34,211,238,.60)",
        shadow="0 10px 32px rgba(23,16,52,.55)",
        shadow_hover="0 20px 50px rgba(23,16,52,.72), 0 0 30px rgba(139,92,246,.28)",
        text="#F5F3FF", body="#DDD6FE", muted="#A5A0C9",
        input_bg="rgba(255,255,255,.07)", input_border="rgba(196,181,253,.26)",
        nav_bg="rgba(46,29,94,.48)",
        hero_text="linear-gradient(90deg,#F5F3FF 0%,#C4B5FD 32%,#67E8F9 66%,#F9A8D4 100%)",
        plot_grid="rgba(196,181,253,.15)", plot_font="#DDD6FE",
        hover_bg="#241A4D", hover_border="#22D3EE", hover_font="#F5F3FF",
        heat=[[0, "#241A4D"], [.45, "#7C3AED"], [1, "#22D3EE"]],
        neutral_bar="#4C3F7A",
    ),
}
T = THEMES[st.session_state.theme]

BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');

/* ---------- aurora background: four drifting colour fields ---------- */
.stApp {
    background:
        radial-gradient(1100px 620px at 10% -8%,  __GLOW1__, transparent 62%),
        radial-gradient(950px 560px at 92% 4%,    __GLOW2__, transparent 62%),
        radial-gradient(900px 520px at 78% 96%,   __GLOW3__, transparent 62%),
        radial-gradient(1000px 560px at 18% 88%,  __GLOW4__, transparent 62%),
        __BG__;
    background-attachment: fixed;
    color: __TEXT__;
    font-family: 'Inter', sans-serif;
}
.stApp::before {
    content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
    background-image:
        linear-gradient(__GRID__ 1px, transparent 1px),
        linear-gradient(90deg, __GRID__ 1px, transparent 1px);
    background-size: 52px 52px;
    mask-image: radial-gradient(ellipse at 50% 0%, #000 32%, transparent 76%);
}
.block-container { position:relative; z-index:1; padding-top:1.2rem; padding-bottom:1rem; max-width:1380px; }
#MainMenu, footer, header {visibility:hidden;}
section[data-testid="stSidebar"] { display:none; }

h1,h2,h3,h4 { color:__TEXT__ !important; font-family:'Manrope',sans-serif; letter-spacing:-.02em; }
p, li { color:__BODY__; }
label, .stRadio label, .stSlider label, div[data-testid="stWidgetLabel"] p {
    color:__BODY__ !important; font-weight:500;
}

@keyframes fadeUp   { from{opacity:0; transform:translateY(18px);} to{opacity:1; transform:none;} }
@keyframes glowPulse{ 0%,100%{opacity:.7; transform:translateY(0)} 50%{opacity:1; transform:translateY(-10px)} }
.fade-up { animation: fadeUp .55s cubic-bezier(.22,.9,.3,1) both; }
.delay-1{animation-delay:.08s} .delay-2{animation-delay:.16s} .delay-3{animation-delay:.24s}

/* ---------- cards ---------- */
.card {
    background: __CARD__; border: 1px solid __CARD_BORDER__;
    border-radius: 21px; padding: 22px 24px; box-shadow: __SHADOW__;
    transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
    animation: fadeUp .5s ease both;
}
.card:hover { transform: translateY(-6px); border-color: __CARD_HOVER__; box-shadow: __SHADOW_HOVER__; }
.glass { backdrop-filter: blur(18px) saturate(160%); -webkit-backdrop-filter: blur(18px) saturate(160%); }
.card h4 { margin:0 0 8px 0; font-size:1.02rem; }
.card p  { color:__MUTED__; font-size:.92rem; line-height:1.6; margin:0; }

/* ---------- hero ---------- */
.hero {
    border-radius: 28px; padding: 52px 44px; margin-bottom: 24px;
    background:
        radial-gradient(600px 300px at 8% 0%, rgba(139,92,246,.32), transparent 65%),
        radial-gradient(560px 300px at 96% 20%, rgba(34,211,238,.26), transparent 65%),
        radial-gradient(520px 300px at 60% 120%, rgba(232,121,166,.24), transparent 65%),
        __CARD__;
    border: 1px solid __CARD_BORDER__;
    backdrop-filter: blur(16px); animation: fadeUp .6s ease both;
    position: relative; overflow: hidden; box-shadow: __SHADOW__;
}
.hero::after{ content:""; position:absolute; width:440px; height:440px; right:-130px; top:-170px;
    background: radial-gradient(circle, rgba(45,212,191,.30), transparent 66%);
    animation: glowPulse 6s ease-in-out infinite; }
.hero h1 { font-family:'Manrope',sans-serif; font-size:3.1rem; font-weight:800; margin:0 0 12px 0;
    line-height:1.08; background: __HERO_TEXT__;
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hero p { font-size:1.05rem; color:__BODY__; max-width:720px; line-height:1.7; }
.pill { display:inline-block; padding:6px 15px; border-radius:999px; font-size:.75rem;
    font-weight:700; letter-spacing:.09em; text-transform:uppercase; margin-bottom:18px;
    background: linear-gradient(90deg, rgba(139,92,246,.22), rgba(34,211,238,.22));
    border:1px solid rgba(34,211,238,.45); color:__TEXT__; }
.tag { display:inline-block; margin:4px 8px 0 0; padding:5px 13px; border-radius:999px;
    font-size:.78rem; background:__INPUT_BG__; border:1px solid __CARD_BORDER__; color:__BODY__; }

/* ---------- buttons ---------- */
.stButton > button {
    border-radius: 13px; font-weight:600; padding:.55rem 1.1rem; width:100%;
    font-family:'Manrope',sans-serif;
    transition: all .25s cubic-bezier(.22,.9,.3,1);
    border:1px solid __INPUT_BORDER__; background: __INPUT_BG__; color:__BODY__;
}
.stButton > button:hover { transform: translateY(-2px); border-color:#22D3EE; color:__TEXT__;
    box-shadow:0 8px 24px rgba(34,211,238,.26); }
.stButton > button[kind="primary"] {
    background: linear-gradient(120deg,#E879A6 0%,#8B5CF6 32%,#3B82F6 64%,#22D3EE 100%);
    background-size:220% 220%; border:none; color:#fff;
    box-shadow:0 8px 26px rgba(139,92,246,.42); }
.stButton > button[kind="primary"]:hover { background-position:100% 0; transform:translateY(-3px);
    color:#fff; box-shadow:0 14px 38px rgba(34,211,238,.48); }

/* ---------- widgets ---------- */
div[data-baseweb="select"] > div, .stNumberInput input, div[data-baseweb="input"] > div {
    background: __INPUT_BG__ !important; border:1px solid __INPUT_BORDER__ !important;
    border-radius:12px !important; color:__TEXT__ !important;
}
div[data-baseweb="tag"] { background: linear-gradient(120deg,#8B5CF6,#22D3EE) !important;
    border-radius:9px !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] { box-shadow:0 0 16px rgba(34,211,238,.7); }
[data-testid="stDataFrame"], div[data-testid="stExpander"] {
    border:1px solid __CARD_BORDER__ !important; border-radius:18px !important; overflow:hidden;
    background: __CARD__ !important; backdrop-filter: blur(14px); }
[data-testid="stFileUploaderDropzone"] {
    background: __INPUT_BG__ !important; border:1.5px dashed __INPUT_BORDER__ !important;
    border-radius:18px !important; transition: all .25s ease; }
[data-testid="stFileUploaderDropzone"]:hover {
    border-color:#22D3EE !important; box-shadow:0 0 22px rgba(34,211,238,.20); }
.js-plotly-plot { animation: fadeUp .6s ease both; }
.stCode, pre { border-radius:16px !important; }

/* ---------- verdict banners ---------- */
.verdict { border-radius:21px; padding:22px 26px; font-weight:700; font-size:1.1rem;
           animation: fadeUp .45s ease both; border:1px solid; }
.v-fraud { background:linear-gradient(120deg, rgba(251,113,133,.18), rgba(232,121,166,.12));
           border-color:rgba(251,113,133,.50); color:#FDA4AF;
           box-shadow:0 0 34px rgba(251,113,133,.22); }
.v-safe  { background:linear-gradient(120deg, rgba(45,212,191,.16), rgba(34,211,238,.12));
           border-color:rgba(45,212,191,.48); color:#5EEAD4;
           box-shadow:0 0 34px rgba(45,212,191,.20); }

/* ---------- footer ---------- */
.site-footer { margin-top:46px; padding:26px 0 12px 0; text-align:center;
               border-top:1px solid __CARD_BORDER__; animation: fadeUp .6s ease both; }
.site-footer p { color:__MUTED__; font-size:.87rem; margin:0; }
</style>
"""

TOKENS = [("bg","__BG__"), ("glow1","__GLOW1__"), ("glow2","__GLOW2__"), ("glow3","__GLOW3__"),
          ("glow4","__GLOW4__"), ("grid","__GRID__"), ("card","__CARD__"),
          ("card_border","__CARD_BORDER__"), ("card_hover_border","__CARD_HOVER__"),
          ("shadow","__SHADOW__"), ("shadow_hover","__SHADOW_HOVER__"), ("text","__TEXT__"),
          ("body","__BODY__"), ("muted","__MUTED__"), ("input_bg","__INPUT_BG__"),
          ("input_border","__INPUT_BORDER__"), ("nav_bg","__NAV_BG__"), ("hero_text","__HERO_TEXT__")]

def build_css(template, t):
    css = template
    for key, token in TOKENS:
        if token in css:
            css = css.replace(token, t[key])
    return css

st.markdown(build_css(BASE_CSS, T), unsafe_allow_html=True)
# ---------------- PREMIUM HEADER + MOTION LAYER ----------------
EXTRA_CSS = """
<style>
@keyframes gradientShift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
@keyframes floatGlow     { 0%,100%{transform:translateY(0) scale(1); opacity:.55} 50%{transform:translateY(-10px) scale(1.08); opacity:.9} }
@keyframes sheen         { 0%{transform:translateX(-120%)} 60%,100%{transform:translateX(220%)} }
@keyframes pulseRing     { 0%{box-shadow:0 0 0 0 rgba(34,211,238,.5)} 70%{box-shadow:0 0 0 13px rgba(34,211,238,0)} 100%{box-shadow:0 0 0 0 rgba(34,211,238,0)} }
@keyframes revealUp      { from{opacity:0; transform:translateY(34px) scale(.985);} to{opacity:1; transform:none;} }

/* ======== THE HEADER ======== */
.st-key-navbar {
    position: sticky; top: 0; z-index: 999;
    background: __NAV_BG__;
    backdrop-filter: blur(24px) saturate(190%); -webkit-backdrop-filter: blur(24px) saturate(190%);
    border: 1px solid __CARD_BORDER__; border-radius: 22px;
    padding: 10px 22px 6px 22px !important; margin-bottom: 26px;
    box-shadow: __SHADOW__; overflow: hidden;
    animation: fadeUp .5s cubic-bezier(.22,.9,.3,1) both;
}
.st-key-navbar::before {
    content:""; position:absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg,#E879A6,#8B5CF6,#3B82F6,#22D3EE,#2DD4BF,#8B5CF6,#E879A6);
    background-size: 320% 100%; animation: gradientShift 9s ease infinite;
}
.st-key-navbar::after {
    content:""; position:absolute; width:280px; height:280px; right:6%; top:-200px;
    background: radial-gradient(circle, rgba(34,211,238,.34), transparent 68%);
    pointer-events:none; animation: floatGlow 7s ease-in-out infinite;
}
.st-key-navbar [data-testid="stHorizontalBlock"] { align-items:center; gap:.3rem; }

.brand-wrap { display:flex; align-items:center; gap:11px; padding-top:2px; }
.brand-mark {
    width:34px; height:34px; border-radius:11px; flex-shrink:0;
    display:flex; align-items:center; justify-content:center; font-size:1rem;
    background: linear-gradient(135deg,#E879A6,#8B5CF6 38%,#3B82F6 70%,#22D3EE);
    background-size:220% 220%;
    animation: gradientShift 6s ease infinite, pulseRing 3.4s ease-out infinite;
    box-shadow:0 6px 20px rgba(139,92,246,.45);
}
.brand-name {
    font-family:'Manrope',sans-serif; font-size:1.12rem; font-weight:800; letter-spacing:-.03em;
    background:linear-gradient(90deg,#E879A6,#8B5CF6 40%,#22D3EE); background-size:220% 100%;
    animation: gradientShift 7s ease infinite;
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.brand-tag { font-size:.63rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; color:__MUTED__; }

/* nav items as tabs */
.st-key-navbar .stButton > button {
    background: transparent !important; border: none !important; border-radius: 12px !important;
    padding: .5rem .55rem !important; font-family:'Manrope',sans-serif;
    font-weight:700 !important; font-size:.88rem !important; letter-spacing:-.01em;
    color: __MUTED__ !important; position: relative; overflow: hidden;
    transition: color .22s ease, background .22s ease, transform .22s ease; box-shadow: none !important;
}
.st-key-navbar .stButton > button:hover {
    color: __TEXT__ !important; background: __INPUT_BG__ !important; transform: translateY(-2px);
}
.st-key-navbar .stButton > button::after {
    content:""; position:absolute; top:0; bottom:0; width:45%; pointer-events:none;
    background: linear-gradient(90deg, transparent, rgba(34,211,238,.26), transparent);
    transform: translateX(-120%);
}
.st-key-navbar .stButton > button:hover::after { animation: sheen .85s ease forwards; }

.st-key-navbar .stButton > button[kind="primary"] {
    background: transparent !important; box-shadow: none !important; transform: none !important;
    background-image: linear-gradient(90deg,#8B5CF6,#22D3EE) !important; background-size: 220% 100% !important;
    -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important;
    animation: gradientShift 5s ease infinite;
}
.st-key-navbar .stButton > button[kind="primary"]::before {
    content:""; position:absolute; left:16%; right:16%; bottom:2px; height:3px; border-radius:3px;
    background: linear-gradient(90deg,#E879A6,#8B5CF6,#22D3EE);
    box-shadow:0 0 14px rgba(34,211,238,.75); animation: fadeUp .3s ease both;
}
.st-key-navbar .stButton > button[kind="primary"]:hover { transform: translateY(-2px); }

.st-key-theme_wrap .stButton > button {
    border: 1px solid __INPUT_BORDER__ !important; background: __INPUT_BG__ !important;
    color: __BODY__ !important; border-radius: 999px !important;
    font-size:.8rem !important; padding:.42rem .8rem !important;
}
.st-key-theme_wrap .stButton > button:hover {
    border-color:#22D3EE !important; color:__TEXT__ !important;
    box-shadow:0 6px 20px rgba(34,211,238,.26) !important;
}

/* ======== SCROLL REVEAL + STAGGER ======== */
@supports (animation-timeline: view()) {
  .card, .js-plotly-plot, div[data-testid="stExpander"], [data-testid="stDataFrame"] {
      animation: revealUp linear both; animation-timeline: view(); animation-range: entry 0% cover 26%;
  }
  [data-testid="stHorizontalBlock"] > div:nth-child(2) .card { animation-range: entry 0% cover 32%; }
  [data-testid="stHorizontalBlock"] > div:nth-child(3) .card { animation-range: entry 0% cover 38%; }
  [data-testid="stHorizontalBlock"] > div:nth-child(4) .card { animation-range: entry 0% cover 44%; }
  .st-key-navbar { animation-timeline: auto; }
}

/* ======== gradient headings ======== */
.grad-head {
    font-family:'Manrope',sans-serif; font-size:1.82rem; font-weight:800; letter-spacing:-.03em; margin:0;
    background:linear-gradient(90deg,__TEXT__ 0%,#8B5CF6 42%,#22D3EE 78%,#2DD4BF 100%);
    background-size:240% 100%; animation: gradientShift 10s ease infinite;
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}

/* ======== card micro-interaction ======== */
.card { position:relative; overflow:hidden; }
.card::after {
    content:""; position:absolute; inset:0; border-radius:inherit; pointer-events:none; opacity:0;
    background: radial-gradient(440px circle at 50% 0%, rgba(34,211,238,.16), transparent 62%);
    transition: opacity .3s ease;
}
.card:hover::after { opacity:1; }

/* ======== footer ======== */
.site-footer { position:relative; }
.site-footer::before {
    content:""; position:absolute; top:0; left:20%; right:20%; height:2px; border-radius:2px;
    background:linear-gradient(90deg,transparent,#E879A6,#8B5CF6,#3B82F6,#22D3EE,transparent);
    background-size:320% 100%; animation: gradientShift 9s ease infinite;
}
.footer-name { font-weight:700; background:linear-gradient(90deg,#8B5CF6,#22D3EE);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
</style>
"""
st.markdown(build_css(EXTRA_CSS, T), unsafe_allow_html=True)
# ---------------- ANIMATED COUNT-UP METRIC ----------------
_COUNTER_TPL = """
<html><head><style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@600;700;800&display=swap');
body{margin:0;background:transparent;font-family:'Manrope',sans-serif;overflow:hidden;}
.mcard{ background:__CARD__; border:1px solid __BORDER__; border-radius:21px; padding:20px 22px;
  backdrop-filter:blur(16px) saturate(160%); box-shadow:__SHADOW__; position:relative; overflow:hidden;
  transition:transform .25s ease, box-shadow .25s ease, border-color .25s ease;
  animation:fadeUp .5s ease both; cursor:help; }
.mcard:hover{ transform:translateY(-6px); border-color:__ACCENT__AA; box-shadow:__SHADOWH__; }
.mcard::before{ content:""; position:absolute; top:0; left:0; right:0; height:3px;
  background:linear-gradient(90deg,__ACCENT__,__ACCENT2__,transparent); }
@keyframes fadeUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:none}}
.lbl{color:__MUTED__;font-size:.71rem;letter-spacing:.13em;text-transform:uppercase;font-weight:700;}
.val{font-size:2.0rem;font-weight:800;margin-top:6px;line-height:1.1;letter-spacing:-.03em;
     background:linear-gradient(90deg,__TEXT__,__ACCENT__);
     -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.sub{color:__ACCENT__;font-size:.81rem;margin-top:4px;font-weight:600;}
.bar{height:3px;border-radius:3px;margin-top:14px;
     background:linear-gradient(90deg,__ACCENT__,__ACCENT2__,transparent);
     transform-origin:left;animation:grow 1.1s cubic-bezier(.22,.9,.3,1) both;}
@keyframes grow{from{transform:scaleX(0)}to{transform:scaleX(1)}}
</style></head><body>
<div class="mcard" title="__TIP__">
  <div class="lbl">__LABEL__</div>
  <div class="val"><span id="__UID__">0</span></div>
  <div class="sub">__SUB__</div>
  <div class="bar"></div>
</div>
<script>
(function(){
  var el=document.getElementById("__UID__"), target=__VALUE__, dur=1400, dec=__DEC__,
      pre="__PRE__", suf="__SUF__", start=null;
  function fmt(n){var s=n.toFixed(dec);var p=s.split(".");
    p[0]=p[0].replace(/\\B(?=(\\d{3})+(?!\\d))/g,",");return pre+p.join(".")+suf;}
  function step(ts){ if(!start)start=ts; var p=Math.min((ts-start)/dur,1);
    var e=1-Math.pow(1-p,3); el.textContent=fmt(target*e);
    if(p<1)requestAnimationFrame(step); }
  requestAnimationFrame(step);
})();
</script></body></html>
"""

def metric_card(label, value, sub="", prefix="", suffix="", decimals=0,
                tooltip="", accent=CYAN, accent2=VIOLET, height=172):
    uid = "c" + str(abs(hash((label, str(value), sub, st.session_state.theme))) % 1_000_000)
    html = _COUNTER_TPL
    for token, val in [("__UID__", uid), ("__LABEL__", str(label)),
                       ("__VALUE__", str(float(value))), ("__SUB__", str(sub)),
                       ("__PRE__", prefix), ("__SUF__", suffix), ("__DEC__", str(decimals)),
                       ("__TIP__", tooltip), ("__CARD__", T["card"]),
                       ("__BORDER__", T["card_border"]), ("__SHADOW__", T["shadow"]),
                       ("__SHADOWH__", T["shadow_hover"]), ("__TEXT__", T["text"]),
                       ("__MUTED__", T["muted"]), ("__ACCENT2__", accent2),
                       ("__ACCENT__", accent)]:
        html = html.replace(token, val)
    components.html(html, height=height)


def card(title, body, glass=True, delay=0, accent=None, side=False):
    cls = "card glass" if glass else "card"
    style, heading = "", title
    if accent:
        edge = "left" if side else "top"
        style = f'style="border-{edge}:3px solid {accent};height:100%;"'
        heading = f'<span style="color:{accent};">{title}</span>'
    st.markdown(f'<div class="{cls} fade-up delay-{delay}" {style}>'
                f'<h4>{heading}</h4><p>{body}</p></div>', unsafe_allow_html=True)


def section_title(title, subtitle=""):
    st.markdown(f'<div class="fade-up" style="margin:14px 0 18px 0;">'
                f'<h2 class="grad-head">{title}</h2>'
                f'<p style="color:{T["muted"]};margin:7px 0 0 0;">{subtitle}</p></div>',
                unsafe_allow_html=True)


def style_fig(fig, height=420, legend_top=True):
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=T["plot_font"], size=12),
        title_font=dict(family="Manrope, sans-serif", color=T["text"], size=15),
        margin=dict(l=52, r=25, t=52, b=45),
        hoverlabel=dict(bgcolor=T["hover_bg"], bordercolor=T["hover_border"],
                        font_color=T["hover_font"], font_size=12),
        legend=(dict(orientation="h", y=1.13, x=0, bgcolor="rgba(0,0,0,0)")
                if legend_top else dict(bgcolor="rgba(0,0,0,0)")),
        transition=dict(duration=500, easing="cubic-in-out"))
    fig.update_xaxes(gridcolor=T["plot_grid"], zerolinecolor=T["plot_grid"], linecolor=T["plot_grid"])
    fig.update_yaxes(gridcolor=T["plot_grid"], zerolinecolor=T["plot_grid"], linecolor=T["plot_grid"])
    return fig


def chart(fig, height=420, legend_top=True):
    st.plotly_chart(style_fig(fig, height, legend_top), width="stretch")


def spacer(px=16):
    st.markdown(f"<div style='height:{px}px'></div>", unsafe_allow_html=True)


def render_footer():
    st.markdown(
        '<div class="site-footer">'
        '<p>Created by <span class="footer-name">Yalda Ashrafi</span> © 2026 — All rights reserved.</p>'
        '<p style="font-size:.78rem;margin-top:6px;opacity:.75;">'
        'FraudSense AI · SMOTE + imblearn Pipeline · Logistic Regression &amp; Random Forest</p>'
        '</div>', unsafe_allow_html=True)
    # ---------------- DATA SOURCE ----------------
REQUIRED = ["Class"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

@st.cache_data(show_spinner="Reading transactions…")
def parse_csv(file_bytes, _label):
    """Parse an uploaded CSV. _label only varies the cache key."""
    raw = pd.read_csv(io.BytesIO(file_bytes))
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing[:6])}")
    if "Time" in raw.columns:
        raw["Hour"] = (raw["Time"] / 3600) % 24
        raw = raw.drop(columns=["Time"])
    elif "Hour" not in raw.columns:
        raw["Hour"] = 0.0
    return raw

@st.cache_data(show_spinner="Loading default dataset…")
def load_default():
    raw = pd.read_csv("creditcard.csv")
    raw["Hour"] = (raw["Time"] / 3600) % 24
    return raw.drop(columns=["Time"])

@st.cache_resource(show_spinner="Loading trained pipelines…")
def load_models():
    return {
        "Logistic Regression": joblib.load("models/lr_pipeline.pkl"),
        "Random Forest":       joblib.load("models/rf_pipeline.pkl"),
    }

if "upload" not in st.session_state:
    st.session_state.upload = None          # (bytes, filename)

def resolve_data():
    """Uploaded file wins; otherwise fall back to creditcard.csv on disk."""
    if st.session_state.upload is not None:
        blob, fname = st.session_state.upload
        return parse_csv(blob, fname), fname
    if os.path.exists("creditcard.csv"):
        return load_default(), "creditcard.csv"
    return None, None

models = load_models()
try:
    df, DATA_NAME = resolve_data()
except ValueError as err:
    st.error(f"That CSV can't be used — {err}")
    st.session_state.upload = None
    df, DATA_NAME = None, None

# ---- boot gate: no dataset available yet ----
if df is None:
    st.markdown("""
    <div class="hero">
      <div class="pill">Getting started</div>
      <h1>Load a dataset</h1>
      <p>Drop in a Credit Card Fraud Detection CSV to begin. It needs the columns
      <b>V1–V28</b>, <b>Amount</b> and <b>Class</b>. <b>Time</b> is optional and is converted
      to hour-of-day automatically.</p>
    </div>""", unsafe_allow_html=True)
    boot = st.file_uploader("Upload transactions CSV", type=["csv"], key="boot_upload")
    if boot is not None:
        st.session_state.upload = (boot.getvalue(), boot.name)
        st.rerun()
    render_footer()
    st.stop()

X, y = df.drop(columns=["Class"]), df["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

@st.cache_data(show_spinner=False)
def get_proba(model_name, _tag):
    return models[model_name].predict_proba(X_test)[:, 1]

@st.cache_data(show_spinner=False)
def get_corr(_tag):
    return df.corr()["Class"].drop("Class").sort_values()

# ---------------- NAVIGATION ----------------
PAGES = ["Home", "Dataset", "Pipeline", "Models", "Prediction", "About"]

if "page" not in st.session_state:
    st.session_state.page = "Home"

def navigate_to(page):
    st.session_state.page = page

def toggle_theme():
    st.session_state.theme = "Dark" if st.session_state.theme == "Light" else "Light"

with st.container(key="navbar"):
    nav = st.columns([2.0, 1, 1, 1, 1, 1.15, 1, 1.15])

    with nav[0]:
        st.markdown(
            '<div class="brand-wrap">'
            '  <div class="brand-mark">🛡️</div>'
            '  <div class="brand-text">'
            '    <div class="brand-name">FraudSense AI</div>'
            '    <div class="brand-tag">Detection Pipeline</div>'
            '  </div>'
            '</div>', unsafe_allow_html=True)

    for col, name in zip(nav[1:7], PAGES):
        with col:
            st.button(name, key=f"nav_{name}",
                      type="primary" if st.session_state.page == name else "secondary",
                      on_click=navigate_to, args=(name,))

    with nav[7]:
        with st.container(key="theme_wrap"):
            nxt = "Dark" if st.session_state.theme == "Light" else "Light"
            st.button(nxt, key="theme_btn", on_click=toggle_theme,
                      help="Switch between light and dark mode")

page = st.session_state.page
PALETTE = {"Logistic Regression": VIOLET, "Random Forest": CYAN}
N_ROWS = len(df)
FRAUD_N = int(y.sum())
# =================== HOME ===================
if page == "Home":
    st.markdown(f"""
    <div class="hero">
      <div class="pill">Supervised Learning · Imbalanced Data</div>
      <h1>Detecting fraud in<br>{100*y.mean():.2f}% of the signal.</h1>
      <p>A leak-free machine learning pipeline combining stratified splitting, SMOTE
      synthetic oversampling and linear/ensemble classifiers to surface fraudulent card
      transactions hidden inside {N_ROWS:,} records — evaluated on Precision, Recall and
      ROC-AUC, never on accuracy.</p>
      <div style="margin-top:16px;">
        <span class="tag">imblearn Pipeline</span><span class="tag">SMOTE</span>
        <span class="tag">Logistic Regression</span><span class="tag">Random Forest</span>
        <span class="tag">GridSearchCV</span><span class="tag">Zero Leakage</span>
      </div>
    </div>""", unsafe_allow_html=True)

    b1, b2, _ = st.columns([1, 1, 3])
    b1.button("Try Prediction", type="primary", key="cta1",
              on_click=navigate_to, args=("Prediction",))
    b2.button("Load Data", key="cta2", on_click=navigate_to, args=("Dataset",))

    spacer(22)
    m = st.columns(4)
    with m[0]: metric_card("Transactions", N_ROWS, DATA_NAME, accent=BLUE, accent2=INDIGO,
                           tooltip="Rows in the currently loaded dataset.")
    with m[1]: metric_card("Fraud cases", FRAUD_N, f"{100*y.mean():.3f}% of volume",
                           tooltip="Positive class — the minority the model must find.",
                           accent=ROSE, accent2=MAGENTA)
    with m[2]: metric_card("Features", X.shape[1], "V1–V28 + Amount + Hour",
                           accent=CYAN, accent2=TEAL,
                           tooltip="V1–V28 are PCA components released by the issuing bank.")
    with m[3]: metric_card("Imbalance ratio", int((y == 0).sum() / max(FRAUD_N, 1)),
                           "legit per fraud", prefix="1 : ", accent=VIOLET, accent2=MAGENTA,
                           tooltip="How many legitimate rows exist for every fraud.")

    spacer(10)
    c = st.columns(3)
    with c[0]: card("Accuracy is a trap",
                    "Predicting “legitimate” for every row scores 99.83% accuracy and catches "
                    "zero fraud. This dashboard reports Precision, Recall, F1, ROC-AUC and PR-AUC only.",
                    accent=ROSE, delay=1)
    with c[1]: card("Synthetic balancing",
                    "SMOTE interpolates new minority points between real fraud cases and their "
                    "nearest neighbours — it creates, it never clones.", accent=VIOLET, delay=2)
    with c[2]: card("Zero leakage",
                    "Scaling and resampling live inside an imblearn Pipeline, so they run on "
                    "training folds only. The test set keeps its real fraud rate.",
                    accent=TEAL, delay=3)

    spacer(18)
    section_title("How it works", "Four stages, end to end")
    stages = [("① Split", "Stratified 80/20 — the test set is set aside before anything else touches the data.", VIOLET),
              ("② Balance", "SMOTE runs inside the pipeline, on training folds only.", MAGENTA),
              ("③ Train", "Logistic Regression and Random Forest, tuned with GridSearchCV.", BLUE),
              ("④ Score", "Precision, Recall, F1, ROC-AUC and PR-AUC on untouched data.", TEAL)]
    for col, (t_, b_, a_) in zip(st.columns(4), stages):
        with col:
            card(t_, b_, accent=a_)

# =================== DATASET ===================
elif page == "Dataset":
    section_title("The Data Problem",
                  "Why extreme class imbalance breaks conventional model training")

    # ---------- UPLOAD PANEL ----------
    with st.expander(f"Data source — currently loaded: {DATA_NAME}", expanded=False):
        st.markdown(
            f'<p style="color:{T["muted"]};font-size:.9rem;margin-bottom:10px;">'
            'Upload your own transactions CSV. Required columns: <b>V1–V28</b>, '
            '<b>Amount</b>, <b>Class</b>. <b>Time</b> is optional and becomes hour-of-day. '
            'Every page — charts, model comparison and live scoring — switches to the new file.'
            '</p>', unsafe_allow_html=True)
        up = st.file_uploader("Choose a CSV", type=["csv"], key="ds_upload",
                              label_visibility="collapsed")
        u1, u2, _ = st.columns([1, 1, 2])
        if up is not None:
            if u1.button("Use this file", type="primary", key="use_upload"):
                st.session_state.upload = (up.getvalue(), up.name)
                st.rerun()
        if st.session_state.upload is not None:
            if u2.button("Revert to default", key="revert_upload"):
                st.session_state.upload = None
                st.rerun()
        st.markdown(
            f'<p style="color:{T["muted"]};font-size:.82rem;margin-top:10px;">'
            'Note: the saved pipelines expect these exact 30 feature columns. A file with '
            'different features will not score correctly.</p>', unsafe_allow_html=True)

    spacer(10)
    c = st.columns(3)
    with c[0]: metric_card("Total transactions", N_ROWS, DATA_NAME, accent=BLUE, accent2=INDIGO,
                           tooltip="Rows in the currently loaded file.")
    with c[1]: metric_card("Legitimate", int((y == 0).sum()), f"{100*(y==0).mean():.2f}%",
                           tooltip="Negative class — overwhelming majority.",
                           accent=TEAL, accent2=CYAN)
    with c[2]: metric_card("Fraudulent", FRAUD_N, f"{100*(y==1).mean():.2f}%",
                           tooltip="Positive class — what the model must find.",
                           accent=ROSE, accent2=MAGENTA)

    spacer(6)
    card("The 99.83% illusion",
         "A model that classifies every transaction as legitimate achieves near-perfect "
         "accuracy while resulting in catastrophic financial loss. Accuracy is therefore "
         "discarded in favour of Precision, Recall and ROC-AUC.",
         accent=AMBER, side=True)
    spacer(18)

    counts = y.value_counts().sort_index()
    left, right = st.columns([1.15, 1])
    with left:
        fig = go.Figure(go.Bar(
            x=["Legitimate", "Fraudulent"], y=counts.values,
            marker=dict(color=[BLUE, ROSE], line=dict(color=T["card_border"], width=1)),
            text=[f"{v:,}" for v in counts.values], textposition="outside",
            textfont=dict(color=T["text"], size=13),
            hovertemplate="<b>%{x}</b><br>%{y:,} transactions<extra></extra>", width=.5))
        fig.update_yaxes(type="log", title="Transactions (log scale)")
        fig.update_layout(title="Class distribution")
        chart(fig, 400)
    with right:
        fig = go.Figure(go.Pie(
            labels=["Legitimate", "Fraudulent"], values=counts.values, hole=.68,
            marker=dict(colors=[INDIGO, ROSE], line=dict(color=T["bg"], width=3)),
            textinfo="percent", textfont=dict(color="#FFFFFF", size=14), pull=[0, .18],
            hovertemplate="<b>%{label}</b><br>%{value:,}<extra></extra>"))
        fig.update_layout(title="Proportion of each class",
                          annotations=[dict(text=f"{100*y.mean():.2f}%<br>fraud", x=.5, y=.5,
                                            showarrow=False,
                                            font=dict(size=18, color=ROSE))])
        chart(fig, 400)

    section_title("Feature signal", "Correlation of each feature with the fraud label")
    corr = get_corr(DATA_NAME)
    fig = go.Figure(go.Bar(
        x=corr.values, y=corr.index, orientation="h",
        marker=dict(color=corr.values,
                    colorscale=[[0, ROSE], [.5, T["neutral_bar"]], [1, CYAN]],
                    line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>corr = %{x:.4f}<extra></extra>"))
    fig.update_layout(title="V17, V14, V12 and V10 carry the strongest fraud signal",
                      xaxis_title="Pearson correlation with Class")
    chart(fig, 720, legend_top=False)

    with st.expander("Inspect raw records"):
        st.dataframe(df.head(50), width="stretch", height=380)
        # =================== PIPELINE ===================
elif page == "Pipeline":
    section_title("The Leak-Free Pipeline",
                  "Order of operations, and why each step lives where it does")

    st.code("""from imblearn.pipeline import Pipeline   # NOT sklearn.pipeline

Pipeline([
    ("scaler",     StandardScaler()),
    ("smote",      SMOTE(random_state=42)),
    ("classifier", LogisticRegression(max_iter=1000))
])""", language="python")

    spacer(10)
    steps = [
        ("01 · Stratified Split", VIOLET,
         "train_test_split(..., stratify=y) runs <b>first</b>, preserving the exact fraud "
         "ratio in both partitions. With so few positives, a random split could starve either side."),
        ("02 · StandardScaler", INDIGO,
         "Fitted on training folds only. Fitting on all data would leak the test set's mean "
         "and variance into training. It also keeps SMOTE's k-NN distances sane, since Amount "
         "ranges to 25,000 while V1–V28 sit near ±3."),
        ("03 · SMOTE", CYAN,
         "Generates synthetic fraud points between a real fraud case and its nearest "
         "neighbours: x_new = x_i + λ(x_nn − x_i). Applied to training folds only."),
        ("04 · Classifier", TEAL,
         "Trained on the balanced fold. On .predict(), imblearn automatically skips the "
         "resampling step — the test set is never touched."),
    ]
    for col, (title, colour, body) in zip(st.columns(4), steps):
        with col:
            card(title, body, accent=colour)

    spacer(20)
    bad, good = st.columns(2)
    with bad:
        card("✗ SMOTE → Split",
             "Synthetic rows interpolated from training data end up inside the test set. "
             "The model is graded on answers it has already seen. Scores look brilliant "
             "and mean nothing.", accent=ROSE)
    with good:
        card("✓ Split → Scale → SMOTE → Train",
             "Resampling is isolated inside every cross-validation fold. The blind exam "
             "reflects the extreme imbalance of the real world.", accent=TEAL)

    section_title("Effect of SMOTE on the training set", "Minority class oversampled to parity")
    before = y_train.value_counts().sort_index()
    fig = go.Figure()
    fig.add_bar(name="Before SMOTE", x=["Legitimate", "Fraud"],
                y=[int(before[0]), int(before[1])], marker_color=T["neutral_bar"],
                text=[f"{before[0]:,}", f"{before[1]:,}"], textposition="outside",
                textfont=dict(color=T["text"]))
    fig.add_bar(name="After SMOTE", x=["Legitimate", "Fraud"],
                y=[int(before[0]), int(before[0])], marker_color=CYAN,
                text=[f"{before[0]:,}", f"{before[0]:,}"], textposition="outside",
                textfont=dict(color=T["text"]))
    fig.update_layout(barmode="group", title="Training folds only — the test set is untouched")
    chart(fig, 400)

    c1, c2 = st.columns(2)
    with c1: card("Logistic Regression",
                  "Linear decision boundary, high coefficient transparency, very fast. "
                  "Fatal without a scaler — regularisation penalties are distorted by large "
                  "transaction-amount variance.", accent=VIOLET)
    with c2: card("Random Forest",
                  "Highly complex non-linear boundary, moderate interpretability, slower to "
                  "train. Splits are ordinal, so it is naturally invariant to feature scale.",
                  accent=CYAN, delay=1)

# =================== MODELS ===================
elif page == "Models":
    section_title("Model Performance",
                  f"Evaluated on the untouched 20% test set — {len(X_test):,} transactions")

    top = st.columns([2, 1.4])
    with top[0]:
        choice = st.multiselect("Models to compare", list(models), default=list(models))
    with top[1]:
        threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01,
                              help="Lower = catch more fraud, but more false alarms")

    if not choice:
        st.info("Select at least one model to display results.")
    else:
        rows, curves = [], {}
        for name in choice:
            proba = get_proba(name, DATA_NAME)
            pred = (proba >= threshold).astype(int)
            rows.append({
                "Model":     name,
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall":    recall_score(y_test, pred),
                "F1":        f1_score(y_test, pred),
                "ROC-AUC":   roc_auc_score(y_test, proba),
                "PR-AUC":    average_precision_score(y_test, proba),
            })
            curves[name] = (proba, pred)

        lead = max(rows, key=lambda r: r["F1"])
        tips = {"Precision": "Of transactions flagged as fraud, how many really were.",
                "Recall": "Of all real fraud, how much the model caught.",
                "F1": "Harmonic mean of precision and recall.",
                "ROC-AUC": "Ability to separate the two distributions overall.",
                "PR-AUC": "Average precision — the honest metric under extreme imbalance."}
        keys = ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
        for col, key, acc in zip(st.columns(5), keys, [VIOLET, MAGENTA, BLUE, CYAN, TEAL]):
            with col:
                metric_card(key, lead[key] * 100, lead["Model"], suffix="%", decimals=2,
                            tooltip=tips[key], accent=acc, accent2=VIOLET)

        st.dataframe(pd.DataFrame(rows).set_index("Model").style
                     .format("{:.4f}").background_gradient(cmap="BuPu", axis=None),
                     width="stretch")

        section_title("Confusion Matrices", f"At threshold = {threshold:.2f}")
        for col, (name, (proba, pred)) in zip(st.columns(len(curves)), curves.items()):
            tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
            with col:
                fig = go.Figure(go.Heatmap(
                    z=[[tn, fp], [fn, tp]],
                    x=["Predicted: Legit", "Predicted: Fraud"],
                    y=["Actual: Legit", "Actual: Fraud"],
                    text=[[f"{tn:,}", f"{fp:,}"], [f"{fn:,}", f"{tp:,}"]],
                    texttemplate="%{text}", textfont=dict(size=17, color=T["text"]),
                    colorscale=T["heat"], showscale=False,
                    hovertemplate="%{y} → %{x}<br>%{text}<extra></extra>"))
                fig.update_layout(title=name)
                chart(fig, 330, legend_top=False)
                st.markdown(
                    f'<div class="card glass" style="padding:14px 18px;">'
                    f'<p><b style="color:{TEAL};">Caught:</b> {tp}/{tp+fn} &nbsp;·&nbsp; '
                    f'<b style="color:{ROSE};">Missed:</b> {fn} &nbsp;·&nbsp; '
                    f'<b style="color:{AMBER};">False alarms:</b> {fp:,}</p></div>',
                    unsafe_allow_html=True)

        section_title("Discrimination Curves", "Threshold-independent view of model quality")
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            for name, (proba, _) in curves.items():
                fpr, tpr, _ = roc_curve(y_test, proba)
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    name=f"{name} · {roc_auc_score(y_test, proba):.4f}",
                    line=dict(color=PALETTE.get(name, CYAN), width=3, shape="spline"),
                    fill="tozeroy", fillcolor="rgba(139,92,246,.10)",
                    hovertemplate="FPR %{x:.4f}<br>TPR %{y:.4f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random (0.50)",
                                     line=dict(color=T["muted"], width=1.5, dash="dash")))
            fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate",
                              yaxis_title="True Positive Rate (Recall)")
            chart(fig, 440)
        with c2:
            fig = go.Figure()
            for name, (proba, _) in curves.items():
                p, r, _ = precision_recall_curve(y_test, proba)
                fig.add_trace(go.Scatter(
                    x=r, y=p, mode="lines",
                    name=f"{name} · {average_precision_score(y_test, proba):.4f}",
                    line=dict(color=PALETTE.get(name, CYAN), width=3),
                    hovertemplate="Recall %{x:.4f}<br>Precision %{y:.4f}<extra></extra>"))
            fig.add_hline(y=float(y_test.mean()),
                          line=dict(color=T["muted"], dash="dash", width=1.5),
                          annotation_text=f"No-skill ({y_test.mean():.4f})",
                          annotation_font_color=T["muted"])
            fig.update_layout(title="Precision–Recall Curve", xaxis_title="Recall",
                              yaxis_title="Precision")
            chart(fig, 440)

        card("Reading the trade-off",
             "ROC-AUC flatters every model here — with so many negatives, the false-positive "
             "rate barely moves. PR-AUC is the stricter, more honest metric under this "
             "prevalence. Lower the threshold to buy recall at the cost of precision; that "
             "choice belongs to the business, not the algorithm.",
             accent=CYAN, side=True)
        # =================== PREDICTION ===================
elif page == "Prediction":
    section_title("Live Transaction Scoring",
                  "Push a single record through the trained pipeline")

    c1, c2, c3 = st.columns([1.2, 1.6, 1.2])
    with c1:
        model_name = st.selectbox("Model", list(models))
    with c2:
        mode = st.radio("Transaction source",
                        ["Random fraud", "Random legitimate", "Enter manually"],
                        horizontal=True)
    with c3:
        threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)

    model = models[model_name]

    if mode == "Random fraud":
        pool = X_test[y_test == 1]
        row = pool.sample(1, random_state=np.random.randint(10_000)) if len(pool) else X_test.head(1)
        truth = 1 if len(pool) else None
    elif mode == "Random legitimate":
        row = X_test[y_test == 0].sample(1, random_state=np.random.randint(10_000))
        truth = 0
    else:
        card("Manual feature entry",
             "V1–V28 default to 0 (the PCA mean). The sliders expose the four components most "
             "correlated with fraud, plus amount and hour of day.", accent=CYAN, side=True)
        vals = {col: 0.0 for col in X.columns}
        f1c, f2c, f3c = st.columns(3)
        vals["Amount"] = f1c.number_input("Amount", 0.0, 30000.0, 100.0)
        vals["Hour"]   = f2c.slider("Hour of day", 0.0, 24.0, 12.0)
        vals["V14"]    = f3c.slider("V14", -20.0, 10.0, 0.0)
        vals["V17"]    = f1c.slider("V17", -25.0, 10.0, 0.0)
        vals["V12"]    = f2c.slider("V12", -20.0, 10.0, 0.0)
        vals["V10"]    = f3c.slider("V10", -25.0, 10.0, 0.0)
        row = pd.DataFrame([vals])[X.columns]
        truth = None

    spacer(8)
    run = st.button("Analyse Transaction", type="primary")

    if run:
        proba = float(model.predict_proba(row)[0, 1])
        flagged = proba >= threshold

        left, right = st.columns([1, 1.25])
        with left:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=proba * 100,
                number=dict(suffix="%", font=dict(size=42, color=T["text"])),
                title=dict(text="Fraud probability", font=dict(size=14, color=T["muted"])),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor=T["muted"],
                              tickfont=dict(color=T["muted"], size=10)),
                    bar=dict(color=ROSE if flagged else TEAL, thickness=.28),
                    bgcolor=T["input_bg"], borderwidth=1, bordercolor=T["card_border"],
                    steps=[dict(range=[0, 25],   color="rgba(45,212,191,.16)"),
                           dict(range=[25, 60],  color="rgba(251,191,36,.16)"),
                           dict(range=[60, 100], color="rgba(251,113,133,.18)")],
                    threshold=dict(line=dict(color=CYAN, width=4),
                                   thickness=.9, value=threshold * 100))))
            chart(gauge, 330, legend_top=False)

        with right:
            if flagged:
                st.markdown('<div class="verdict v-fraud">FLAGGED AS FRAUD<br>'
                            '<span style="font-weight:400;font-size:.9rem;">'
                            'Transaction blocked pending manual review.</span></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="verdict v-safe">APPROVED AS LEGITIMATE<br>'
                            '<span style="font-weight:400;font-size:.9rem;">'
                            'Score below the decision threshold.</span></div>',
                            unsafe_allow_html=True)

            spacer(14)
            g1, g2 = st.columns(2)
            with g1: metric_card("Model score", proba * 100, model_name, suffix="%",
                                 decimals=3, tooltip="Pipeline output probability for class 1.",
                                 accent=ROSE if flagged else TEAL, accent2=MAGENTA, height=164)
            with g2: metric_card("Threshold", threshold * 100, "decision cut-off", suffix="%",
                                 decimals=0, tooltip="Scores at or above this are flagged.",
                                 accent=CYAN, accent2=BLUE, height=164)

            if truth is not None:
                correct = flagged == bool(truth)
                actual = "Fraud" if truth == 1 else "Legitimate"
                colour = TEAL if correct else ROSE
                card(f"Ground truth: {actual}",
                     f'Prediction was <b style="color:{colour};">'
                     f'{"correct" if correct else "INCORRECT"}</b> at this threshold.',
                     accent=colour, side=True)

        with st.expander("Feature values used for this prediction"):
            st.dataframe(row.T.rename(columns={row.index[0]: "Value"}),
                         width="stretch", height=420)

    # ---------- BATCH SCORING ----------
    spacer(26)
    section_title("Batch scoring", "Score a whole file of transactions at once")

    batch_file = st.file_uploader("Upload transactions to score (CSV)", type=["csv"],
                                  key="batch_upload")
    if batch_file is not None:
        batch = pd.read_csv(io.BytesIO(batch_file.getvalue()))
        if "Time" in batch.columns:
            batch["Hour"] = (batch["Time"] / 3600) % 24
            batch = batch.drop(columns=["Time"])
        elif "Hour" not in batch.columns:
            batch["Hour"] = 0.0

        missing = [c for c in X.columns if c not in batch.columns]
        if missing:
            st.error(f"Missing column(s): {', '.join(missing[:8])}")
        else:
            scores = model.predict_proba(batch[X.columns])[:, 1]
            out = batch.copy()
            out["fraud_probability"] = scores
            out["prediction"] = np.where(scores >= threshold, "FRAUD", "LEGITIMATE")
            n_flag = int((scores >= threshold).sum())

            b = st.columns(3)
            with b[0]: metric_card("Rows scored", len(out), batch_file.name,
                                   accent=BLUE, accent2=INDIGO,
                                   tooltip="Total transactions in the uploaded file.")
            with b[1]: metric_card("Flagged as fraud", n_flag, f"{100*n_flag/len(out):.2f}% of file",
                                   accent=ROSE, accent2=MAGENTA,
                                   tooltip="Rows scoring at or above the threshold.")
            with b[2]: metric_card("Highest score", float(scores.max()) * 100,
                                   "riskiest transaction", suffix="%", decimals=2,
                                   accent=AMBER, accent2=ROSE,
                                   tooltip="Maximum fraud probability in this batch.")

            fig = go.Figure(go.Histogram(
                x=scores, nbinsx=50, marker=dict(color=CYAN, line=dict(width=0)),
                hovertemplate="Score %{x:.3f}<br>%{y} transactions<extra></extra>"))
            fig.add_vline(x=threshold, line=dict(color=ROSE, dash="dash", width=2),
                          annotation_text="threshold", annotation_font_color=T["muted"])
            fig.update_yaxes(type="log", title="Transactions (log scale)")
            fig.update_layout(title="Distribution of fraud scores",
                              xaxis_title="Predicted fraud probability")
            chart(fig, 380, legend_top=False)

            st.dataframe(out.sort_values("fraud_probability", ascending=False).head(100),
                         width="stretch", height=380)
            st.download_button("Download scored CSV",
                               data=out.to_csv(index=False).encode("utf-8"),
                               file_name=f"scored_{batch_file.name}",
                               mime="text/csv", type="primary")

# =================== ABOUT ===================
elif page == "About":
    section_title("About this project",
                  "Project 2 — Supervised Learning (Fraud Detection Pipeline)")

    c = st.columns(3)
    with c[0]: card("Objective",
                    "Build and tune a classification model that identifies fraudulent "
                    "transactions in a highly imbalanced dataset, using strict Precision, "
                    "Recall and ROC-AUC evaluation rather than accuracy.", accent=VIOLET)
    with c[1]: card("Dataset",
                    "Kaggle Credit Card Fraud Detection — 284,807 anonymised European card "
                    "transactions over two days. Features V1–V28 are PCA components; Amount "
                    "and Hour are raw. 492 fraud cases (0.17%). Custom CSVs can be uploaded "
                    "on the Dataset page.", accent=BLUE)
    with c[2]: card("Stack",
                    "Python · pandas · NumPy · scikit-learn · imbalanced-learn · Plotly · "
                    "Streamlit · joblib. Models persisted as .pkl pipelines and loaded at runtime.",
                    accent=CYAN)

    spacer(14)
    c1, c2 = st.columns(2)
    with c1: card("Method",
                  "Stratified 80/20 split → StandardScaler → SMOTE → classifier, all wrapped "
                  "in an imblearn Pipeline so resampling is confined to training folds. "
                  "Hyperparameters tuned with GridSearchCV using average precision as the "
                  "scoring function.", accent=MAGENTA)
    with c2: card("Key finding",
                  "Logistic Regression maximises recall but floods the queue with false "
                  "positives; Random Forest holds a far better precision–recall balance. The "
                  "decision threshold, not the algorithm, controls where a bank sits on that "
                  "trade-off.", accent=TEAL, delay=1)

    with st.expander("Project structure"):
        st.code("""fraud-detection/
├── creditcard.csv
├── 01_load.py        # data loading & inspection
├── 02_explore.py     # EDA and class imbalance charts
├── 03_split.py       # stratified train/test split
├── 04_train.py       # imblearn pipelines + model fitting
├── 05_evaluate.py    # metrics, confusion matrix, ROC, PR curves
├── 06_tune.py        # GridSearchCV hyperparameter tuning
├── app.py            # this Streamlit dashboard
└── models/
    ├── lr_pipeline.pkl
    └── rf_pipeline.pkl""", language="text")

# ---------------- FOOTER (renders on every page) ----------------
render_footer()