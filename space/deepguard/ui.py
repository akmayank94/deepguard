"""
deepguard/ui.py — visual design for the DeepGuard Space.

Concept: a forensic video-lab instrument. One ambient animation (the face-landmark
scan in the hero) and one orchestrated reveal when a report appears. Everything
else stays still.
"""

import html
import math
import random

import numpy as np

# ── Tokens ────────────────────────────────────────────────────────────────────
INK = "#0E1A22"        # page
PANEL = "#14242F"      # raised surfaces
PANEL_2 = "#1A2E3B"    # inputs / inner wells
LINE = "#22394A"       # hairlines
TEXT = "#E4EDF1"
MUTED = "#8DA4B2"
SCAN = "#5CE1D0"       # phosphor teal: scanning + actions
FAKE = "#FF5D5D"
REAL = "#3DDC97"
WARN = "#F2B544"

# ── Icons (24px stroke icons, drawn for this app) ─────────────────────────────
_I = {
    "shield": '<path d="M12 3l7 3v5c0 4.6-3 8.3-7 10-4-1.7-7-5.4-7-10V6l7-3z"/><path d="M9 12l2 2 4-4"/>',
    "upload": '<path d="M12 15V4"/><path d="M7.5 8.5L12 4l4.5 4.5"/><path d="M5 15v3a2 2 0 002 2h10a2 2 0 002-2v-3"/>',
    "report": '<rect x="5" y="3" width="14" height="18" rx="2"/><path d="M9 8h6M9 12h6M9 16h3"/>',
    "face": '<circle cx="12" cy="12" r="8.5"/><circle cx="9" cy="10.5" r=".9"/><circle cx="15" cy="10.5" r=".9"/><path d="M9 15c1.7 1.3 4.3 1.3 6 0"/>',
    "spectrum": '<path d="M3 12h2l2-6 3 12 3-9 2 5 2-2h4"/>',
    "texture": '<path d="M4 20L20 4M4 14L14 4M10 20L20 10M4 8l4-4M16 20l4-4"/>',
    "sensor": '<rect x="6" y="6" width="12" height="12" rx="1.5"/><rect x="9.5" y="9.5" width="5" height="5"/><path d="M9 3v3M15 3v3M9 18v3M15 18v3M3 9h3M3 15h3M18 9h3M18 15h3"/>',
    "pulse": '<path d="M20.5 9.5c0 5-8.5 10.5-8.5 10.5S3.5 14.5 3.5 9.5A4.5 4.5 0 0112 7a4.5 4.5 0 018.5 2.5z"/><path d="M6 12h3l1.5-2.5 2 5 1.5-2.5h4"/>',
    "alert": '<path d="M12 4l9 16H3L12 4z"/><path d="M12 10v4M12 17.2v.1"/>',
    "check": '<circle cx="12" cy="12" r="9"/><path d="M8 12.5l2.7 2.7L16.5 9.5"/>',
    "cross": '<circle cx="12" cy="12" r="9"/><path d="M9 9l6 6M15 9l-6 6"/>',
    "film": '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M7 5v14M17 5v14M3 9h4M3 15h4M17 9h4M17 15h4"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "flip": '<path d="M12 3v18"/><path d="M8 7L4 12l4 5V7zM16 7l4 5-4 5V7z"/>',
}


def icon(name, size=18, cls="dg-ic"):
    return (f'<svg class="{cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" '
            f'aria-hidden="true">{_I[name]}</svg>')


# ── Branch vocabulary (plain-language names users understand) ─────────────────
BRANCHES = [
    dict(key="A", name="Facial appearance", icon="face", color="#7AA2FF",
         what="Visible texture, blending and lighting artifacts", net="EfficientNet-B3"),
    dict(key="B", name="Colour-channel spectrum", icon="spectrum", color=SCAN,
         what="High-frequency energy that disagrees between R, G and B", net="EfficientNet-B0"),
    dict(key="C", name="Directional texture", icon="texture", color="#B58CFF",
         what="Filter-bank energy and edge-direction co-occurrence", net="MLP · 9,369 features"),
    dict(key="D", name="Camera sensor noise", icon="sensor", color="#FF9F5A",
         what="Whether noise looks like it came from a real camera sensor", net="MLP · 30 features"),
    dict(key="E", name="Skin pulse", icon="pulse", color="#FF7AB6",
         what="Tiny heartbeat colour changes in the skin over ~5 seconds", net="MLP · 32 features"),
]


# ── Hero: face-landmark scan (generated once, deterministic) ──────────────────
def _face_mesh_svg():
    rng = random.Random(7)
    cx, cy, rx, ry = 150, 158, 86, 110

    def ell(x0, y0, a, b, n, t0=0.0, t1=2 * math.pi):
        return [(x0 + a * math.cos(t0 + (t1 - t0) * i / n), y0 + b * math.sin(t0 + (t1 - t0) * i / n))
                for i in range(n + (0 if t1 - t0 >= 2 * math.pi - 1e-6 else 1))]

    oval = ell(cx, cy, rx, ry, 40)
    eyes = ell(cx - 34, 138, 17, 7, 12) + ell(cx + 34, 138, 17, 7, 12)
    brows = ell(cx - 34, 124, 22, 9, 6, math.pi * 1.1, math.pi * 1.9) + \
        ell(cx + 34, 124, 22, 9, 6, math.pi * 1.1, math.pi * 1.9)
    nose = [(cx, 140 + 9 * i) for i in range(6)] + [(cx - 11, 190), (cx - 5, 194), (cx + 5, 194), (cx + 11, 190)]
    mouth = ell(cx, 222, 28, 9, 14)
    inner = []
    while len(inner) < 70:
        x, y = rng.uniform(cx - rx, cx + rx), rng.uniform(cy - ry, cy + ry)
        if ((x - cx) / (rx * 0.92)) ** 2 + ((y - cy) / (ry * 0.92)) ** 2 < 1:
            inner.append((x, y))
    pts = oval + eyes + brows + nose + mouth + inner

    def poly(p, closed=True):
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in p)
        return f'<path d="{d}{" Z" if closed else ""}"/>'

    lines = (poly(oval) + poly(eyes[:12]) + poly(eyes[12:]) + poly(brows[:7], False) +
             poly(brows[7:], False) + poly(nose[:6], False) + poly(nose[6:], False) + poly(mouth))
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{1.6 if i < 120 else 1.2}" '
        f'style="animation-delay:{(i * 137) % 3000}ms"/>' for i, (x, y) in enumerate(pts))
    corners = ('<path d="M34 30h-14v14M266 30h14v14M34 290h-14v-14M266 290h14v-14"/>')
    return f"""
<svg class="dg-mesh" viewBox="0 0 300 320" role="img" aria-label="Face landmarks being scanned">
  <defs>
    <linearGradient id="dgBeam" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="{SCAN}" stop-opacity="0"/>
      <stop offset=".85" stop-color="{SCAN}" stop-opacity=".22"/>
      <stop offset="1" stop-color="{SCAN}" stop-opacity=".9"/>
    </linearGradient>
    <clipPath id="dgFrame"><rect x="20" y="30" width="260" height="260"/></clipPath>
  </defs>
  <g class="dg-mesh-corners">{corners}</g>
  <g class="dg-mesh-lines">{lines}</g>
  <g class="dg-mesh-dots">{dots}</g>
  <g clip-path="url(#dgFrame)"><rect class="dg-beam" x="20" y="-30" width="260" height="60" fill="url(#dgBeam)"/></g>
</svg>"""


MESH_SVG = _face_mesh_svg()


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;600;700&family=Instrument+Sans:wght@400;500;600&display=swap');

:root, .dark {{
  --dg-ink:{INK}; --dg-panel:{PANEL}; --dg-panel2:{PANEL_2}; --dg-line:{LINE};
  --dg-text:{TEXT}; --dg-muted:{MUTED}; --dg-scan:{SCAN}; --dg-fake:{FAKE}; --dg-real:{REAL}; --dg-warn:{WARN};
  --dg-display:'Chakra Petch', 'Instrument Sans', sans-serif;
  --dg-body:'Instrument Sans', system-ui, sans-serif;
}}
body, gradio-app, .gradio-container {{ background:var(--dg-ink) !important; }}
.gradio-container {{ max-width:1160px !important; margin:0 auto !important; padding:0 20px !important;
  font-family:var(--dg-body) !important; color:var(--dg-text); }}
.gradio-container :focus-visible {{ outline:2px solid var(--dg-scan) !important; outline-offset:2px; }}
footer {{ opacity:.55; }}
.dg-ic {{ flex:none; vertical-align:middle; }}
.dg-num {{ font-family:var(--dg-display); font-variant-numeric:tabular-nums; letter-spacing:.01em; }}

/* Hero ------------------------------------------------------------------- */
.dg-hero {{ display:grid; grid-template-columns:minmax(0,1.25fr) minmax(0,1fr); gap:32px; align-items:center;
  padding:40px 4px 28px; border-bottom:1px solid var(--dg-line); }}
.dg-brand {{ display:flex; align-items:center; gap:12px; color:var(--dg-scan); }}
.dg-brand h1 {{ font-family:var(--dg-display) !important; font-weight:700 !important; font-size:3.1rem !important;
  line-height:1 !important; margin:0 !important; color:var(--dg-text) !important; letter-spacing:-.01em; }}
.dg-lede {{ font-size:1.12rem; line-height:1.6; color:var(--dg-text); max-width:34em; margin:18px 0 22px; }}
.dg-status {{ display:inline-flex; align-items:center; gap:8px; font-size:.86rem; color:var(--dg-muted);
  border:1px solid var(--dg-line); border-radius:999px; padding:5px 12px 5px 10px; }}
.dg-status i {{ width:8px; height:8px; border-radius:50%; background:var(--dg-real);
  box-shadow:0 0 0 3px rgba(61,220,151,.18); }}
.dg-status.off i {{ background:var(--dg-fake); box-shadow:0 0 0 3px rgba(255,93,93,.18); }}
.dg-calib {{ margin-top:26px; display:grid; grid-template-columns:auto auto auto; gap:6px 28px; width:max-content;
  max-width:100%; }}
.dg-calib div {{ display:flex; flex-direction:column; }}
.dg-calib b {{ font-family:var(--dg-display); font-size:1.5rem; font-weight:600; color:var(--dg-text); }}
.dg-calib span {{ font-size:.82rem; color:var(--dg-muted); line-height:1.35; }}
.dg-mesh-wrap {{ position:relative; justify-self:center; width:100%; max-width:340px; }}
.dg-mesh {{ width:100%; height:auto; display:block; }}
.dg-mesh-lines path {{ fill:none; stroke:var(--dg-scan); stroke-opacity:.22; stroke-width:1; }}
.dg-mesh-corners path {{ fill:none; stroke:var(--dg-muted); stroke-width:1.5; }}
.dg-mesh-dots circle {{ fill:var(--dg-scan); opacity:.55; animation:dg-twinkle 3s ease-in-out infinite; }}
.dg-beam {{ animation:dg-sweep 4.2s cubic-bezier(.45,0,.55,1) infinite; }}
.dg-mesh-cap {{ text-align:center; font-size:.82rem; color:var(--dg-muted); margin-top:6px; }}
@keyframes dg-sweep {{ 0% {{ transform:translateY(0); }} 100% {{ transform:translateY(290px); }} }}
@keyframes dg-twinkle {{ 0%,100% {{ opacity:.35; }} 50% {{ opacity:.95; }} }}

/* Workbench -------------------------------------------------------------- */
#dg-bench {{ gap:24px !important; margin-top:28px; align-items:flex-start; }}
.dg-step {{ display:flex; align-items:center; gap:10px; margin:0 0 12px; }}
.dg-step em {{ font-style:normal; font-family:var(--dg-display); font-weight:600; width:28px; height:28px;
  border-radius:8px; display:grid; place-items:center; background:rgba(92,225,208,.12); color:var(--dg-scan); }}
.dg-step h2 {{ font-family:var(--dg-display) !important; font-size:1.25rem !important; font-weight:600 !important;
  margin:0 !important; color:var(--dg-text) !important; }}
.dg-step p {{ margin:0 0 0 auto; color:var(--dg-muted); font-size:.85rem; }}
#dg-upload {{ border:1px dashed #33536A !important; border-radius:14px !important; background:var(--dg-panel) !important; }}
#dg-run {{ font-family:var(--dg-display) !important; font-weight:600 !important; font-size:1.05rem !important;
  letter-spacing:.02em; min-height:52px; border-radius:12px !important; transition:transform .15s ease, box-shadow .2s ease; }}
#dg-run:hover {{ transform:translateY(-1px); box-shadow:0 8px 24px -10px rgba(92,225,208,.7); }}
#dg-run:active {{ transform:translateY(0); }}
.dg-tips {{ list-style:none; padding:0; margin:6px 0 0; display:grid; gap:9px; }}
.dg-tips li {{ display:flex; gap:10px; align-items:flex-start; color:var(--dg-muted); font-size:.88rem; line-height:1.5; }}
.dg-tips li svg {{ color:var(--dg-scan); margin-top:1px; }}

/* Report ----------------------------------------------------------------- */
.dg-report {{ background:var(--dg-panel); border:1px solid var(--dg-line); border-radius:16px; overflow:hidden; }}
.dg-sec {{ padding:20px 22px; border-top:1px solid var(--dg-line); }}
.dg-sec:first-child {{ border-top:0; }}
.dg-sec h3 {{ display:flex; align-items:center; gap:8px; font-family:var(--dg-body) !important; font-size:.95rem !important;
  font-weight:600 !important; color:var(--dg-text) !important; margin:0 0 4px !important; }}
.dg-sec h3 svg {{ color:var(--dg-muted); }}
.dg-note {{ color:var(--dg-muted); font-size:.82rem; line-height:1.5; margin:0 0 14px; max-width:60ch; }}

.dg-verdict {{ display:grid; grid-template-columns:132px 1fr; gap:22px; align-items:center; }}
.dg-ring {{ position:relative; width:132px; height:132px; }}
.dg-ring svg {{ transform:rotate(-90deg); }}
.dg-ring .trk {{ stroke:var(--dg-line); }}
.dg-ring .val {{ stroke:var(--tone); stroke-linecap:round; stroke-dasharray:var(--c);
  stroke-dashoffset:var(--off); animation:dg-ring 1.1s cubic-bezier(.2,.8,.2,1) both; }}
.dg-ring-in {{ position:absolute; inset:0; display:grid; place-content:center; text-align:center; }}
.dg-ring-in b {{ font-family:var(--dg-display); font-size:1.9rem; line-height:1; color:var(--dg-text); }}
.dg-ring-in span {{ font-size:.72rem; color:var(--dg-muted); margin-top:4px; }}
@keyframes dg-ring {{ from {{ stroke-dashoffset:var(--c); }} to {{ stroke-dashoffset:var(--off); }} }}
.dg-v-title {{ display:flex; align-items:center; gap:10px; color:var(--tone); }}
.dg-v-title strong {{ font-family:var(--dg-display); font-size:1.75rem; font-weight:700; line-height:1.1; }}
.dg-v-text {{ color:var(--dg-text); font-size:.95rem; line-height:1.55; margin:8px 0 12px; max-width:52ch; }}
.dg-tier {{ display:inline-flex; align-items:center; gap:6px; font-size:.8rem; font-weight:600; padding:4px 10px;
  border-radius:999px; background:color-mix(in srgb, var(--tone) 14%, transparent); color:var(--tone); }}
.dg-facts {{ display:flex; flex-wrap:wrap; gap:8px 20px; margin-top:14px; color:var(--dg-muted); font-size:.84rem; }}
.dg-facts span {{ display:inline-flex; align-items:center; gap:6px; }}
.dg-facts b {{ color:var(--dg-text); font-weight:600; }}

.dg-tl {{ position:relative; display:flex; align-items:flex-end; gap:4px; height:92px; padding-bottom:1px;
  border-bottom:1px solid var(--dg-line); }}
.dg-tl::after {{ content:""; position:absolute; left:0; right:0; bottom:50%; border-top:1px dashed #456477; }}
.dg-tl i {{ flex:1; border-radius:3px 3px 1px 1px; transform-origin:bottom; animation:dg-rise .55s cubic-bezier(.2,.8,.2,1) both; }}
@keyframes dg-rise {{ from {{ transform:scaleY(0); }} to {{ transform:scaleY(1); }} }}
.dg-tl-axis {{ display:flex; justify-content:space-between; color:var(--dg-muted); font-size:.78rem; margin-top:6px; }}
.dg-tl-key {{ display:flex; gap:16px; color:var(--dg-muted); font-size:.78rem; margin-top:8px; }}
.dg-tl-key span {{ display:inline-flex; align-items:center; gap:6px; }}
.dg-tl-key i {{ width:10px; height:10px; border-radius:2px; display:inline-block; }}

.dg-br {{ display:grid; grid-template-columns:34px 1fr 52px; gap:4px 12px; align-items:center; padding:9px 0; }}
.dg-br + .dg-br {{ border-top:1px solid rgba(34,57,74,.6); }}
.dg-br-ic {{ width:34px; height:34px; border-radius:9px; display:grid; place-items:center; grid-row:span 2;
  color:var(--bc); background:color-mix(in srgb, var(--bc) 13%, transparent); }}
.dg-br-name {{ font-size:.9rem; font-weight:600; color:var(--dg-text); }}
.dg-br-pct {{ text-align:right; grid-row:span 2; font-family:var(--dg-display); font-size:1rem; color:var(--dg-text); }}
.dg-br-bar {{ height:5px; border-radius:3px; background:var(--dg-line); overflow:hidden; }}
.dg-br-bar i {{ display:block; height:100%; width:var(--w); background:var(--bc); border-radius:3px;
  transform-origin:left; animation:dg-grow .9s cubic-bezier(.2,.8,.2,1) both; }}
.dg-br.top .dg-br-name::after {{ content:"highest weight"; margin-left:8px; font-size:.72rem; font-weight:500;
  color:var(--dg-scan); }}
@keyframes dg-grow {{ from {{ transform:scaleX(0); }} to {{ transform:scaleX(1); }} }}

.dg-callout {{ display:flex; gap:10px; align-items:flex-start; border-radius:10px; padding:11px 13px; font-size:.86rem;
  line-height:1.5; margin-top:10px; }}
.dg-callout.warn {{ background:rgba(242,181,68,.08); border:1px solid rgba(242,181,68,.28); color:#F6D58E; }}
.dg-callout.warn svg {{ color:var(--dg-warn); margin-top:1px; }}
.dg-callout.err {{ background:rgba(255,93,93,.07); border:1px solid rgba(255,93,93,.3); color:#FFC2C2; }}
.dg-callout.err svg {{ color:var(--dg-fake); margin-top:1px; }}

.dg-empty {{ padding:46px 26px; text-align:center; color:var(--dg-muted); }}
.dg-empty .dg-ghost {{ display:flex; gap:4px; align-items:flex-end; height:56px; max-width:260px; margin:0 auto 18px; opacity:.35; }}
.dg-empty .dg-ghost i {{ flex:1; background:var(--dg-line); border-radius:2px; }}
.dg-empty b {{ display:block; color:var(--dg-text); font-family:var(--dg-display); font-size:1.15rem; margin-bottom:6px; }}
.dg-empty p {{ margin:0 auto; max-width:38ch; font-size:.9rem; line-height:1.55; }}

/* Gallery ---------------------------------------------------------------- */
#dg-frames {{ margin-top:22px; border-radius:16px !important; }}
#dg-frames .caption-label {{ font-size:.72rem !important; }}
#dg-frames .grid-container {{ grid-template-columns:repeat(auto-fill, minmax(96px, 1fr)) !important; }}

/* About ------------------------------------------------------------------ */
.dg-about {{ margin-top:44px; padding-top:30px; border-top:1px solid var(--dg-line); }}
.dg-about h2 {{ font-family:var(--dg-display) !important; font-size:1.5rem !important; font-weight:600 !important;
  margin:0 0 6px !important; color:var(--dg-text) !important; }}
.dg-about > p {{ color:var(--dg-muted); max-width:62ch; line-height:1.6; margin:0 0 22px; }}
.dg-signals {{ display:grid; grid-template-columns:repeat(5, minmax(0,1fr)); gap:1px; background:var(--dg-line);
  border:1px solid var(--dg-line); border-radius:14px; overflow:hidden; }}
.dg-signal {{ background:var(--dg-panel); padding:16px 15px 18px; }}
.dg-signal .dg-br-ic {{ margin-bottom:12px; }}
.dg-signal b {{ display:block; color:var(--dg-text); font-size:.92rem; margin-bottom:5px; }}
.dg-signal p {{ margin:0 0 10px; color:var(--dg-muted); font-size:.82rem; line-height:1.5; }}
.dg-signal small {{ color:#6F8999; font-size:.76rem; }}
.dg-cols {{ display:grid; grid-template-columns:1.2fr 1fr; gap:36px; margin-top:30px; }}
.dg-cols h3 {{ font-family:var(--dg-body) !important; font-size:1rem !important; font-weight:600 !important;
  color:var(--dg-text) !important; margin:0 0 10px !important; }}
.dg-cols p, .dg-cols li {{ color:var(--dg-muted); font-size:.9rem; line-height:1.65; }}
.dg-cols ul {{ list-style:none !important; padding:0 !important; margin:0 !important; display:grid; gap:8px; }}
.dg-cols li {{ position:relative; padding-left:18px !important; margin:0 !important; }}
.dg-cols li::before {{ content:""; position:absolute; left:2px; top:.62em; width:6px; height:6px; border-radius:2px;
  background:var(--dg-scan); opacity:.7; }}
.dg-table {{ width:100%; border-collapse:collapse !important; font-size:.88rem; border:0 !important; margin:0 !important; }}
.dg-table tr, .dg-table tbody {{ background:transparent !important; border:0 !important; }}
.dg-table td {{ padding:9px 0 !important; border:0 !important; border-bottom:1px solid var(--dg-line) !important;
  background:transparent !important; color:var(--dg-muted); }}
.dg-table td:last-child {{ text-align:right; color:var(--dg-text); font-family:var(--dg-display); }}
.dg-foot {{ display:flex; justify-content:space-between; flex-wrap:wrap; gap:10px; color:#6F8999; font-size:.82rem;
  padding:28px 0 8px; margin-top:34px; border-top:1px solid var(--dg-line); }}

/* Responsive + motion ---------------------------------------------------- */
@media (max-width:900px) {{
  .dg-hero {{ grid-template-columns:1fr; padding-top:26px; }}
  .dg-mesh-wrap {{ max-width:240px; order:-1; }}
  .dg-signals {{ grid-template-columns:1fr 1fr; }}
  .dg-cols {{ grid-template-columns:1fr; }}
}}
@media (max-width:560px) {{
  .gradio-container {{ padding:0 14px !important; }}
  .dg-brand h1 {{ font-size:2.4rem !important; }}
  .dg-verdict {{ grid-template-columns:1fr; justify-items:start; }}
  .dg-calib {{ grid-template-columns:1fr 1fr; }}
  .dg-signals {{ grid-template-columns:1fr; }}
}}
@media (prefers-reduced-motion: reduce) {{
  .dg-beam, .dg-mesh-dots circle, .dg-ring .val, .dg-tl i, .dg-br-bar i {{ animation:none !important; }}
  .dg-beam {{ display:none; }}
}}
"""

FORCE_DARK = "() => { document.body.classList.add('dark'); }"


# ── HTML builders ─────────────────────────────────────────────────────────────
def hero_html(model_ok: bool, on_gpu: bool):
    status = ('<span class="dg-status"><i></i>Model ready · running on ZeroGPU</span>' if model_ok and on_gpu else
              '<span class="dg-status"><i></i>Model ready · running on CPU</span>' if model_ok else
              '<span class="dg-status off"><i></i>Model failed to load</span>')
    return f"""
<header class="dg-hero">
  <div>
    <div class="dg-brand">{icon("shield", 40)}<h1>DeepGuard</h1></div>
    <p class="dg-lede">Upload a short video of a face. DeepGuard reads five independent forensic signals,
      from how the skin looks to the camera's own sensor noise, and tells you whether the clip looks manipulated.</p>
    {status}
    <div class="dg-calib">
      <div><b>87.8%</b><span>AUC on held-out<br>FaceForensics++ videos</span></div>
      <div><b>72.7%</b><span>AUC on Celeb-DF v2,<br>never seen in training</span></div>
      <div><b>22.2M</b><span>parameters across<br>five branches</span></div>
    </div>
  </div>
  <div class="dg-mesh-wrap">{MESH_SVG}<div class="dg-mesh-cap">Tracks 468 facial landmarks per frame</div></div>
</header>"""


def step_html(n, title, aside=""):
    aside_html = f"<p>{aside}</p>" if aside else ""
    return f'<div class="dg-step"><em>{n}</em><h2>{title}</h2>{aside_html}</div>'


TIPS_HTML = f"""
<ul class="dg-tips">
  <li>{icon("face", 16)}One face, clearly visible and mostly front-facing.</li>
  <li>{icon("clock", 16)}At least 3 seconds. Only the first 15 seconds are analysed.</li>
  <li>{icon("film", 16)}MP4, MOV, AVI or MKV, up to 100 MB. Uploads are deleted within 10 minutes.</li>
</ul>"""

EMPTY_REPORT = f"""
<div class="dg-report"><div class="dg-empty">
  <div class="dg-ghost">{''.join(f'<i style="height:{h}%"></i>' for h in (30,55,40,70,45,60,35,80,50,65,40,58))}</div>
  <b>Your report will appear here</b>
  <p>Add a video on the left and press <strong>Analyze video</strong>. Most clips take 10–30 seconds.</p>
</div></div>"""


def error_html(msg):
    return (f'<div class="dg-report"><div class="dg-sec"><div class="dg-callout err">{icon("cross", 18)}'
            f'<div><b>This video could not be analysed.</b><br>{html.escape(msg)}</div></div></div></div>')


def _tc(sec):
    return f"{int(sec // 60):02d}:{sec % 60:04.1f}"


def report_html(res, clip, elapsed, tta):
    prob = float(res["fake_probability"])
    is_fake = prob >= 0.5
    tone = FAKE if is_fake else REAL
    conf = max(prob, 1 - prob)
    if conf >= 0.85:
        tier = "High confidence"
    elif conf >= 0.65:
        tier = "Moderate confidence"
    else:
        tier = "Low confidence: treat as uncertain"

    if is_fake:
        title, vicon = "Likely manipulated", "alert"
        text = (f"The model gives this clip a <b>{prob*100:.1f}%</b> chance of being a deepfake, "
                f"averaged over {len(res['frame_probs'])} face frames.")
    else:
        title, vicon = "No manipulation found", "check"
        text = (f"The model gives this clip a <b>{prob*100:.1f}%</b> chance of being a deepfake, "
                f"so it looks authentic across {len(res['frame_probs'])} face frames.")

    r = 54
    c = 2 * math.pi * r
    off = c * (1 - prob)
    ring = f"""
<div class="dg-ring" style="--tone:{tone};--c:{c:.1f};--off:{off:.1f}">
  <svg width="132" height="132" viewBox="0 0 132 132">
    <circle class="trk" cx="66" cy="66" r="{r}" fill="none" stroke-width="10"/>
    <circle class="val" cx="66" cy="66" r="{r}" fill="none" stroke-width="10"/>
  </svg>
  <div class="dg-ring-in"><b>{prob*100:.0f}%</b><span>fake score</span></div>
</div>"""

    verdict = f"""
<div class="dg-sec" style="--tone:{tone}">
  <div class="dg-verdict">
    {ring}
    <div>
      <div class="dg-v-title">{icon(vicon, 26)}<strong>{title}</strong></div>
      <p class="dg-v-text">{text}</p>
      <span class="dg-tier">{tier} · {conf*100:.0f}%</span>
      <div class="dg-facts">
        <span>{icon("film", 15)}<b>{len(res['frame_probs'])}</b> frames scored</span>
        <span>{icon("clock", 15)}<b>{elapsed:.1f} s</b> total</span>
        <span>{icon("flip", 15)}Mirror check <b>{"on" if tta else "off"}</b></span>
      </div>
    </div>
  </div>
</div>"""

    fps = clip.fps or 25.0
    n = len(res["frame_probs"])
    bars = "".join(
        f'<i title="{_tc(fi / fps)} · score {p:.2f}" style="height:{max(6, p * 100):.0f}%;'
        f'background:{FAKE if p >= 0.5 else REAL};animation-delay:{250 + k * 35}ms"></i>'
        for k, (p, fi) in enumerate(zip(res["frame_probs"], clip.aligned_indices)))
    timeline = f"""
<div class="dg-sec">
  <h3>{icon("film", 17)}Frame by frame</h3>
  <p class="dg-note">Fake score for each of the {n} aligned face frames, taken from the steadiest
    part of the clip. The dashed line is the 0.5 decision threshold.</p>
  <div class="dg-tl">{bars}</div>
  <div class="dg-tl-axis dg-num"><span>{_tc(clip.aligned_indices[0] / fps)}</span><span>{_tc(clip.aligned_indices[-1] / fps)}</span></div>
  <div class="dg-tl-key"><span><i style="background:{FAKE}"></i>Looks manipulated</span><span><i style="background:{REAL}"></i>Looks authentic</span></div>
</div>"""

    gates = res["gate_weights"]
    top = int(np.argmax(gates))
    rows = ""
    for i, b in enumerate(BRANCHES):
        w = float(gates[i]) * 100
        rows += f"""
  <div class="dg-br{' top' if i == top else ''}" style="--bc:{b['color']};--w:{max(1.5, w):.1f}%">
    <div class="dg-br-ic">{icon(b['icon'], 18)}</div>
    <div class="dg-br-name">{b['name']}</div>
    <div class="dg-br-pct">{w:.0f}%</div>
    <div class="dg-br-bar"><i style="animation-delay:{300 + i * 90}ms"></i></div>
  </div>"""
    branches = f"""
<div class="dg-sec">
  <h3>{icon("report", 17)}What the model leaned on</h3>
  <p class="dg-note">How much weight the fusion layer gave each signal for this clip. A high weight
    shows where the model looked, not proof that the signal found an artifact.</p>
  {rows}
</div>"""

    warn = "".join(f'<div class="dg-callout warn">{icon("alert", 17)}<div>{html.escape(w)}</div></div>'
                   for w in clip.warnings)
    caveat = f"""
<div class="dg-sec">
  {warn}
  <div class="dg-callout warn">{icon("alert", 17)}<div>Trained on face-swap and face-reenactment fakes from
  FaceForensics++ (2016–2020). It has not seen diffusion-generated or lip-sync videos, so modern fakes can be
  missed. A research prototype: never use it as the only evidence.</div></div>
</div>"""

    return f'<div class="dg-report">{verdict}{timeline}{branches}{caveat}</div>'


def about_html():
    signals = "".join(f"""
  <div class="dg-signal" style="--bc:{b['color']}">
    <div class="dg-br-ic">{icon(b['icon'], 18)}</div>
    <b>{b['name']}</b><p>{b['what']}</p><small>{b['net']}</small>
  </div>""" for b in BRANCHES)
    return f"""
<section class="dg-about">
  <h2>How DeepGuard decides</h2>
  <p>A deepfake can look perfect and still leave traces in places the eye never checks. DeepGuard looks
    at five of them at once, and a learned attention gate decides how much to trust each one for the clip in front of it.</p>
  <div class="dg-signals">{signals}</div>
  <div class="dg-cols">
    <div>
      <h3>What happens to your video</h3>
      <ul>
        <li>MediaPipe tracks the main face and finds the steadiest 20-frame stretch.</li>
        <li>Those frames are eye-aligned and cropped; 64 untouched crops (about 5 seconds) are kept for
          sensor-noise and pulse analysis.</li>
        <li>All five branches score the clip in one pass on a GPU, and the frame scores are averaged.</li>
        <li>The preprocessing is the same code that produced the training data.</li>
      </ul>
    </div>
    <div>
      <h3>How it was tested</h3>
      <table class="dg-table">
        <tr><td>Training data</td><td>FaceForensics++ C23</td></tr>
        <tr><td>Split</td><td>By source video, 0% overlap</td></tr>
        <tr><td>Validation AUC</td><td>91.58%</td></tr>
        <tr><td>Held-out test AUC</td><td>87.81%</td></tr>
        <tr><td>Celeb-DF v2 test AUC</td><td>72.72%</td></tr>
      </table>
    </div>
  </div>
  <div class="dg-foot">
    <span>Built by Mayank and Saloni Nayyar, supervised by Ms. Jyoti Kumari, Keshav Mahavidyalaya, University of Delhi.</span>
    <span>Presented at ICET-AICS 2026</span>
  </div>
</section>"""
