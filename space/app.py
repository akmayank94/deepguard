"""
app.py — DeepGuard on Hugging Face Spaces (Gradio + ZeroGPU)

CPU: face tracking, alignment and forensic feature extraction (deepguard/pipeline.py)
GPU: one batched forward pass through the five-branch detector (ZeroGPU, a few seconds)
"""

import os
import time

try:
    import spaces                      # Hugging Face ZeroGPU: import before torch
    GPU = spaces.GPU
except ImportError:                    # running locally without the package
    def GPU(*args, **kwargs):
        return (lambda f: f) if not (len(args) == 1 and callable(args[0])) else args[0]

import gradio as gr
import numpy as np
import torch

from deepguard.inference import load_checkpoint, build_tensors, forward_clip
from deepguard.pipeline import prepare_clip, PipelineError
from deepguard import ui

# ── Model ─────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.environ.get("DEEPGUARD_MODEL", "model.pt")
try:
    MODEL, MODEL_META = load_checkpoint(MODEL_PATH, DEVICE)
    MODEL_ERR = None
except Exception as e:                 # surfaced in the UI instead of crashing
    MODEL, MODEL_META, MODEL_ERR = None, {}, f"{type(e).__name__}: {e}"
    print("MODEL LOAD FAILED:", MODEL_ERR)


@GPU(duration=20)
def gpu_forward(tensors, use_tta):
    return forward_clip(MODEL, tensors, use_tta)


# ── Handler ───────────────────────────────────────────────────────────────────
def analyze(video_path, use_tta, show_frames, progress=gr.Progress()):
    hide = gr.update(value=None, visible=False)
    if MODEL is None:
        return ui.error_html(f"The model failed to load ({MODEL_ERR}). Please try again later."), hide
    if not video_path:
        return ui.error_html("No video was added. Drop a clip into the upload box first."), hide

    t_start = time.time()
    try:
        clip = prepare_clip(video_path, progress=progress)
        progress(0.93, desc="Scoring with the five-branch model")
        res = gpu_forward(build_tensors(clip), bool(use_tta))
    except PipelineError as e:
        return ui.error_html(str(e)), hide

    elapsed = time.time() - t_start
    report = ui.report_html(res, clip, elapsed, bool(use_tta))
    if not show_frames:
        return report, hide
    gallery = [(img, f"{'Fake' if p >= 0.5 else 'Real'} {p:.2f}")
               for img, p in zip(clip.aligned_rgb, res["frame_probs"])]
    return report, gr.update(value=gallery, visible=True)


# ── Layout ────────────────────────────────────────────────────────────────────
theme = gr.themes.Base(
    primary_hue=gr.themes.Color(c50="#E8FBF8", c100="#C9F5EE", c200="#9CEDE1", c300="#5CE1D0",
                                c400="#3FCDBB", c500="#2BB3A2", c600="#1F9283", c700="#187267",
                                c800="#135A52", c900="#0F4842", c950="#0A2F2B"),
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Instrument Sans"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    radius_size=gr.themes.sizes.radius_lg,
).set(
    body_background_fill=ui.INK, body_background_fill_dark=ui.INK,
    body_text_color=ui.TEXT, body_text_color_dark=ui.TEXT,
    body_text_color_subdued=ui.MUTED, body_text_color_subdued_dark=ui.MUTED,
    background_fill_primary=ui.PANEL, background_fill_primary_dark=ui.PANEL,
    background_fill_secondary=ui.PANEL_2, background_fill_secondary_dark=ui.PANEL_2,
    block_background_fill=ui.PANEL, block_background_fill_dark=ui.PANEL,
    block_border_color=ui.LINE, block_border_color_dark=ui.LINE,
    block_label_background_fill=ui.PANEL_2, block_label_background_fill_dark=ui.PANEL_2,
    block_label_text_color=ui.MUTED, block_label_text_color_dark=ui.MUTED,
    block_title_text_color=ui.TEXT, block_title_text_color_dark=ui.TEXT,
    border_color_primary=ui.LINE, border_color_primary_dark=ui.LINE,
    input_background_fill=ui.PANEL_2, input_background_fill_dark=ui.PANEL_2,
    checkbox_background_color=ui.PANEL_2, checkbox_background_color_dark=ui.PANEL_2,
    checkbox_background_color_selected=ui.SCAN, checkbox_background_color_selected_dark=ui.SCAN,
    checkbox_border_color=ui.LINE, checkbox_border_color_dark=ui.LINE,
    checkbox_label_background_fill=ui.PANEL, checkbox_label_background_fill_dark=ui.PANEL,
    checkbox_label_text_color=ui.TEXT, checkbox_label_text_color_dark=ui.TEXT,
    button_primary_background_fill=ui.SCAN, button_primary_background_fill_dark=ui.SCAN,
    button_primary_background_fill_hover="#7BE9DB", button_primary_background_fill_hover_dark="#7BE9DB",
    button_primary_text_color=ui.INK, button_primary_text_color_dark=ui.INK,
    color_accent=ui.SCAN, color_accent_soft="#173A3A", color_accent_soft_dark="#173A3A",
    loader_color=ui.SCAN, loader_color_dark=ui.SCAN,
    shadow_drop="none", shadow_drop_lg="none", block_shadow="none", block_shadow_dark="none",
)

with gr.Blocks(title="DeepGuard · deepfake video detector", theme=theme, css=ui.CSS,
               js=ui.FORCE_DARK, delete_cache=(600, 600)) as demo:
    gr.HTML(ui.hero_html(MODEL is not None, DEVICE == "cuda"))
    with gr.Row(equal_height=False, elem_id="dg-bench"):
        with gr.Column(scale=5, min_width=300):
            gr.HTML(ui.step_html(1, "Add a video"))
            video = gr.Video(label="Video", show_label=False, sources=["upload"], height=280,
                             elem_id="dg-upload")
            with gr.Row():
                tta = gr.Checkbox(value=True, label="Mirror check (scores flipped frames too)")
                show = gr.Checkbox(value=True, label="Show the face frames used")
            btn = gr.Button("Analyze video", variant="primary", elem_id="dg-run")
            gr.HTML(ui.TIPS_HTML)
        with gr.Column(scale=7, min_width=340):
            gr.HTML(ui.step_html(2, "Read the report"))
            report = gr.HTML(ui.EMPTY_REPORT)
    gallery = gr.Gallery(label="Aligned face frames the model scored", columns=10, height=250,
                         object_fit="cover", visible=False, allow_preview=True, elem_id="dg-frames")
    gr.HTML(ui.about_html())

    btn.click(analyze, inputs=[video, tta, show], outputs=[report, gallery], concurrency_limit=2)

if __name__ == "__main__":
    demo.queue(max_size=20).launch(max_file_size="100mb", ssr_mode=False)
