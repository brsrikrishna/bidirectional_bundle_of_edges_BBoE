"""
Build a single horizontal montage of the clean, text-free path-found panels
(produced by render_bboe_stills.py), with a title above each panel and one
shared horizontal legend underneath the whole row.

Run order:
    python3 trial_scripts/render_bboe_stills.py   # (re)generate the 8 panels
    python3 trial_scripts/create_bboe_montage.py  # compose this montage

Output: media/bboe_path_found_montage.png
"""
import sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]   # repository root
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw, ImageFont, ImageChops

PANEL_DIR = str(ROOT / "media/montage_panels")

# (panel file, title) in the requested left-to-right order.
# Note: "Unicycle" dropped from the first four titles per request.
ITEMS = [
    ("unicycle_easy.png",      "Easy"),
    ("unicycle_medium.png",    "Medium"),
    ("unicycle_hard.png",      "Hard"),
    ("unicycle_very_hard.png", "Very Hard"),
    ("unicycle.png",           "Unicycle"),
    ("cwt.png",                "CWT"),
    ("treaded.png",            "Treaded"),
    ("quadrotor.png",          "Quadrotor"),
]

# Panels whose canvas carries large invisible margins (the 3D plot) get cropped
# to their content so they fill the panel like the 2D envs. The 2D panels are
# NOT trimmed — their white is legitimate empty environment.
TRIM = {"quadrotor.png"}

OUT_PATH   = str(ROOT / "media/bboe_path_found_montage.png")
TARGET_H   = 500          # common panel height
TITLE_H    = 64           # title strip above each panel
PAD        = 18           # gap between panels
MARGIN     = 18           # outer border
LEG_GAP    = 14           # gap between panel row and legend
FG         = (0, 0, 0)
BG         = (255, 255, 255)
BORDER     = (210, 210, 210)
FONT_PATH  = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_MAX   = 42

# unified palette (must match render_bboe_stills.py)
C_FORWARD, C_REVERSE = 'steelblue', 'tomato'
START_FACE, GOAL_FACE = '#555555', 'lime'


def autocrop(im, pad=24):
    """Crop white margins to the content bounding box, keeping a uniform pad."""
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bbox = ImageChops.difference(im, bg).getbbox()
    if bbox is None:
        return im
    x0 = max(0, bbox[0] - pad)
    y0 = max(0, bbox[1] - pad)
    x1 = min(im.size[0], bbox[2] + pad)
    y1 = min(im.size[1], bbox[3] + pad)
    return im.crop((x0, y0, x1, y1))


def fit_font_size(titles, panel_widths):
    for size in range(FONT_MAX, 9, -1):
        font = ImageFont.truetype(FONT_PATH, size)
        if all(font.getsize(t)[0] <= w - 16 for t, w in zip(titles, panel_widths)):
            return font
    return ImageFont.truetype(FONT_PATH, 10)


def make_legend(width_px):
    """Render a horizontal 4-item legend strip to a PIL image."""
    handles = [
        Line2D([], [], marker='o', color='none', markerfacecolor=START_FACE,
               markeredgecolor='black', markersize=15, label='Start'),
        Line2D([], [], marker='o', color='none', markerfacecolor=GOAL_FACE,
               markeredgecolor='black', markersize=15, label='Goal'),
        Line2D([], [], color=C_REVERSE, lw=4, label='Reverse tree'),
        Line2D([], [], color=C_FORWARD, lw=4, label='Forward tree'),
    ]
    fig = plt.figure(figsize=(12, 0.9), dpi=200)
    fig.legend(handles=handles, loc='center', ncol=4, frameon=False,
               fontsize=24, handlelength=2.2, handletextpad=0.7,
               columnspacing=3.2)
    tmp = f"{PANEL_DIR}/_legend.png"
    fig.savefig(tmp, dpi=200, bbox_inches='tight', pad_inches=0.05,
                facecolor='white')
    plt.close(fig)
    leg = Image.open(tmp).convert("RGB")
    # scale legend to ~55% of montage width
    target_w = int(width_px * 0.55)
    scale = target_w / leg.size[0]
    return leg.resize((target_w, int(leg.size[1] * scale)), Image.LANCZOS)


def main():
    # 1) load + height-normalise panels
    frames = []
    for fname, _ in ITEMS:
        im = Image.open(os.path.join(PANEL_DIR, fname)).convert("RGB")
        if fname in TRIM:
            im = autocrop(im)
        w, h = im.size
        new_w = round(w * TARGET_H / h)
        frames.append(im.resize((new_w, TARGET_H), Image.LANCZOS))
    panel_widths = [f.size[0] for f in frames]

    # 2) uniform title font
    font = fit_font_size([t for _, t in ITEMS], panel_widths)

    # 3) build each panel = title strip + framed image
    panel_h = TITLE_H + TARGET_H
    panels = []
    for fr, (_, title) in zip(frames, ITEMS):
        w = fr.size[0]
        panel = Image.new("RGB", (w, panel_h), BG)
        draw = ImageDraw.Draw(panel)
        tw, th = font.getsize(title)
        draw.text(((w - tw) // 2, (TITLE_H - th) // 2), title, fill=FG, font=font)
        panel.paste(fr, (0, TITLE_H))
        draw.rectangle([0, TITLE_H, w - 1, panel_h - 1], outline=BORDER, width=1)
        panels.append(panel)

    # 4) horizontal layout
    row_w = sum(panel_widths) + PAD * (len(panels) - 1)
    total_w = row_w + 2 * MARGIN

    legend = make_legend(total_w)

    total_h = MARGIN + panel_h + LEG_GAP + legend.size[1] + MARGIN
    montage = Image.new("RGB", (total_w, total_h), BG)
    x = MARGIN
    for p in panels:
        montage.paste(p, (x, MARGIN))
        x += p.size[0] + PAD
    # centred legend below the row
    lx = (total_w - legend.size[0]) // 2
    ly = MARGIN + panel_h + LEG_GAP
    montage.paste(legend, (lx, ly))

    montage.save(OUT_PATH)
    print(f"Saved montage -> {OUT_PATH}  ({montage.size[0]}x{montage.size[1]}, "
          f"title font {font.size})")


if __name__ == "__main__":
    main()
