"""Build the OMARI seminar deck (editable .pptx) and the presenter guide from one slide specification.

Every number shown on a slide is read from the finalized website bundle
(website/public/experiments/, schema 1.1) at build time; figures come from
presentation/assets/ (run make_figures.py first).

    presentation/.venv/Scripts/python presentation/src/build_deck.py [--url https://…]

--url sets the participant-facing workshop URL and generates its QR code.
Without it, every activity slide shows an editable placeholder instead.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "website" / "public" / "experiments"
PRES = ROOT / "presentation"
FIG = PRES / "assets" / "figures"
EQ = PRES / "assets" / "equations"
QR = PRES / "assets" / "qr"
OUT_PPTX = PRES / "OMARI_From_Pixels_to_Predictions.pptx"
OUT_GUIDE = PRES / "PRESENTER_GUIDE.md"

TITLE = "From Pixels to Predictions"
SUBTITLE = "Developing AI Models for Medical Image Interpretation"
AUTHORS = "Kevin Dick  ·  Youssef Megahed"
EVENT = "Ottawa Medical AI Research Institute (OMARI) Seminar Series"
URL_PLACEHOLDER = "[Workshop website URL: add before the session]"

# ---------------------------------------------------------------- design tokens (shared with website/src/styles/global.css)
PAPER, RAISED, INK, INK2, INK3 = "F6F7F8", "FFFFFF", "18232B", "3D4B55", "6A7882"
RULE, RULE2, WELL, WELL_INK = "D6DDE1", "B5C0C6", "0D1317", "C9D3D8"
ACCENT = "1F4F66"            # activity / website colour (deepened --p-live)
M = {"densenet161": "0072B2", "vit_b16": "E69F00", "usf_mae": "009E73"}
M_TEXT = {"densenet161": "0072B2", "vit_b16": "9A6A00", "usf_mae": "00775A"}
NAME = {"densenet161": "DenseNet-161", "vit_b16": "ViT-B/16", "usf_mae": "USF-MAE"}
BADGE = {"model": ("MODEL OUTPUT", "1D7A4E"), "data": ("DATASET IMAGE", "4A5A8A"), "meta": ("DATASET METADATA", "4A5A8A"),
         "illus": ("ILLUSTRATIVE DIAGRAM", "8A6D1F"), "web": ("WEBSITE ACTIVITY", ACCENT)}
SERIF, SANS, MONO = "Georgia", "Segoe UI", "Consolas"
SHORT = ["Abdomen", "Brain", "Femur", "Thorax", "Cervix", "Other"]

CHAPTERS = [  # id, short label, long label, start, end
    ("1", "Opening", "Chapter 1 · Opening challenge", 0, 10),
    ("2", "Data", "Chapter 2 · Understanding ultrasound data", 10, 25),
    ("3", "Models", "Chapter 3 · CNNs, transformers and USF-MAE", 25, 50),
    ("4", "Evaluation", "Chapter 4 · Training and evaluation", 50, 70),
    ("B", "Break", "Break", 70, 80),
    ("5", "Explain", "Chapter 5 · Explainability", 80, 105),
    ("6", "Detective", "Chapter 6 · Model detective", 105, 115),
    ("7", "Closing", "Chapter 7 · Closing discussion", 115, 120),
]
CH = {c[0]: c for c in CHAPTERS}
SW, SH = 13.333, 7.5


def rgb(h: str) -> RGBColor:
    return RGBColor.from_string(h)


# ---------------------------------------------------------------- data

def load(name):
    return json.loads((BUNDLE / name).read_text(encoding="utf-8"))


EV, LE, DS, FA, MAN = (load(n) for n in ("evaluation.json", "label_efficiency.json", "dataset_stats.json", "faithfulness.json", "manifest.json"))
FP = json.loads((PRES / "assets" / "figure_provenance.json").read_text(encoding="utf-8"))
CLASSES = DS["classes"]


def mean_sd(vals):
    v = [x for x in vals if x is not None]
    mu = sum(v) / len(v)
    sd = (sum((x - mu) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else float("nan")
    return mu, sd, len(v)


def le_at(model, metric, frac):
    i = LE["fractions"].index(frac)
    return mean_sd(LE["per_model"][model][metric][i])


def f3(x):
    return f"{x:.3f}"


def f4(x):
    return f"{x:.4f}"


def n(x):
    return f"{x:,}"


PM = EV["per_model"]
SEED = EV["seed_summary"]["usf_mae"]
OPEN = next(s for s in MAN["samples"] if "opening" in (s.get("tags") or []))
TOT = {c: sum(DS["class_counts"][s][i] for s in ("train", "val", "test")) for i, c in enumerate(CLASSES)}
SPL = DS["split"]
MBS = DS["machines_by_split"]
ND = DS["near_duplicates"]

# ---------------------------------------------------------------- low-level drawing helpers


def set_bg(slide, color):
    f = slide.background.fill
    f.solid()
    f.fore_color.rgb = rgb(color)


def _runs(p, text, size, color, bold, font, italic=False):
    """Plain text with **bold** spans."""
    parts = re.split(r"(\*\*[^*]+\*\*)", text)
    for part in parts:
        if not part:
            continue
        r = p.add_run()
        b = part.startswith("**")
        r.text = part[2:-2] if b else part
        r.font.size = Pt(size)
        r.font.name = font
        r.font.bold = bold or b
        r.font.italic = italic
        r.font.color.rgb = rgb(color)


def text(slide, x, y, w, h, content, size=16, color=INK, bold=False, font=SANS, align="l", anchor="t",
         spacing=1.08, italic=False, space_after=0, bullets=False, margin=0.0):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, Inches(margin))
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
    paras = content if isinstance(content, list) else [content]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[align]
        p.line_spacing = spacing
        p.space_after = Pt(space_after)
        if bullets:
            _bullet(p)
        _runs(p, para, size, color, bold, font, italic)
    return tb


def _bullet(p, char="•"):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(Emu(Inches(0.24))))
    pPr.set("indent", str(-Emu(Inches(0.24))))
    bu = etree.SubElement(pPr, qn("a:buChar"))
    bu.set("char", char)


def box(slide, x, y, w, h, fill=None, line=None, lw=0.75, shape=MSO_SHAPE.RECTANGLE, radius=None, dash=False):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill)
    else:
        s.fill.background()
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(lw)
        if dash:
            s.line.dash_style = 4  # MSO_LINE.DASH
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    s.text_frame.margin_left = s.text_frame.margin_right = Inches(0.08)
    return s


def label_in(shape, content, size=14, color=INK, bold=False, font=SANS, align="c", anchor="m", spacing=1.0):
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
    paras = content if isinstance(content, list) else [content]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[align]
        p.line_spacing = spacing
        _runs(p, para, size, color, bold, font)
    return shape


def arrow(slide, x1, y1, x2, y2, color=INK3, width=1.5, head=True):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(width)
    if head:
        ln = c.line._get_or_add_ln()
        te = etree.SubElement(ln, qn("a:tailEnd"))
        te.set("type", "triangle")
        te.set("w", "med")
        te.set("len", "med")
    return c


def gradient_fade(slide, x, y, w, h, color, angle=0):
    """Rectangle fading from opaque `color` (left) to transparent (right)."""
    s = box(slide, x, y, w, h, fill=color)
    spPr = s._element.spPr
    for tag in ("a:solidFill", "a:noFill", "a:gradFill"):
        for el in spPr.findall(qn(tag)):
            spPr.remove(el)
    g = etree.SubElement(spPr, qn("a:gradFill"))
    g.set("rotWithShape", "1")
    lst = etree.SubElement(g, qn("a:gsLst"))
    for pos, alpha in ((0, 100000), (55000, 70000), (100000, 0)):
        gs = etree.SubElement(lst, qn("a:gs"))
        gs.set("pos", str(pos))
        c = etree.SubElement(gs, qn("a:srgbClr"))
        c.set("val", color)
        a = etree.SubElement(c, qn("a:alpha"))
        a.set("val", str(alpha))
    lin = etree.SubElement(g, qn("a:lin"))
    lin.set("ang", str(angle * 60000))
    lin.set("scaled", "0")
    # gradFill must come right after prstGeom, before a:ln
    prst = spPr.find(qn("a:prstGeom"))
    spPr.remove(g)
    prst.addnext(g)
    return s


def picture(slide, path, x, y, w=None, h=None, fit="contain"):
    """Place an image inside the (x, y, w, h) box, preserving aspect ratio."""
    path = Path(path)
    iw, ih = Image.open(path).size
    if w is not None and h is not None and fit == "cover":
        pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))
        ra, rb = iw / ih, w / h
        if ra > rb:  # image wider: crop left/right
            c = (1 - rb / ra) / 2
            pic.crop_left = pic.crop_right = c
        else:
            c = (1 - ra / rb) / 2
            pic.crop_top = pic.crop_bottom = c
        return pic, (x, y, w, h)
    if w is not None and h is not None:
        r = min(w / iw, h / ih)
        pw, ph = iw * r, ih * r
        px, py = x + (w - pw) / 2, y + (h - ph) / 2
    elif w is not None:
        pw, ph, px, py = w, w * ih / iw, x, y
    else:
        pw, ph, px, py = h * iw / ih, h, x, y
    return slide.shapes.add_picture(str(path), Inches(px), Inches(py), Inches(pw), Inches(ph)), (px, py, pw, ph)


def equation(slide, name, x, y, h, max_w=None):
    iw, ih = Image.open(EQ / f"{name}.png").size
    if max_w and h * iw / ih > max_w:
        h = max_w * ih / iw
    return picture(slide, EQ / f"{name}.png", x, y, h=h)


def badge(slide, kind, x, y, extra=None, dark=False):
    lab, col = BADGE[kind]
    t = lab + (f" · {extra}" if extra else "")
    w = 0.13 + 0.075 * len(t)
    s = box(slide, x, y, w, 0.26, fill=col, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.5)
    label_in(s, t, size=8.5, color="FFFFFF", bold=True)
    return w


def caption(slide, x, y, w, content, size=11, color=INK3, h=0.5):
    return text(slide, x, y, w, h, content, size=size, color=color)


def table(slide, x, y, w, h, rows, col_w=None, size=12, header_fill=INK, header_color="FFFFFF", fills=None, colors=None,
          bold_cells=None, align=None, row_h=None, font=SANS, tight=False):
    nr, nc = len(rows), len(rows[0])
    gt = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(h))
    tbl = gt.table
    tblPr = tbl._tbl.tblPr
    for attr in ("bandRow", "firstRow"):
        tblPr.set(attr, "0")
    style = tblPr.find(qn("a:tableStyleId"))
    if style is not None:
        style.text = "{5940675A-B579-460E-94D1-54222C63F5DA}"  # 'No Style, Table Grid'
    if col_w:
        for i, cw in enumerate(col_w):
            tbl.columns[i].width = Inches(cw)
    for r in range(nr):
        if row_h:
            tbl.rows[r].height = Inches(row_h)
        for c in range(nc):
            cell = tbl.cell(r, c)
            cell.margin_left = cell.margin_right = Inches(0.07)
            cell.margin_top = cell.margin_bottom = Inches(0.0 if tight else 0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            val = rows[r][c]
            fill = header_fill if r == 0 else (fills[r][c] if fills and fills[r][c] else RAISED)
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(fill)
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            a = (align[c] if align else ("l" if c == 0 else "c"))
            p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[a]
            col = header_color if r == 0 else (colors[r][c] if colors and colors[r][c] else INK)
            b = r == 0 or bool(bold_cells and bold_cells[r][c])
            _runs(p, str(val), size, col, b, font)
            _cell_border(cell, RULE)
    return tbl


def _cell_border(cell, color):
    tcPr = cell._tc.get_or_add_tcPr()
    for edge in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        ln = etree.SubElement(tcPr, qn(edge))
        ln.set("w", str(Emu(Pt(0.75))))
        sf = etree.SubElement(ln, qn("a:solidFill"))
        c = etree.SubElement(sf, qn("a:srgbClr"))
        c.set("val", color)
    # schema order: borders must precede the fill element
    fill = tcPr.find(qn("a:solidFill"))
    if fill is not None:
        tcPr.remove(fill)
        tcPr.append(fill)


def mix(c1, c2, t):
    a = [int(c1[i:i + 2], 16) for i in (0, 2, 4)]
    b = [int(c2[i:i + 2], 16) for i in (0, 2, 4)]
    return "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(a, b))


# ---------------------------------------------------------------- slide chrome

@dataclass
class Notes:
    message: str
    points: list[str]
    plain: str
    website: str
    ask: list[str]
    caveats: list[str] = field(default_factory=list)
    cue: str | None = None


@dataclass
class SlideSpec:
    chapter: str
    title: str
    minutes: float
    notes: Notes
    kind: str = "content"   # content | activity | break | appendix
    activity: str | None = None


SPECS: list[SlideSpec] = []
STATE = {"url": None, "qr": None, "slides": []}


def chrome(slide, spec: SlideSpec, dark=False, idx=0):
    """Chapter kicker, timeline footer and slide number (drawn last, so always on top)."""
    fg = WELL_INK if dark else INK3
    ch = CH[spec.chapter]
    if spec.kind != "appendix":
        if spec.kind not in ("activity", "title"):
            text(slide, 0.6, 0.32, 8.5, 0.3, ch[2].upper(), size=10, color=fg, bold=True)
        x0, wtot, y = 0.6, SW - 1.2, SH - 0.36
        for c in CHAPTERS:
            w = wtot * (c[4] - c[3]) / 120
            on = c[0] == spec.chapter
            col = (WELL_INK if dark else INK) if on else ("33414A" if dark else RULE)
            box(slide, x0 + 0.02, y, w - 0.04, 0.05 if not on else 0.07, fill=col)
            x0 += w
    else:
        text(slide, 0.6, 0.32, 8, 0.3, "APPENDIX", size=10, color=INK3, bold=True)
    stamp = f"{ch[3]}–{ch[4]} min  ·  {idx}" if spec.kind != "appendix" else str(idx)
    text(slide, SW - 3.6, 0.32, 3.0, 0.3, stamp, size=10, color=fg, align="r")


def new_slide(prs, spec: SlideSpec, dark=False, bg=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s, bg or (WELL if dark else PAPER))
    SPECS.append(spec)
    STATE["slides"].append((s, spec, dark, len(SPECS)))
    s.notes_slide.notes_text_frame.text = render_notes(spec, len(SPECS))
    return s


def heading(slide, title, sub=None, dark=False, y=0.62, size=30, w=11.8):
    text(slide, 0.6, y, w, 0.75, title, size=size, color=("FFFFFF" if dark else INK), font=SERIF)
    if sub:
        text(slide, 0.6, y + 0.72, w, 0.45, sub, size=15.5, color=(WELL_INK if dark else INK2))


def render_notes(spec: SlideSpec, num: int) -> str:
    n_ = spec.notes
    ch = CH[spec.chapter]
    lines = [f"SLIDE {num} · {spec.title}",
             f"Time: {spec.minutes:g} min  (chapter window {ch[3]}–{ch[4]} min)" if spec.kind != "appendix" else "Time: appendix / reference only",
             "", "MAIN MESSAGE", n_.message, "", "TALKING POINTS"]
    lines += [f"• {p}" for p in n_.points]
    lines += ["", "FOR A MIXED AUDIENCE", n_.plain, "", "WEBSITE", n_.website]
    if n_.cue:
        lines += ["", "PRESENTER CUE", n_.cue]
    lines += ["", "ASK THE ROOM"] + [f"• {q}" for q in n_.ask]
    if n_.caveats:
        lines += ["", "SCIENTIFIC CAVEATS"] + [f"• {c}" for c in n_.caveats]
    return "\n".join(lines)


def url_text():
    return STATE["url"] or URL_PLACEHOLDER


def qr_block(slide, x, y, size=1.55, dark=False):
    if STATE["qr"]:
        picture(slide, STATE["qr"], x, y, size, size)
    else:
        s = box(slide, x, y, size, size, fill=("1A252C" if dark else RAISED), line=(WELL_INK if dark else RULE2), lw=1.25, dash=True)
        label_in(s, ["QR code", "added when the", "website URL is set"], size=9.5, color=(WELL_INK if dark else INK3))


# ---------------------------------------------------------------- slide builders

def activity_slide(prs, chapter, title, minutes, web_chapter, steps, question, discuss, notes: Notes):
    spec = SlideSpec(chapter, title, minutes, notes, kind="activity", activity=web_chapter)
    s = new_slide(prs, spec)
    box(s, 0, 0, 4.55, SH, fill=ACCENT)
    text(s, 0.6, 0.32, 3.8, 0.3, CH[chapter][2].upper(), size=10, color="BFD3DD", bold=True)  # re-draw kicker on band
    text(s, 0.6, 1.0, 3.6, 0.35, "EXPLORE ON THE WORKSHOP WEBSITE", size=11, color="BFD3DD", bold=True)
    text(s, 0.6, 1.45, 3.6, 1.0, web_chapter, size=34, color="FFFFFF", font=SERIF)
    text(s, 0.6, 2.45, 3.6, 1.3, title, size=17, color="FFFFFF")
    chip = box(s, 0.6, 3.85, 1.55, 0.42, fill="FFFFFF", shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.5)
    label_in(chip, f"{minutes:g} minutes", size=13, color=ACCENT, bold=True)
    qr_block(s, 0.6, 4.6, 1.55, dark=True)
    text(s, 2.3, 4.6, 2.05, 1.6, ["Open the companion site:", url_text()], size=10.5, color="DCE7EC")
    # right side
    text(s, 5.1, 0.95, 7.6, 0.35, "DO", size=11, color=ACCENT, bold=True)
    text(s, 5.1, 1.3, 7.6, 2.3, [f"{i + 1}.  {st}" for i, st in enumerate(steps)], size=17, color=INK, space_after=6, spacing=1.05)
    q = box(s, 5.1, 3.75, 7.6, 1.35, fill="E7EFF3", shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
    text(s, 5.35, 3.85, 7.2, 0.3, "INVESTIGATE", size=11, color=ACCENT, bold=True)
    text(s, 5.35, 4.15, 7.2, 0.9, question, size=18, color=INK, font=SERIF)
    text(s, 5.1, 5.35, 7.6, 0.3, "DISCUSS", size=11, color=ACCENT, bold=True)
    text(s, 5.1, 5.68, 7.6, 0.9, discuss, size=16, color=INK2)
    return s


def build(prs):
    # =========================================================== CHAPTER 1
    s = new_slide(prs, SlideSpec("1", "Title", 1, kind="title", notes=Notes(
        message="Welcome. Today we follow one question from raw ultrasound pixels to a model's prediction, and then ask whether we should trust it.",
        points=["Introduce both presenters by name only, as they prefer to be introduced.",
                "The talk and the website are designed to be used together: the slides carry the story, the website lets everyone test the claims on real model outputs.",
                "Everything shown as a result comes from saved experiments: three trained models on a public prenatal ultrasound dataset (FETAL_PLANES_DB)."],
        plain="We will build intuition first, then look at real results. No coding is needed; laptops or phones are optional but helpful.",
        website="Show the URL or QR code now so people can open the site for the Chapter 1 activity and keep it open for the rest of the session.",
        ask=["Who here has reviewed fetal ultrasound images? Who has trained an image model?"],
        caveats=["The background image is a FETAL_PLANES_DB test image (CC BY 4.0, Burgos-Artizzu et al. 2020)."])), dark=True)
    pic, g = picture(s, FIG / "opening.png", 6.55, 1.0, 6.78, 5.5)  # native 640 px: ~95 ppi, no upscaling blur
    gradient_fade(s, g[0] - 0.01, g[1], 2.2, g[3], WELL)
    text(s, 0.75, 1.55, 7.6, 0.4, "OMARI SEMINAR · INTERACTIVE WORKSHOP", size=12, color="8FB3C4", bold=True)
    text(s, 0.75, 2.05, 7.8, 1.0, TITLE, size=48, color="FFFFFF", font=SERIF)
    text(s, 0.75, 3.05, 6.8, 1.1, SUBTITLE, size=24, color=WELL_INK, font=SERIF, spacing=1.05)
    box(s, 0.78, 4.4, 0.7, 0.04, fill=M["usf_mae"])
    text(s, 0.75, 4.6, 7, 0.45, AUTHORS, size=19, color="FFFFFF")
    text(s, 0.75, 5.05, 7, 0.4, EVENT, size=13.5, color=WELL_INK)
    text(s, 0.75, 5.55, 7, 0.4, f"Companion website: {url_text()}", size=12, color="8FB3C4")
    text(s, 9.0, 6.85, 4.1, 0.3, "Image: FETAL_PLANES_DB test set, CC BY 4.0", size=9, color="8A99A2", align="r")

    s = new_slide(prs, SlideSpec("1", "Can AI recognise this ultrasound plane?", 3, Notes(
        message="Before any AI, everyone commits to an answer. Interpreting a single ultrasound frame is genuinely hard.",
        points=["Give the room 60 seconds to vote by show of hands for A–F. Do not reveal the answer: we come back to it in Chapter 7.",
                "Point out what makes it hard: speckle texture, partial anatomy, and on-screen scanner text.",
                "This is a real test-set image from the same split the models were evaluated on."],
        plain="Each image in the dataset was labelled by an expert as one of six standard views. We are asking the room to do what the models do.",
        website="Chapter 1 of the website shows this exact image with a guess-then-reveal interaction.",
        ask=["Which plane is it? How confident are you, 0–100%?", "What cue did you use: a structure, the shape, the texture?"],
        caveats=["Do not reveal the expert label or model predictions yet (they appear on slide 34 and on the website)."])), dark=True)
    heading(s, "Which plane is this?", "Commit to an answer before any model does.", dark=True)
    picture(s, FIG / "opening.png", 0.6, 1.95, 7.9, 4.85)
    for i, c in enumerate(CLASSES):
        y = 2.05 + i * 0.74
        k = box(s, 9.05, y, 0.55, 0.55, fill="1E2B33", line="3A4A54", shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.18)
        label_in(k, "ABCDEF"[i], size=16, color="FFFFFF", bold=True)
        text(s, 9.8, y, 3.3, 0.55, c, size=19, color="FFFFFF", anchor="m")
    badge(s, "data", 0.6, 6.88 - 0.05, "FETAL_PLANES_DB test image · CC BY 4.0", dark=True)

    s = new_slide(prs, SlideSpec("1", "Three models, one image", 3, Notes(
        message="We compare three ways of learning image representations, chosen to differ in architecture and in what they learned before seeing our labels.",
        points=["DenseNet-161: a convolutional network pretrained with labels on ImageNet (everyday photographs).",
                "ViT-B/16: a Vision Transformer, also pretrained with labels on ImageNet: same data source, different architecture.",
                "USF-MAE: the same transformer architecture as ViT-B/16, but pretrained without labels on about 370,000 ultrasound images by masked-image reconstruction. We use the 500-epoch pretrained checkpoint.",
                "All three were then fine-tuned on the same labelled FETAL_PLANES_DB training patients and tested on the same held-out patients."],
        plain="Think of pretraining as prior experience. Two models learned from photographs with labels; one learned from ultrasound itself without labels.",
        website="Chapter 1: reveal the three models' real predictions for the opening image and browse the 'models disagree' set.",
        ask=["Which kind of prior experience would you expect to help most for ultrasound, and why?"],
        caveats=["The models differ in more than one factor (architecture, pretraining data, pretraining objective, fine-tuning recipe); comparisons cannot isolate a single cause.",
                 "USF-MAE's unlabelled pretraining corpus included FETAL_PLANES_DB images (discussed in Chapter 3)."])))
    heading(s, "Three models, one image", "Same labels and same test patients; different ways of learning to see.")
    cards = [("densenet161", "Convolutional network", ["Local filters, stacked", "Pretrained **with labels** on ImageNet photographs", f"{PM_PARAMS['densenet161']} parameters"]),
             ("vit_b16", "Vision Transformer", ["Image as a sequence of patches", "Pretrained **with labels** on ImageNet photographs", f"{PM_PARAMS['vit_b16']} parameters"]),
             ("usf_mae", "Transformer, ultrasound-pretrained", ["Same architecture as ViT-B/16", "Pretrained **without labels** on ~370k ultrasound images (500 epochs)", f"{PM_PARAMS['usf_mae']} parameters"])]
    for i, (m, kind, lines) in enumerate(cards):
        x = 0.6 + i * 4.1
        box(s, x, 1.95, 3.85, 3.0, fill=RAISED, line=RULE)
        box(s, x, 1.95, 3.85, 0.09, fill=M[m])
        text(s, x + 0.25, 2.2, 3.4, 0.5, NAME[m], size=24, color=M_TEXT[m], font=SERIF, bold=True)
        text(s, x + 0.25, 2.72, 3.4, 0.35, kind, size=13.5, color=INK3, bold=True)
        text(s, x + 0.25, 3.15, 3.4, 1.7, lines, size=14.5, color=INK2, bullets=True, space_after=4)
    q = box(s, 0.6, 5.2, 12.1, 1.3, fill=INK, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
    text(s, 0.95, 5.3, 11.4, 0.3, "THE QUESTION FOR TODAY", size=10.5, color="8FB3C4", bold=True)
    text(s, 0.95, 5.6, 11.4, 0.85, "How do different approaches to learning image representations affect prediction, generalisation and explainability in prenatal ultrasound?",
         size=18.5, color="FFFFFF", font=SERIF)

    activity_slide(prs, "1", "Opening challenge: guess, then reveal", 3, "Chapter 1",
                   ["Open Chapter 1 and pick a plane for the image.", "Reveal the expert label and all three models' probabilities.", "Switch to the “models disagree” set and look at two more images."],
                   "When the three models disagree, is the mistake anatomically understandable?",
                   "Correct, confident, and correct for the right reason: which of these can you actually see from a prediction?",
                   Notes(message="Participants experience the gap between a prediction and a reason.",
                         points=["Keep this short (3 min); it is a warm-up.", "Invite one or two people to describe a disagreement case they found."],
                         plain="The website shows real probabilities saved from the evaluation run, not a live re-run.",
                         website="Chapter 1 › 'Which plane is this?' figure: answer buttons, Reveal, then the pool selector (workshop / models disagree / any model wrong).",
                         cue="Ask participants to open Chapter 1, make a guess, then press Reveal.",
                         ask=["Did anyone's guess differ from the expert label?", "Did confidence track correctness in the cases you looked at?"],
                         caveats=["Predictions are the saved test-set outputs behind the Chapter 4 metrics."]))

    # =========================================================== CHAPTER 2
    s = new_slide(prs, SlideSpec("2", "Why ultrasound is hard for humans and machines", 3, Notes(
        message="Ultrasound images are noisy, operator-dependent and scanner-dependent; a model has to cope with all of that.",
        points=["Speckle: the grainy texture is an interference pattern, not tissue structure.",
                "Shadowing and other artefacts hide anatomy behind bone and change with probe angle.",
                "Standard planes are defined by anatomy, but every acquisition differs in angle, depth, gain and gestational age.",
                f"The dataset comes from {len(DS['machines'])} scanner groups and several operators; on-screen text and calipers are part of the pixels."],
        plain="The six panels show one real example of each class. Notice how different 'Other' can look: it is a catch-all class.",
        website="Chapter 2 preprocessing lab lets you add speckle, change resolution and adjust brightness on these images.",
        ask=["Which of these would you find hardest to recognise? Why?"],
        caveats=["Examples were chosen as images all three models classify correctly, so they are clean, not typical, cases."])))
    heading(s, "Why ultrasound is hard", "Six standard classes, one real test image each (all correctly classified by all three models).")
    order = ["abdomen", "brain", "femur", "thorax", "cervix", "other"]
    for i, (c, cls) in enumerate(zip(order, CLASSES)):
        x, y = 0.6 + (i % 3) * 2.72, 1.95 + (i // 3) * 2.32
        box(s, x, y, 2.6, 1.85, fill=WELL)
        picture(s, FIG / f"class_{c}.png", x, y, 2.6, 1.85)
        text(s, x, y + 1.88, 2.6, 0.3, cls, size=12.5, color=INK, bold=True)
    items = [("Speckle", "grainy interference texture overlays every structure"),
             ("Artefacts", "acoustic shadowing, reverberation, on-screen text and calipers"),
             ("Acquisition", "probe angle, depth, gain and gestational age all vary"),
             ("Scanners", f"{len(DS['machines'])} machine groups with different image character")]
    for i, (h_, b_) in enumerate(items):
        y = 2.0 + i * 1.12
        box(s, 8.95, y + 0.05, 0.06, 0.8, fill=INK)
        text(s, 9.2, y, 3.6, 0.35, h_, size=16, color=INK, bold=True)
        text(s, 9.2, y + 0.35, 3.6, 0.7, b_, size=13, color=INK2)
    badge(s, "data", 0.6, 6.78, "FETAL_PLANES_DB · CC BY 4.0")

    s = new_slide(prs, SlideSpec("2", "FETAL_PLANES_DB at a glance", 4, Notes(
        message=f"{n(DS['total_images'])} images from {n(DS['total_patients'])} patients in six classes, with strong class imbalance.",
        points=[f"Classes range from {n(TOT['Fetal abdomen'])} abdomen images to {n(TOT['Other'])} 'Other' images.",
                "Imbalance affects both training (we used a class-weighted loss) and evaluation (we report macro-F1, which weights each class equally).",
                f"We kept the dataset authors' official patient-level train/test split; a patient-disjoint ~10% of training patients forms the validation set ({n(SPL['val_patients'])} patients).",
                f"Test set: {n(SPL['test_images'])} images from {n(SPL['test_patients'])} patients, nearly as large as the training set ({n(SPL['train_images'])} images)."],
        plain="If one class dominates, a model can score well by favouring it. That is why we look beyond accuracy.",
        website="Chapter 2 shows the class table and how loss weights change with the class counts.",
        ask=["If 'Other' is a third of the data, what accuracy would a model get by always answering 'Other'?"],
        caveats=["Counts are computed from FETAL_PLANES_DB_data.csv and the project's split manifest (dataset_stats.json)."])))
    heading(s, "FETAL_PLANES_DB at a glance", "Burgos-Artizzu et al., Scientific Reports 2020 · public, CC BY 4.0")
    cd = CategoryChartData()
    cd.categories = SHORT
    for split, lab in (("train", f"Train ({n(SPL['train_patients'])} patients)"), ("val", f"Validation ({SPL['val_patients']})"), ("test", f"Test ({SPL['test_patients']})")):
        cd.add_series(lab, DS["class_counts"][split])
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.5), Inches(1.95), Inches(8.2), Inches(4.75), cd)
    ch_ = gf.chart
    style_chart(ch_, ["4A5A8A", "B5C0C6", "18232B"], legend=True, vmax=None, fmt="#,##0", gap=60, overlap=-5)
    stats = [(n(DS["total_images"]), "images"), (n(DS["total_patients"]), "patients"), ("6", "classes"),
             (f"{TOT['Other'] / TOT['Fetal abdomen']:.1f}×", "largest / smallest class")]
    for i, (v, l_) in enumerate(stats):
        y = 2.0 + i * 1.15
        text(s, 9.2, y, 3.6, 0.6, v, size=30, color=INK, bold=True)
        text(s, 9.2, y + 0.62, 3.6, 0.35, l_, size=13, color=INK3)
    badge(s, "meta", 0.6, 6.78, "dataset_stats.json")

    s = new_slide(prs, SlideSpec("2", "Split by patient, not by image", 4, Notes(
        message="Images from the same patient look alike, so every patient must sit entirely in train, validation or test. Otherwise test scores are inflated.",
        points=[f"Our split is patient-disjoint: {n(SPL['train_patients'])} / {n(SPL['val_patients'])} / {n(SPL['test_patients'])} patients, overlap {SPL['patient_overlap']}.",
                f"Patient-disjoint is not the same as dissimilar: a deliberately loose perceptual-hash detector flags {n(ND['cross_split_pairs'])} look-alike pairs between training/validation and test images (from different patients), involving {n(ND['test_images_involved'])} test images.",
                f"Scanner shift is built into the official split: Voluson S10 contributes {MBS['train']['Voluson S10']} training images but {n(MBS['test']['Voluson S10'])} test images.",
                "Model selection used only the validation patients; the test set was used once for reporting."],
        plain="If you study with this year's exam questions, your grade says little about what you learned. Image-level splitting does exactly that with patients.",
        website="Chapter 2 split audit and leakage simulator show these numbers and what leakage does to a score.",
        ask=["Where else could leakage hide in a clinical dataset (same visit, same machine, same operator)?"],
        caveats=["dHash flags perceptually generic frames, not only copies; it is a screening signal, not proof of duplication.",
                 "The diagram is illustrative; the numbers are from the project's split manifest."])))
    heading(s, "Split by patient, not by image", "Images of the same patient are near-copies; they must never straddle the train/test boundary.")
    split_diagram(s)
    badge(s, "illus", 0.6, 6.78)
    badge(s, "meta", 2.55, 6.78, "split manifest")

    activity_slide(prs, "2", "Preprocessing laboratory", 4, "Chapter 2",
                   ["Open the preprocessing lab and pick a sample image.", "Raise brightness by +0.15, then halve the resolution.", "Add speckle and watch the tensor histogram and the predictions."],
                   "Which change would a model notice that a sonographer would barely register?",
                   "Resizing, normalisation and colour handling were decided before training. Which of them are really modelling decisions?",
                   Notes(message="Preprocessing is part of the model: the tensor the network sees is not the image the clinician sees.",
                         points=["Each model uses its own established input recipe (e.g. 224×224 stretch; different interpolation and normalisation).",
                                 "Predictions for the exported settings (brightness ±0.15, contrast, speckle, ½ and ¼ resolution) are real saved model outputs; other slider positions are a live simulation."],
                         plain="Small, almost invisible changes to pixels can move a model's probabilities.",
                         website="Chapter 2 › Preprocessing lab (presets per model; brightness, resolution, speckle; histogram of the tensor).",
                         cue="Ask participants to open Chapter 2, choose the USF-MAE preset, and move brightness to +0.15.",
                         ask=["Did any prediction change? Which model was most sensitive on your image?"],
                         caveats=["Perturbation predictions exist only for the exported settings; the website labels live-simulation versus model output."]))

    # =========================================================== CHAPTER 3
    s = new_slide(prs, SlideSpec("3", "DenseNet-161: local filters, reused features", 5, Notes(
        message="A CNN slides small learned filters across the image; stacking them builds features from edges to anatomy. DenseNet feeds every layer the outputs of all earlier layers in its block.",
        points=["Convolution: the same small weight pattern is applied everywhere. That is locality and weight sharing.",
                "Hierarchy: deeper layers see larger regions (receptive field grows) at lower resolution.",
                "Dense connectivity: each layer receives the concatenation of all previous feature maps, so features are reused rather than relearned.",
                "Classification: global average pooling turns the final 7×7 maps into one vector, followed by a linear layer to six classes. This pooling is what makes CAM possible in Chapter 5."],
        plain="Imagine a set of stencils for small patterns; the network learns which stencils matter and combines them layer by layer.",
        website="Chapter 3A: kernel explorer, receptive-field table and dense-block connectivity diagram.",
        ask=["Why might weight sharing suit ultrasound, where the same texture appears everywhere?"],
        caveats=["Diagram is schematic; DenseNet-161 has four dense blocks with 6, 12, 36 and 24 layers."])))
    heading(s, "DenseNet-161: local filters, reused features", "Convolution finds local patterns; dense connections let every layer reuse all earlier features.")
    densenet_diagram(s)
    badge(s, "illus", 0.6, 6.78)

    s = new_slide(prs, SlideSpec("3", "What DenseNet sees, block by block", 4, Notes(
        message="Real activations of the trained DenseNet-161 on the opening image: resolution falls and channels multiply as the network goes deeper.",
        points=[f"Shown: one exported channel per dense block (the one with the highest mean activation), {', '.join(f'{sh[1]}×{sh[2]} ({sh[0]:,} channels)' for _, sh, _ in FP['feature_maps'])}.",
                "Early maps still look like the image (edges, boundaries); late maps are coarse and abstract.",
                "The 7×7 final maps are what the classifier pools; this coarse grid is why DenseNet heatmaps are blurry."],
        plain="Each square is one 'detector' the network built; brighter means it responded more strongly there.",
        website="Chapter 3A › real feature maps: browse other channels and blocks.",
        ask=["At which depth does the image stop being recognisable to you?"],
        caveats=["A single channel out of hundreds is not a summary of the layer; channels were selected automatically by mean activation.",
                 "Feature maps were computed with the DenseNet checkpoint in the bundle (functionally equivalent rerun weights; see appendix)."])))
    heading(s, "What DenseNet sees, block by block", "Trained DenseNet-161 on the opening image: one exported channel per dense block.")
    xs = 0.6
    for i, (layer, shp, idx) in enumerate(FP["feature_maps"]):
        x = 0.6 + i * 3.08
        picture(s, FIG / f"feat_{layer.split('.')[-1]}.png", x, 2.0, 2.75, 2.75)
        text(s, x, 4.85, 2.75, 0.35, f"Dense block {i + 1}", size=16, color=INK, bold=True)
        text(s, x, 5.2, 2.75, 0.6, [f"{shp[1]}×{shp[2]} grid · {shp[0]:,} channels", f"channel {idx} shown"], size=12, color=INK3)
        if i < 3:
            arrow(s, x + 2.8, 3.37, x + 3.03, 3.37, color=INK3)
    text(s, 0.6, 6.0, 12, 0.4, "Finer detail  →  larger context: each step halves the grid and widens what one cell can see.", size=14, color=INK2)
    badge(s, "model", 0.6, 6.78, "DenseNet-161 features · opening image")

    s = new_slide(prs, SlideSpec("3", "ViT-B/16: an image as a sequence of patches", 4, Notes(
        message="A Vision Transformer cuts the image into 16×16-pixel patches, turns each into a vector (a token), adds position information, and lets all tokens interact.",
        points=["224×224 input → 14×14 = 196 patches of 16×16 pixels.",
                "Each patch is flattened and linearly projected to a 768-dimensional token; a learned position embedding records where it came from.",
                "A special [CLS] token is added; after 12 transformer blocks its final state feeds the classifier.",
                "There is no built-in notion of locality: neighbouring patches must learn to relate to each other."],
        plain="Like cutting a picture into a jigsaw, writing a description of each piece, and numbering the pieces so their arrangement isn't lost.",
        website="Chapter 3B: patch → token explorer, including the similarity of the learned position embeddings.",
        ask=["What might be lost when an image is cut into fixed 16-pixel squares?"],
        caveats=["Diagram uses the real opening image resized to the 224×224 model input; the token strip is schematic."])))
    heading(s, "ViT-B/16: an image as a sequence of patches", "196 patches → 196 tokens (+ [CLS]) → 12 transformer blocks → class.")
    vit_diagram(s)
    badge(s, "illus", 0.6, 6.78, "real image, schematic tokens")

    s = new_slide(prs, SlideSpec("3", "Self-attention: every patch can consult every other", 5, Notes(
        message="Self-attention lets each token gather information from all others, weighted by learned similarity. The weights are inspectable, but they are not an explanation.",
        points=["Each token emits a query, a key and a value; attention weights are the softmax of query–key similarity; the output is the weighted sum of values.",
                "12 heads per block learn different relations; there are 12 blocks.",
                "Panels: real [CLS]-token attention of the trained ViT-B/16 on the opening image, averaged over the 12 heads, after blocks 1, 6 and 12.",
                "Early attention is broad; later it concentrates on a few tokens, including a patch in the black corner (an 'attention sink' that carries little anatomy)."],
        plain="Attention is where the model routes information, a bit like who talks to whom in a meeting. It does not tell you which remark decided the vote.",
        website="Chapter 3B attention explorer: choose block, head and query patch; compare ViT-B/16 with USF-MAE.",
        cue="Ask participants to open Chapter 3, select an ultrasound image, and compare attention patterns across transformer blocks.",
        ask=["Does a bright attention region necessarily mean it influenced the predicted class?"],
        caveats=["Attention weights are not faithful attributions by default (Jain & Wallace 2019); Chapter 5 tests this.",
                 "Head-averaging hides head-specific patterns; maps are min–max scaled per panel."])))
    heading(s, "Self-attention: every patch can consult every other", "Real [CLS] attention of the trained ViT-B/16 (mean of 12 heads).")
    equation(s, "eq_attention", 0.6, 1.95, 0.62)
    for i, b in enumerate((1, 6, 12)):
        x = 0.6 + i * 4.1
        picture(s, FIG / f"attn_vit_b16_block{b}.png", x, 2.85, 3.85, 2.66)
        text(s, x, 5.55, 3.85, 0.3, f"After block {b} of 12", size=14, color=INK, bold=True)
    picture(s, FIG / "colorbar.png", 9.3, 2.05, 2.2, 0.17)
    text(s, 9.3, 2.25, 2.2, 0.3, "low  →  high attention", size=10, color=INK3)
    warn = box(s, 0.6, 5.95, 12.1, 0.62, fill="FBF3E2", line="E6C98A", shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12)
    label_in(warn, "Attention shows where information is **routed**, not **why** the class was chosen. A bright patch is a hypothesis, not evidence.", size=14.5, color="5A4410", align="l")
    badge(s, "model", 0.6, 6.78, "ViT-B/16 attention · opening image")

    s = new_slide(prs, SlideSpec("3", "USF-MAE: learning from unlabelled ultrasound", 3, Notes(
        message="A masked autoencoder hides most patches and learns to reconstruct them. To do that well on ultrasound, the encoder must learn ultrasound structure, without any labels.",
        points=["During pretraining the encoder sees only the visible patches; a small decoder predicts the pixels of the masked ones; the loss is computed on masked patches only.",
                "USF-MAE was pretrained this way on ~370k unlabelled ultrasound images (OpenUS-46), starting from ImageNet MAE weights. Our classifier uses the 500-epoch checkpoint.",
                "Panels: real reconstructions by the 500-epoch USF-MAE checkpoint of the opening image at 25, 50, 75 and 90% masking (visible patches pasted back).",
                f"Masked-patch reconstruction error (MSE in the model's normalised input space) grows with the masking ratio: {', '.join(f'{r['mask_ratio']:.0%}: {r['loss_masked_mse']:.3f}' for r in FP['reconstructions'])}."],
        plain="Like a fill-in-the-blanks exercise for images: guessing the missing pieces forces the model to learn what ultrasound anatomy usually looks like.",
        website="Chapter 3C: masking explorer (ratio, seed) and the authentic 500-epoch reconstructions; the historical 100-epoch images are behind a clearly labelled toggle.",
        ask=["At 90% masking the reconstruction is blurry but plausible. What has the model learned, and what has it not?"],
        caveats=["Reconstructions show what the pretrained encoder–decoder can infer, not what the fine-tuned classifier uses.",
                 "All images on this slide are from usf_mae_500ep.pth (sha256 c659a061…); the 100-epoch checkpoint is not used anywhere in this deck."])))
    heading(s, "USF-MAE: learning from unlabelled ultrasound", "Masked autoencoding: hide patches, reconstruct them, learn ultrasound structure without labels.")
    text(s, 0.6, 2.2, 1.55, 0.4, "Encoder input", size=12.5, color=INK2, bold=True)
    text(s, 0.6, 2.5, 1.55, 0.8, "(visible patches only)", size=11, color=INK3)
    text(s, 0.6, 4.55, 1.55, 0.4, "Reconstruction", size=12.5, color=INK2, bold=True)
    text(s, 0.6, 4.85, 1.55, 0.8, "(500-epoch checkpoint)", size=11, color=INK3)
    for i, r in enumerate(FP["reconstructions"]):
        pct = int(round(r["mask_ratio"] * 100))
        x = 2.1 + i * 2.08
        picture(s, FIG / f"mae_masked_{pct}.png", x, 1.95, 1.98, 1.98)
        picture(s, FIG / f"mae_recon_{pct}.png", x, 4.1, 1.98, 1.98)
        text(s, x, 6.12, 1.98, 0.3, f"{pct}% masked", size=13, color=INK, bold=True, align="c")
    equation(s, "eq_mae", 10.5, 2.1, 0.6, max_w=2.25)
    text(s, 10.5, 2.95, 2.3, 2.5, ["The decoder predicts the pixels of every masked patch p; the loss counts only the masked set M.", "No labels needed."], size=12.5, color=INK2, space_after=6)
    badge(s, "model", 0.6, 6.78, "usf_mae_500ep.pth reconstructions")

    s = new_slide(prs, SlideSpec("3", "From reconstruction to classification", 2, Notes(
        message="After pretraining, the decoder is discarded; the encoder plus a new six-way head is fine-tuned with labels, exactly as for the other two models.",
        points=["Pipeline: ImageNet MAE → USF-MAE pretraining on unlabelled ultrasound (500 epochs) → fine-tune encoder + linear head on labelled FETAL_PLANES_DB training patients.",
                "Pretraining overlap: USF-MAE's unlabelled pretraining corpus (OpenUS-46) included FETAL_PLANES_DB images, possibly including our test images.",
                "This is not label leakage (no labels were used in pretraining), but it is transductive exposure: the encoder may have seen the test images' pixels. It could favour USF-MAE, especially with few labels and on the Voluson S10 scanner.",
                "The overlap is stated by the project configuration and has not been independently verified."],
        plain="It is a bit like having flipped through the exam booklet without the answer key: you never saw the answers, but the pictures are familiar.",
        website="Chapter 3C overlap callout; the same disclosure travels with every USF-MAE result on the site.",
        ask=["How would you design an experiment that removes this overlap?"],
        caveats=["Do not describe the overlap as 'leakage' without qualification; it is unlabelled exposure.",
                 "An external test set or a pretraining corpus with FETAL_PLANES_DB removed would be needed to settle its effect."])))
    heading(s, "From reconstruction to classification", "Pretrain without labels, then fine-tune with labels: the decoder is thrown away.")
    mae_pipeline(s)
    disc = box(s, 0.6, 5.0, 12.1, 1.5, fill="FBF3E2", line="E6C98A", shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
    text(s, 0.9, 5.1, 11.6, 0.3, "PRETRAINING-OVERLAP DISCLOSURE", size=10.5, color="8A6D1F", bold=True)
    text(s, 0.9, 5.4, 11.6, 1.05, "USF-MAE's unlabelled pretraining corpus included FETAL_PLANES_DB images, possibly including this workshop's test images. "
         "This is **not label leakage**, but it is exposure to test pixels that may favour USF-MAE, especially with few labels. Stated by the project configuration; not independently verified.",
         size=14, color="4A3A10")
    badge(s, "illus", 0.6, 6.78)

    s = new_slide(prs, SlideSpec("3", "Three ways to learn a representation", 1, Notes(
        message="The three models differ in architecture, in pretraining data, and in pretraining signal. Keep this table in mind when reading any comparison.",
        points=["DenseNet vs ViT: same pretraining data and signal (ImageNet labels), different architecture.",
                "ViT vs USF-MAE: same architecture and size, different pretraining (labelled photographs vs unlabelled ultrasound).",
                "Fine-tuning recipes follow each model family's established practice (e.g. layer-wise learning-rate decay for the transformers), not one shared setting."],
        plain="ViT-B/16 vs USF-MAE is the closest thing we have to a controlled comparison of 'photographs vs ultrasound' pretraining.",
        website="Chapter 3 model cards summarise these facts, with collapsed provenance details.",
        ask=["Which pair would you compare to ask whether ultrasound pretraining matters? What is still confounded?"],
        caveats=["Parameter counts are for the six-class models as trained."])))
    heading(s, "Three ways to learn a representation", "Read every later comparison through this table.")
    rows = [["", "DenseNet-161", "ViT-B/16", "USF-MAE"],
            ["Architecture", "Convolutional (dense blocks)", "Transformer, 16×16 patches", "Transformer, 16×16 patches"],
            ["Built-in bias", "Locality, weight sharing", "Global mixing, learned positions", "Global mixing, learned positions"],
            ["Pretraining data", "ImageNet photographs", "ImageNet photographs", "~370k ultrasound images (+ ImageNet MAE start)"],
            ["Pretraining signal", "Labels (supervised)", "Labels (supervised)", "No labels: masked reconstruction, 500 epochs"],
            ["Parameters", PM_PARAMS["densenet161"], PM_PARAMS["vit_b16"], PM_PARAMS["usf_mae"]],
            ["Fine-tuning", "All weights, on the same labelled patients and validation-based model selection", "", ""]]
    tbl = table(s, 0.6, 1.95, 12.1, 4.5, rows, col_w=[2.2, 3.3, 3.3, 3.3], size=14, row_h=0.62)
    tbl.cell(6, 1).merge(tbl.cell(6, 3))
    for i, m in enumerate(("densenet161", "vit_b16", "usf_mae")):
        c = tbl.cell(0, i + 1)
        c.fill.fore_color.rgb = rgb(M[m])
    badge(s, "meta", 0.6, 6.78, "model configs")

    activity_slide(prs, "3", "Inside the three models", 1, "Chapter 3",
                   ["In the attention explorer, pick block 12 and click a patch on the anatomy.", "Switch between ViT-B/16 and USF-MAE for the same image.", "Open the masking explorer and raise the mask ratio to 90%."],
                   "Does a bright attention region necessarily mean it influenced the predicted class?",
                   "What did USF-MAE have to learn to reconstruct 75% of an ultrasound image?",
                   Notes(message="Hands-on comparison of CNN features, transformer attention and MAE reconstruction.",
                         points=["One minute to orient people; they can continue during the break.", "Encourage people to find the 'attention sink' patches in late blocks."],
                         plain="Everything labelled 'Model output' on the site is precomputed from the real checkpoints.",
                         website="Chapter 3 › feature maps (3A), attention explorer (3B: block, head, query patch), masking explorer and reconstructions (3C).",
                         cue="Ask participants to open Chapter 3, select an ultrasound image, and compare attention patterns across transformer blocks.",
                         ask=["Did the two transformers attend to the same places?"],
                         caveats=["Attention ≠ attribution; reconstructions ≠ what the classifier uses."]))

    # =========================================================== CHAPTER 4
    s = new_slide(prs, SlideSpec("4", "Fine-tune, select, then test once", 3, Notes(
        message="All three encoders get a new six-way head and are fine-tuned; the validation patients choose the checkpoint; the test patients are used once.",
        points=["Training: class-weighted loss, augmentation (small rotations, scaling, flips, brightness/contrast), up to 25 epochs with early stopping on validation macro-F1.",
                "Accuracy counts correct images; it is dominated by large classes such as 'Other'.",
                "Macro-F1 averages the per-class F1 scores, so the smallest class counts as much as the largest.",
                "We also report 95% confidence intervals from a patient-clustered bootstrap (1,000 resamples)."],
        plain="Accuracy asks 'how often right overall?' Macro-F1 asks 'how good on each type of image, on average?'",
        website="Chapter 4 opening section explains linear probing vs full fine-tuning (both were run; results here are full fine-tuning).",
        ask=["Why should the validation set, not the test set, pick the best epoch?"],
        caveats=["Recipes (learning rates, schedules, input transforms) differ by model family; see the appendix."])))
    heading(s, "Fine-tune, select, then test once", "One workflow for all three models; only the test patients decide the reported numbers.")
    workflow_diagram(s)
    equation(s, "eq_macro_f1", 0.9, 4.95, 0.78)
    text(s, 6.3, 4.85, 6.4, 1.2, ["**Accuracy**: fraction of images classified correctly; dominated by large classes.",
                                 "**Macro-F1**: mean of per-class F1 (P = precision, R = recall); each plane counts equally."], size=14, color=INK2, space_after=6)
    badge(s, "illus", 0.6, 6.78)

    s = new_slide(prs, SlideSpec("4", "With all labels, no significant difference", 4, Notes(
        message="At 100% labels the three models perform almost identically, and none of the pairwise differences is statistically significant.",
        points=[f"Accuracy / macro-F1: DenseNet-161 {f4(PM['densenet161']['accuracy'])} / {f4(PM['densenet161']['macro_f1'])}; ViT-B/16 {f4(PM['vit_b16']['accuracy'])} / {f4(PM['vit_b16']['macro_f1'])}; USF-MAE seed 0 {f4(PM['usf_mae']['accuracy'])} / {f4(PM['usf_mae']['macro_f1'])}.",
                f"USF-MAE over three seeds: accuracy {f4(SEED['mean']['accuracy'])} ± {f4(SEED['sd']['accuracy'])}, macro-F1 {f4(SEED['mean']['macro_f1'])} ± {f4(SEED['sd']['macro_f1'])} (mean ± sample sd). DenseNet and ViT have one seed at 100%.",
                "The 95% confidence intervals overlap almost completely.",
                "Exact McNemar tests on paired test predictions: " + "; ".join(f"{NAME[c['a']]} vs {NAME[c['b']]} p = {c['p_value']:.2f}" for c in EV["comparisons"]) + "; Holm-adjusted p = 1 for all."],
        plain="On this test set the three approaches are tied within measurement uncertainty.",
        website="Chapter 4 › 'What the test set says': per-model metrics, CIs and the McNemar table.",
        ask=["If they are tied with all labels, is there any reason to prefer one?"],
        caveats=["'Not significantly different' is not 'equivalent': no equivalence test was run.",
                 "McNemar pairs images and ignores patient clustering.",
                 "DenseNet metrics come from the original run's saved predictions (see appendix on the checkpoint)."])))
    heading(s, "With all labels: no significant difference", "Test set: 5,271 images, 896 patients. Points = estimate; bars = 95% patient-clustered bootstrap CI.")
    forest_plot(s)
    badge(s, "model", 0.6, 6.78, "evaluation.json")

    s = new_slide(prs, SlideSpec("4", "Where the errors are", 3, Notes(
        message="Most errors involve the catch-all 'Other' class; confusions among the five specific planes are rare.",
        points=["Rows are the expert label, columns the prediction; shading is the row percentage.",
                "Shown for USF-MAE (seed 0). DenseNet-161 and ViT-B/16 show the same pattern (all three matrices are on the website).",
                "'Other' images predicted as femur, abdomen or thorax, and specific planes predicted as 'Other', account for most errors.",
                "Some 'Other' confusions likely reflect label ambiguity as much as model error."],
        plain="Read across a row: of all the true thorax images, how many did the model call thorax, and what did it call the rest?",
        website="Chapter 4 confusion explorer: click a cell to see the actual misclassified images.",
        ask=["Is a femur image called 'Other' a dangerous mistake? Depends on the clinical use."],
        caveats=["Per-class results come from one fine-tuning seed."])))
    heading(s, "Where the errors are", "USF-MAE confusion matrix (seed 0, test set): rows = expert label, columns = prediction.")
    confusion(s, "usf_mae")
    badge(s, "model", 0.6, 6.78, "evaluation.json")

    s = new_slide(prs, SlideSpec("4", "Confidence is not correctness: calibration", 3, Notes(
        message="All three models were under-confident: their probabilities were too soft. One fitted temperature per model fixes this almost completely.",
        points=["Expected calibration error (ECE, 10 bins) compares confidence with accuracy; 0 is perfect.",
                "Temperature T was fitted on the validation set; T < 1 sharpens the probabilities.",
                "  ".join(f"{NAME[m]}: ECE {f3(PM[m]['ece'])} → {f3(PM[m]['ece_after_temperature'])} (T = {PM[m]['temperature_fit_on_val']:.2f})." for m in PM),
                "Temperature scaling does not change any prediction, only the confidence attached to it."],
        plain="A model that says '60%' when it is right 95% of the time is honest about rankings but misleading about certainty.",
        website="Chapter 4 calibration lab: reliability diagram and the 'Apply fitted T' button.",
        ask=["For a clinical tool, is under-confidence safer than over-confidence?"],
        caveats=["Calibration was measured on the same test distribution; it may not hold on new scanners or sites."])))
    heading(s, "Confidence is not correctness: calibration", "Expected calibration error on the test set, before and after one validation-fitted temperature.")
    cd = CategoryChartData()
    cd.categories = [NAME[m] for m in PM]
    cd.add_series("ECE as trained", [round(PM[m]["ece"], 4) for m in PM])
    cd.add_series("ECE after temperature scaling", [round(PM[m]["ece_after_temperature"], 4) for m in PM])
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.5), Inches(1.95), Inches(7.6), Inches(4.7), cd)
    style_chart(gf.chart, [RULE2, INK], legend=True, vmax=0.16, fmt="0.000", labels=True, gap=70, overlap=-10)
    equation(s, "eq_temperature", 8.6, 2.2, 0.5)
    text(s, 8.6, 2.85, 4.2, 0.8, "z = logits, T fitted by minimising validation negative log-likelihood.", size=13, color=INK2)
    for i, m in enumerate(PM):
        y = 3.75 + i * 0.85
        box(s, 8.6, y + 0.08, 0.08, 0.6, fill=M[m])
        text(s, 8.85, y, 4, 0.35, NAME[m], size=14, color=M_TEXT[m], bold=True)
        text(s, 8.85, y + 0.35, 4, 0.35, f"T = {PM[m]['temperature_fit_on_val']:.2f}  ·  ECE {f3(PM[m]['ece'])} → {f3(PM[m]['ece_after_temperature'])}", size=13, color=INK2)
    badge(s, "model", 0.6, 6.78, "evaluation.json")

    d5f, d5fs, _ = le_at("densenet161", "macro_f1", 0.05)
    u5f, u5fs, _ = le_at("usf_mae", "macro_f1", 0.05)
    d5a, d5as, _ = le_at("densenet161", "accuracy", 0.05)
    u5a, u5as, _ = le_at("usf_mae", "accuracy", 0.05)
    imgs5 = LE["per_model"]["usf_mae"]["n_train_images"][1]
    pats5 = LE["per_model"]["usf_mae"]["n_train_patients"][1][0]
    s = new_slide(prs, SlideSpec("4", "When labels are scarce", 4, Notes(
        message="The ultrasound-pretrained USF-MAE is clearly ahead of ImageNet-pretrained DenseNet-161 at 5% labels, ahead on average at 1% and 10%, and the gap closes as labels increase.",
        points=[f"At 5% of training patients ({pats5} patients, {min(imgs5)}–{max(imgs5)} images): macro-F1 {f3(u5f)} ± {f3(u5fs)} vs {f3(d5f)} ± {f3(d5fs)}; accuracy {f3(u5a)} ± {f3(u5as)} vs {f3(d5a)} ± {f3(d5as)} (USF-MAE vs DenseNet-161, mean ± sd over 3 seeds).",
                "At 5% the seed ranges do not overlap. At 1% USF-MAE is also ahead on average, but DenseNet's seeds vary widely. From 25% upward the curves nearly coincide.",
                "Accuracy and macro-F1 tell the same story here, but they are different metrics: quote both with their names.",
                "ViT-B/16, the architecture-matched control, was not run below 100% labels. Without it, ultrasound pretraining cannot be separated from architecture."],
        plain="With only a few dozen labelled patients, prior ultrasound experience seems to help a lot. With hundreds of patients, it matters little.",
        website="Chapter 4 label-efficiency chart: toggle macro-F1/accuracy and inspect every seed.",
        ask=["Where in clinical AI are labels genuinely scarce (rare anomalies, new planes, new devices)?"],
        caveats=["No significance test was run on the low-label differences; three seeds per point.",
                 "Models differ in architecture, pretraining data and fine-tuning recipe, not one factor.",
                 "USF-MAE's pretraining corpus included FETAL_PLANES_DB images, which may favour it most at low label fractions.",
                 "Label fractions sample training patients; the same seed gives the same subset for both models. Exact subsets were not saved.",
                 "Treat this as hypothesis-generating until replicated on an external dataset."])))
    heading(s, "When labels are scarce", "Fine-tuning on 1–100% of training patients · 3 seeds per point · same patient subsets for both models.")
    picture(s, FIG / "label_efficiency.png", 0.55, 1.8, 8.85, 4.05)
    for i, (big, lab, sub) in enumerate([(f"{f3(u5f)} vs {f3(d5f)}", "macro-F1 at 5% labels", "USF-MAE vs DenseNet-161"),
                                         (f"{f3(u5a)} vs {f3(d5a)}", "accuracy at 5% labels", "mean of 3 seeds each"),
                                         ("not run", "ViT-B/16 below 100%", "architecture control missing")]):
        y = 1.95 + i * 1.3
        text(s, 9.6, y, 3.4, 0.5, big, size=22, color=INK, bold=True)
        text(s, 9.6, y + 0.5, 3.4, 0.3, lab, size=12.5, color=INK2, bold=True)
        text(s, 9.6, y + 0.8, 3.4, 0.3, sub, size=11.5, color=INK3)
    cav = box(s, 0.6, 5.95, 12.1, 0.62, fill="FBF3E2", line="E6C98A", shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12)
    label_in(cav, "Suggestive, not settled: no significance test · three factors differ between models · pretraining overlap · no external test set.", size=13.5, color="5A4410", align="l")
    n_runs = sum(1 for m in LE["per_model"].values() for row in m["runs"] for r in row if r)
    badge(s, "model", 0.6, 6.78, f"label_efficiency.json · {n_runs} runs")

    s = new_slide(prs, SlideSpec("4", "Same models, different scanners", 2, Notes(
        message="Performance depends on the scanner. One scanner is almost absent from training but makes up a fifth of the test set.",
        points=[f"Voluson S10: {MBS['train']['Voluson S10']} training images, {n(MBS['test']['Voluson S10'])} test images.",
                "Accuracy on Voluson S10 test images is a few points below Aloka for all three models.",
                f"'Other' machines: only {MBS['test']['Other']} test images, and all three models made identical predictions on every one of them; the per-class scores there are not comparable because some classes are absent.",
                "This is a within-dataset hint of domain shift, not an external validation."],
        plain="A model trained mostly on one scanner may behave differently on another. A new hospital is a bigger shift still.",
        website="Chapter 4 › performance by ultrasound machine table.",
        ask=["What would you need to see before using such a model on a scanner it has not been trained on?"],
        caveats=["Scanner is confounded with operator, patient population and class mix.",
                 "USF-MAE's pretraining exposure may matter most on Voluson S10 (see overlap disclosure)."])))
    heading(s, "Same models, different scanners", "Test accuracy by ultrasound machine, with the training images each machine contributed.")
    rows = [["Machine", "Train images", "Test images", "DenseNet-161", "ViT-B/16", "USF-MAE"]]
    fills = [[None] * 6]
    for mach in ("Aloka", "Voluson E6", "Voluson S10", "Other"):
        r = [mach, n(MBS["train"][mach]), n(MBS["test"][mach])] + [f3(PM[m]["by_machine"][mach]["accuracy"]) for m in PM]
        rows.append(r)
        fills.append([("FBF3E2" if mach == "Voluson S10" else None)] * 6)
    tbl = table(s, 0.6, 2.0, 12.1, 3.0, rows, col_w=[2.6, 1.9, 1.9, 1.9, 1.9, 1.9], size=16, fills=fills, row_h=0.58)
    for i, m in enumerate(PM):
        tbl.cell(0, 3 + i).fill.fore_color.rgb = rgb(M[m])
    text(s, 0.6, 5.2, 12.1, 1.2, [f"**Voluson S10** is {MBS['test']['Voluson S10'] / SPL['test_images']:.0%} of the test set but {MBS['train']['Voluson S10'] / SPL['train_images']:.1%} of training images: a built-in scanner shift.",
                                  f"**Other** (n = {MBS['test']['Other']}): all three models give identical predictions; per-class scores there are not comparable."], size=14.5, color=INK2, space_after=6)
    badge(s, "model", 0.6, 6.78, "evaluation.json")
    badge(s, "meta", 4.05, 6.78, "dataset_stats.json")

    activity_slide(prs, "4", "Evaluation laboratory", 1, "Chapter 4",
                   ["Click a confusion-matrix cell to see the misclassified images.", "Press “Apply fitted T” in the calibration lab.", "Toggle the label-efficiency chart between macro-F1 and accuracy."],
                   "Which errors look like model mistakes, and which look like label ambiguity?",
                   "Which single result from this chapter would you report to a clinical colleague, and with which caveat?",
                   Notes(message="Introduce the evaluation lab now; participants explore it during the break.",
                         points=["One minute: show where things are, then release people to the break."],
                         plain="All Chapter 4 numbers on the site come from the same saved evaluation as these slides.",
                         website="Chapter 4 › confusion explorer, threshold/ROC, prevalence simulator, calibration lab, label-efficiency curves, machine table.",
                         cue="Ask participants to open Chapter 4 and click the largest off-diagonal cell before the break.",
                         ask=["(after the break) Did anyone find an error they would also have made?"],
                         caveats=["Prevalence and some calibration panels use a clearly labelled hypothetical classifier when real data are not applicable."]))

    # =========================================================== BREAK
    s = new_slide(prs, SlideSpec("B", "Break", 10, Notes(
        message="Ten-minute break. Invite self-guided exploration.",
        points=["Restart at 80 minutes with explainability.", "Leave this slide up; it lists three self-guided activities."],
        plain="Use the break to try the website, or to rest.",
        website="Suggested: Chapter 3 attention explorer, Chapter 4 confusion drill-down, Chapter 5 explainability lab preview.",
        ask=["(informal) What surprised you so far?"])), dark=True)
    text(s, 0.75, 1.5, 11, 0.4, "70–80 MIN", size=13, color="8FB3C4", bold=True)
    text(s, 0.75, 1.95, 11, 1.2, "Ten-minute break", size=54, color="FFFFFF", font=SERIF)
    text(s, 0.75, 3.15, 11, 0.5, "If you like, try one of these on the workshop website:", size=19, color=WELL_INK)
    for i, (c, t) in enumerate([("Chapter 3", "Find an 'attention sink' patch in a late transformer block."),
                                ("Chapter 4", "Open the largest off-diagonal confusion cell. Model error, or label ambiguity?"),
                                ("Chapter 5", "Pick one image and compare two explanation methods for the same model.")]):
        y = 3.95 + i * 0.78
        k = box(s, 0.78, y, 1.6, 0.52, fill="1E2B33", line="3A4A54", shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.3)
        label_in(k, c, size=13, color="FFFFFF", bold=True)
        text(s, 2.6, y, 7.9, 0.52, t, size=16.5, color="FFFFFF", anchor="m")
    qr_block(s, 11.1, 3.95, 1.55, dark=True)
    text(s, 10.85, 5.6, 2.05, 0.7, url_text(), size=10, color=WELL_INK, align="c")

    # =========================================================== CHAPTER 5
    s = new_slide(prs, SlideSpec("5", "Why explain a prediction?", 3, Notes(
        message="Explanations are themselves models of the model. A heatmap can be plausible (it looks right to us) without being faithful (reflecting what the model actually used).",
        points=["Reasons to explain: debugging, detecting shortcuts, supporting clinical review, regulatory expectations.",
                "Plausibility is judged by humans; faithfulness has to be tested by intervening on the input or the model.",
                "Every method makes assumptions (layer choice, gradients, baselines) that shape the picture."],
        plain="A convincing heatmap is like a confident witness: persuasive, but it still needs corroborating evidence.",
        website="Chapter 5 introduction and toy CAM / Grad-CAM / Grad-CAM++ demonstration.",
        ask=["Have you seen a heatmap in a paper or product that convinced you? What would have changed your mind?"],
        caveats=["Neither plausibility nor faithfulness scores establish clinical reliability."])))
    heading(s, "Why explain a prediction?", "Two different questions that are easy to confuse.")
    for i, (h_, q_, how, col) in enumerate([("Plausible", "Does the explanation look right to an expert?", "Judged by people. Easy to show, easy to be fooled by.", "8A6D1F"),
                                            ("Faithful", "Does it reflect what the model actually relied on?", "Tested by intervention: remove or keep the highlighted pixels and measure the change.", "1D7A4E")]):
        x = 0.6 + i * 6.2
        box(s, x, 1.95, 5.9, 3.5, fill=RAISED, line=RULE)
        box(s, x, 1.95, 0.1, 3.5, fill=col)
        text(s, x + 0.4, 2.15, 5.3, 0.6, h_, size=30, color=col, font=SERIF)
        text(s, x + 0.4, 2.95, 5.3, 1.0, q_, size=20, color=INK, font=SERIF)
        text(s, x + 0.4, 4.2, 5.3, 1.1, how, size=15, color=INK2)
    text(s, 0.6, 5.75, 12.1, 0.7, "A map can be plausible and unfaithful, or faithful and implausible. **Neither one proves clinical reliability.**", size=18, color=INK)
    badge(s, "illus", 0.6, 6.78)

    s = new_slide(prs, SlideSpec("5", "From CAM to Grad-CAM++", 5, Notes(
        message="Class activation maps weight the final convolutional feature maps by how much each one supports the class, then sum them into a heatmap.",
        points=["CAM: uses the classifier weights directly; needs global average pooling right before the classifier (DenseNet has it).",
                "Grad-CAM: weights each feature map by the average gradient of the class score; works at any convolutional layer.",
                "Grad-CAM++: uses higher-order gradient terms to weight pixels, which can help when several regions support the class.",
                "Resolution matters: DenseNet's final features are 7×7, so each cell covers a 32×32-pixel region of the 224×224 input. Heatmaps are upsampled and look smoother than they are."],
        plain="Ask the network: which of your detectors fired for 'abdomen', and where were they firing? Blend those detector maps, weighted by importance.",
        website="Chapter 5 toy CAM / Grad-CAM / Grad-CAM++ panel with a deletion curve; the real lab follows.",
        ask=["Why might a 7×7 map miss a thin structure like a femur?"],
        caveats=["Formula shown is Grad-CAM; Grad-CAM++ changes how α is computed.", "All maps are post-hoc and depend on the chosen layer."])))
    heading(s, "From CAM to Grad-CAM++", "Weight the final feature maps by their importance for the class; sum; keep the positive part.")
    gradcam_diagram(s)
    equation(s, "eq_gradcam", 0.6, 4.65, 0.72)
    badge(s, "model", 2.5, 6.78, "DenseNet Grad-CAM panel")
    for i, (h_, b_) in enumerate([("CAM", "weights = classifier weights after global pooling"),
                                  ("Grad-CAM", "weights = average gradient of the class score"),
                                  ("Grad-CAM++", "weights = higher-order gradient terms per location")]):
        x = 0.6 + i * 4.1
        text(s, x, 5.6, 3.9, 0.35, h_, size=16, color=INK, bold=True)
        text(s, x, 5.95, 3.9, 0.6, b_, size=13, color=INK2)
    badge(s, "illus", 0.6, 6.78)

    s = new_slide(prs, SlideSpec("5", "Attribution for transformers", 4, Notes(
        message="Transformers have no convolutional feature maps, so we use the tokens and the attention matrices. Different methods answer different questions.",
        points=["Attention rollout (Abnar & Zuidema 2020): multiply the attention matrices of all 12 blocks to trace how information flows to [CLS]. It is class-agnostic.",
                "Gradient-weighted attention rollout: weights each block's attention by its gradient for the target class before rolling out. This is the workshop's simplified, Chefer-inspired variant (no LRP); older files call it 'chefer'.",
                "Token Grad-CAM / Grad-CAM++: Grad-CAM applied to the 14×14 patch tokens after a chosen block.",
                "Panels: USF-MAE on the opening image. Plain rollout lands on the image edge and the black corner; adding class gradients moves it onto the abdomen."],
        plain="Plain rollout follows the wiring; gradient-weighted rollout asks which wiring mattered for 'abdomen'.",
        website="Chapter 5 compatibility table (which method works for which architecture) and the explainability lab.",
        ask=["Why would a class-agnostic method be a poor explanation of a specific prediction?"],
        caveats=["The gradient-weighted rollout here is not the full Chefer et al. (2021) method.",
                 "Each map is min–max scaled to its own range; colours are not comparable across maps."])))
    heading(s, "Attribution for transformers", "Use the tokens and the attention matrices instead of convolutional feature maps.")
    meths = [("Attention rollout", "Multiply attention across all 12 blocks. Class-agnostic.", "xai_usf_rollout"),
             ("Gradient-weighted attention rollout", "Weight each block's attention by the class gradient, then roll out.", "xai_usf_gradrollout")]
    for i, (h_, b_, f_) in enumerate(meths):
        x = 0.6 + i * 4.05
        picture(s, FIG / f"{f_}.png", x, 1.95, 3.85, 2.66)
        text(s, x, 4.68, 3.85, 0.35, h_, size=14.5, color=INK, bold=True)
        text(s, x, 5.05, 3.85, 0.8, b_, size=12.5, color=INK2)
    text(s, 8.9, 1.95, 3.9, 0.35, "Also in the lab", size=14.5, color=INK, bold=True)
    text(s, 8.9, 2.35, 3.9, 2.4, ["**Token Grad-CAM / ++**: Grad-CAM on the 14×14 patch tokens after block 5, 8 or 11.",
                                  "**Naming**: gradient-weighted rollout is a simplified variant inspired by Chefer et al. 2021 (no LRP). Older files call it “chefer”."],
         size=13, color=INK2, space_after=8)
    picture(s, FIG / "colorbar.png", 8.9, 5.05, 2.4, 0.18)
    text(s, 8.9, 5.27, 3.9, 0.3, "low → high (each map scaled to its own range)", size=10, color=INK3)
    text(s, 0.6, 6.0, 12.1, 0.5, "USF-MAE, opening image, target class Fetal abdomen.", size=12, color=INK3)
    badge(s, "model", 0.6, 6.78, "USF-MAE attributions")

    s = new_slide(prs, SlideSpec("5", "Same image, three architectures", 4, Notes(
        message="For the same image and the same correct prediction, the three models' explanations highlight different regions with different spread.",
        points=["DenseNet-161 Grad-CAM: one broad blob over the abdomen (7×7 resolution).",
                "ViT-B/16 gradient-weighted rollout: patchy regions over the abdomen plus a hotspot in the black corner.",
                "USF-MAE gradient-weighted rollout: a compact region near the right abdominal wall.",
                "All three models predict Fetal abdomen with ≥96% probability; the maps are for that class."],
        plain="Three models agree on the answer but would 'point' to different places if asked why.",
        website="Chapter 5 explainability lab: same image, switch model; architecture-compare view.",
        ask=["Which map would you find most convincing? Is 'most convincing' the same as 'most faithful'?"],
        caveats=["Methods differ across architectures (Grad-CAM for the CNN, gradient-weighted rollout for the transformers); differences mix model and method.",
                 "Maps are min–max scaled per panel.", "DenseNet maps were computed with the functionally equivalent rerun checkpoint (appendix)."])))
    heading(s, "Same image, three architectures", "Opening image · target class Fetal abdomen · all three models correct.")
    for i, (m, f_, meth) in enumerate([("densenet161", "xai_densenet_gradcam", "Grad-CAM · final 7×7 features"),
                                       ("vit_b16", "xai_vit_gradrollout", "Gradient-weighted attention rollout"),
                                       ("usf_mae", "xai_usf_gradrollout", "Gradient-weighted attention rollout")]):
        x = 0.6 + i * 4.1
        box(s, x, 1.95, 3.85, 0.07, fill=M[m])
        picture(s, FIG / f"{f_}.png", x, 2.1, 3.85, 2.66)
        p_ = OPEN["predictions"][m]["probs"][OPEN["label"]]
        text(s, x, 4.85, 3.85, 0.4, NAME[m], size=19, color=M_TEXT[m], font=SERIF, bold=True)
        text(s, x, 5.27, 3.85, 0.6, [meth, f"P(Fetal abdomen) = {p_:.1%}"], size=12.5, color=INK2)
    picture(s, FIG / "colorbar.png", 0.6, 6.15, 2.4, 0.18)
    text(s, 3.1, 6.08, 9, 0.3, "low → high · each map scaled to its own range", size=10.5, color=INK3)
    badge(s, "model", 0.6, 6.78, "saved attributions · opening image")

    s = new_slide(prs, SlideSpec("5", "Same model, four explanations", 3, Notes(
        message="Even for a single model and a single image, explanation methods disagree. The choice of method is part of the result.",
        points=["All four maps explain the same USF-MAE prediction (Fetal abdomen) on the same image.",
                "Attention rollout lights up the image edge and the black corner; gradient-weighted rollout a compact region near the right abdominal wall; token Grad-CAM a few small spots; token Grad-CAM++ a broad spread over much of the image.",
                "MetaCAM++ appears on the website as a reserved slot for ongoing research. We present no method details or results for it."],
        plain="Ask four witnesses the same question and you get four answers. Chapter 6 asks which witness to believe.",
        website="Chapter 5 lab: change Method and Layer for the same model and image.",
        ask=["If methods disagree, how would you decide which to show a clinician?"],
        caveats=["Each map is scaled to its own range.", "MetaCAM++ is a placeholder; do not describe its method or results."])))
    heading(s, "Same model, four explanations", "USF-MAE · opening image · target class Fetal abdomen.")
    four = [("xai_usf_rollout", "Attention rollout"), ("xai_usf_gradrollout", "Gradient-weighted rollout"),
            ("xai_usf_tokengradcam_b11", "Token Grad-CAM · block 11"), ("xai_usf_tokengradcampp_b11", "Token Grad-CAM++ · block 11")]
    for i, (f_, lab) in enumerate(four):
        x = 0.6 + i * 3.06
        picture(s, FIG / f"{f_}.png", x, 2.0, 2.9, 2.0)
        text(s, x, 4.08, 2.9, 0.35, lab, size=13.5, color=INK, bold=True)
    mc = box(s, 0.6, 4.85, 12.1, 1.15, fill=RAISED, line=RULE2, dash=True)
    text(s, 0.9, 4.95, 11.6, 0.3, "METACAM++ · ONGOING RESEARCH", size=10.5, color=INK3, bold=True)
    text(s, 0.9, 5.25, 11.6, 0.7, "Reserved for a method under development. No methodology or results are presented in this workshop.", size=15, color=INK2)
    badge(s, "model", 0.6, 6.78, "USF-MAE attributions")

    activity_slide(prs, "5", "Explainability laboratory", 6, "Chapter 5",
                   ["Pick an image the models disagree on.", "Compare each model's map for the expert label, then for the predicted class.", "Change the method and the layer; watch what moves."],
                   "Does the map change more when you change the model or when you change the method?",
                   "Would you show one of these maps to a clinician as the reason for a prediction? Which, and with what warning?",
                   Notes(message="Participants test the 'methods disagree' claim on their own images.",
                         points=["Encourage choosing a misclassified image: explanations of wrong answers are revealing.",
                                 "Point out the raw value range shown beside each map: a 'hot' region can be a tiny absolute value."],
                         plain="Every map on the site is a saved output of the real models.",
                         website="Chapter 5 › Explainability laboratory: model × method × class × layer, opacity, threshold, colour scale; architecture-compare view.",
                         cue="Ask participants to open the Explainability laboratory and choose an image from the 'models disagree' set.",
                         ask=["Who found a map that looks convincing but highlights something non-anatomical?"],
                         caveats=["MetaCAM++ is a placeholder slot only."]))

    # =========================================================== CHAPTER 6
    fem, ab = FP["perturbations"]["femur"], FP["perturbations"]["abdomen"]
    s = new_slide(prs, SlideSpec("6", "Can a model be right for the wrong reasons?", 4, Notes(
        message="Test an explanation by intervening: blur the region a heatmap says is most important and see whether the prediction changes.",
        points=["Intervention: the top-10% region of DenseNet-161's Grad-CAM map is blurred; every model is re-run on the edited image.",
                f"Femur image: all three models drop from ≥{int(min(v['p_true'] for v in fem['before'].values()) * 100)}% to ≤{max(v['p_true'] for v in fem['after'].values()):.0%} probability for femur and switch to 'Other'. Here the highlighted region was necessary.",
                f"Opening abdomen image: DenseNet itself barely moves ({ab['before']['densenet161']['p_true']:.0%} → {ab['after']['densenet161']['p_true']:.0%}), USF-MAE barely moves, but ViT-B/16 drops to {ab['after']['vit_b16']['p_true']:.0%} and switches to 'Other'. Blurring the region DenseNet's own map ranked highest barely changed DenseNet's prediction: that region alone was not necessary (it may still have been used).",
                "Two images are anecdotes; the next slide summarises 30 images."],
        plain="If you cover what a model says it is looking at and it still answers the same, it was not really looking there, or not only there.",
        website="Chapter 6 model detective: lock your hypotheses first, then open the perturbation evidence table.",
        ask=["Can a model predict correctly for the wrong reasons? What would that look like in this table?"],
        caveats=["Blurring creates an unnatural image; changes may partly reflect the edit itself (distribution shift).",
                 "Baseline probabilities are from the exported PNG pipeline, matching the perturbed images; they can differ slightly from Chapter 1's.",
                 "Perturbation scores do not establish causality or clinical reliability."])))
    heading(s, "Can a model be right for the wrong reasons?", "Blur the top-10% region of DenseNet-161's Grad-CAM map, then ask every model again.")
    perturbation_case(s, 0.6, "Femur", fem)
    perturbation_case(s, 6.75, "Abdomen (opening image)", ab)
    badge(s, "model", 0.6, 6.78, "saved perturbation experiments")

    s = new_slide(prs, SlideSpec("6", "Testing explanations against a random map", 3, Notes(
        message="Across 30 test images, some explanation methods are no better than a random map at identifying the pixels the model needs.",
        points=["Deletion: progressively blur the highest-ranked pixels; a faithful map makes the class probability fall fast (low area under the curve, AUC).",
                "Insertion: progressively restore them onto a blurred image; a faithful map makes it rise fast (high AUC).",
                "Compare every method with a random map for the same model.",
                "Against each model's random map (differences within 0.02 count as ≈): better on both scores for DenseNet CAM and Grad-CAM, ViT-B/16 gradient-weighted rollout and token Grad-CAM, and USF-MAE gradient-weighted rollout and token Grad-CAM++.",
                "DenseNet Grad-CAM++ is clearly better on insertion and ≈ random on deletion; ViT-B/16 attention rollout and token Grad-CAM++ are worse on deletion but better on insertion; USF-MAE attention rollout is ≈ random on both; USF-MAE token Grad-CAM is worse on deletion and ≈ random on insertion."],
        plain="We check whether the heatmap ranks pixels better than chance at predicting what the model needs.",
        website="Chapter 6 faithfulness table (bottom of the chapter).",
        ask=["If a method is worse than random, what is it showing?"],
        caveats=["n = 30 images, 10 steps, Gaussian-blur baseline, true-label probability; sds up to 0.29 (most 0.2–0.29), no significance test. Most rankings are not statistically supported.",
                 "Faithfulness metrics have their own assumptions (blur baseline, out-of-distribution images).",
                 "Verdicts compare means with a 0.02 margin in both directions; they are descriptive, not tests."])))
    heading(s, "Testing explanations against a random map", "Deletion AUC (lower is better) · insertion AUC (higher is better) · 30 test images · mean ± sd.", size=28)
    faithfulness_table(s)
    badge(s, "model", 0.6, 6.78, "faithfulness.json")

    activity_slide(prs, "6", "Model detective", 3, "Chapter 6",
                   ["Choose an image and lock in your hypotheses first.", "Open the perturbation evidence: which edits flip which model?", "Click a row to compare the perturbed image and maps."],
                   "Can a model predict correctly for the wrong reasons? Find one candidate.",
                   "Which evidence would convince you that a model relies on anatomy, not on text, calipers or scanner style?",
                   Notes(message="Form a hypothesis, then test it with interventions.",
                         points=["The hypothesis lock is deliberate: it prevents post-hoc storytelling.", "Ask for one volunteer to share a case where the evidence contradicted their hypothesis."],
                         plain="Detective work: suspect first, evidence second, verdict last.",
                         website="Chapter 6 › hypothesis lock, perturbation evidence table (Δp, flips, map ρ), faithfulness table.",
                         cue="Ask participants to open Chapter 6, tick at least one hypothesis, then press 'Lock hypotheses and reveal evidence'.",
                         ask=["Did anyone's hypothesis survive the evidence?"],
                         caveats=["Perturbation results do not independently establish causality or clinical reliability."]))

    # =========================================================== CHAPTER 7
    s = new_slide(prs, SlideSpec("7", "Back to the opening image", 2, Notes(
        message="The opening image is a fetal abdomen. All three models were right and confident, yet their explanations differed, and blurring the region DenseNet's map ranked highest barely changed DenseNet's own prediction.",
        points=["Reveal: expert label Fetal abdomen. " + ", ".join(f"{NAME[m]} {OPEN['predictions'][m]['probs'][OPEN['label']]:.1%}" for m in ("densenet161", "vit_b16", "usf_mae")) + ".",
                "Compare with the room's vote from slide 2.",
                "Recap the lessons on the slide in one sentence each."],
        plain="Being right is the start of the evidence, not the end.",
        website="Chapter 7 shows the same image again with the reveal.",
        ask=["Did the room do better or worse than the models? Were you as confident?"],
        caveats=["Probabilities are from the saved evaluation run (original-resolution image)."])), dark=True)
    text(s, 0.6, 0.62, 7, 0.75, "Back to the opening image", size=30, color="FFFFFF", font=SERIF)
    picture(s, FIG / "opening.png", 0.6, 1.6, 6.3, 4.36)
    text(s, 0.6, 6.0, 6.3, 0.35, "Expert label: Fetal abdomen", size=17, color="FFFFFF", bold=True)
    for i, m in enumerate(("densenet161", "vit_b16", "usf_mae")):
        p_ = OPEN["predictions"][m]["probs"][OPEN["label"]]
        y = 1.65 + i * 0.6
        text(s, 7.4, y, 2.0, 0.45, NAME[m], size=14, color="FFFFFF", bold=True, anchor="m")
        box(s, 9.4, y + 0.12, 2.6 * p_, 0.22, fill=M[m])
        text(s, 12.05, y, 0.9, 0.45, f"{p_:.1%}", size=13, color="FFFFFF", anchor="m")
    lessons = ["Representation learning: different pretraining gave the same accuracy with all labels.",
               "Scarce labels: ultrasound pretraining looked most useful with few labels (with caveats).",
               "Evaluation: macro-F1, confidence intervals and calibration change what a number means.",
               "Explainability: maps disagree, and plausible is not faithful.",
               "Generalisation: scanner shift and pretraining overlap limit what this test set can show."]
    text(s, 7.4, 3.6, 5.5, 0.35, "WHAT WE LEARNED", size=11, color="8FB3C4", bold=True)
    text(s, 7.4, 3.95, 5.5, 2.8, lessons, size=13.5, color=WELL_INK, bullets=True, space_after=5)
    badge(s, "model", 7.4, 6.78, "saved test-set predictions")

    s = new_slide(prs, SlideSpec("7", "What would it take to trust it?", 3, Notes(
        message="Strong test-set numbers are necessary but far from sufficient for clinical use. Here is the evidence still missing.",
        points=["External validation: new hospitals, scanners and operators, with a pretraining corpus that excludes the test data.",
                "Prospective evaluation in the real workflow, with clinically meaningful endpoints and failure analysis.",
                "Calibration and uncertainty checked on the deployment population; a policy for low-confidence cases.",
                "Explanations tested for faithfulness and usefulness to clinicians, not only for plausibility.",
                "Regulatory, privacy and monitoring plans for drift after deployment."],
        plain="Today's models are good at a narrow task on one public dataset. Clinical trust needs evidence from where the model will actually be used.",
        website="Chapter 7 deployment checklist and the links to the notebooks.",
        ask=["Which single additional study would most change your confidence?", "Who should be in the room when deciding to deploy?"],
        caveats=["No clinical claims are made for these models; they are research and teaching artefacts."])))
    heading(s, "What would it take to trust it?", "Evidence still needed before any clinical use.")
    needs = [("External validation", "new sites, scanners and operators; no pretraining overlap"),
             ("Prospective study", "in the real workflow, with clinical endpoints"),
             ("Calibrated uncertainty", "checked on the deployment population"),
             ("Tested explanations", "faithful and useful to clinicians, not just plausible"),
             ("Governance", "regulation, privacy and monitoring for drift")]
    for i, (h_, b_) in enumerate(needs):
        y = 1.95 + i * 0.66
        k = box(s, 0.6, y, 0.46, 0.46, fill=INK, shape=MSO_SHAPE.OVAL)
        label_in(k, str(i + 1), size=14, color="FFFFFF", bold=True)
        text(s, 1.25, y - 0.02, 2.9, 0.5, h_, size=16, color=INK, bold=True, anchor="m")
        text(s, 4.2, y - 0.02, 4.0, 0.5, b_, size=13.5, color=INK2, anchor="m")
    th = box(s, 8.6, 1.95, 4.1, 3.2, fill=INK, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
    text(s, 8.9, 2.1, 3.6, 0.3, "TAKE-HOME", size=10.5, color="8FB3C4", bold=True)
    text(s, 8.9, 2.45, 3.6, 2.6, "How a model learned to see matters most when labels are scarce, and a convincing heatmap is a hypothesis, not a reason.",
         size=19, color="FFFFFF", font=SERIF)
    qr_block(s, 8.6, 5.3, 1.25)
    text(s, 10.0, 5.3, 2.8, 1.25, ["Keep exploring:", url_text()], size=11, color=INK2)
    text(s, 0.6, 5.4, 7.6, 0.4, "Thank you", size=24, color=INK, font=SERIF)
    text(s, 0.6, 5.9, 7.6, 0.4, AUTHORS, size=16, color=INK2)
    text(s, 0.6, 6.25, 7.6, 0.35, EVENT, size=12, color=INK3)

    # =========================================================== APPENDIX
    s = new_slide(prs, SlideSpec("7", "References and figure attribution", 0, Notes(
        message="Reference slide; not presented.", points=["Leave up during questions if useful."], plain="—", website="—", ask=["—"]), kind="appendix"))
    heading(s, "References and figure attribution", y=0.62, size=26)
    refs = ["Burgos-Artizzu X.P. et al. Evaluation of deep convolutional neural networks for automatic classification of common maternal fetal ultrasound planes. Sci. Rep. 10, 10200 (2020). Data: zenodo.org/records/3904280 (CC BY 4.0).",
            "Huang G. et al. Densely connected convolutional networks. CVPR 2017.",
            "Dosovitskiy A. et al. An image is worth 16×16 words. ICLR 2021. Steiner A. et al. How to train your ViT? TMLR 2022.",
            "He K. et al. Masked autoencoders are scalable vision learners. CVPR 2022.",
            "Megahed Y., Ducharme R., Erman A., Walker M.C., Hawken S., Chan A.D.C. USF-MAE: Ultrasound self-supervised foundation model with masked autoencoding. arXiv:2510.22990.",
            "Zhou B. et al. Learning deep features for discriminative localization (CAM). CVPR 2016. Selvaraju R.R. et al. Grad-CAM. ICCV 2017. Chattopadhay A. et al. Grad-CAM++. WACV 2018.",
            "Abnar S., Zuidema W. Quantifying attention flow in transformers. ACL 2020. Chefer H., Gur S., Wolf L. Transformer interpretability beyond attention visualization. CVPR 2021.",
            "Jain S., Wallace B.C. Attention is not explanation. NAACL 2019. Adebayo J. et al. Sanity checks for saliency maps. NeurIPS 2018.",
            "Petsiuk V. et al. RISE (deletion/insertion metrics). BMVC 2018. Guo C. et al. On calibration of modern neural networks. ICML 2017.",
            "Geirhos R. et al. Shortcut learning in deep neural networks. Nat. Mach. Intell. 2020."]
    text(s, 0.6, 1.45, 12.1, 4.3, refs, size=11.5, color=INK2, bullets=True, space_after=3)
    text(s, 0.6, 5.75, 12.1, 0.95, ["**Figures.** Ultrasound images: FETAL_PLANES_DB (CC BY 4.0). Feature maps, attention, attributions, reconstructions, perturbations and all metrics: saved outputs of this project's models (website bundle schema 1.1). Diagrams: illustrative, drawn for this deck.",
                                     "**Reuse.** USF-MAE checkpoints have no stated licence and derive from CC BY-NC 4.0 MAE weights; treat USF-MAE reconstructions and USF-MAE-derived maps as non-commercial research and teaching material."],
         size=10.5, color=INK3, space_after=3)

    s = new_slide(prs, SlideSpec("7", "Provenance and limitations", 0, Notes(
        message="Reference slide for questions about data, checkpoints and limitations.", points=["Use if someone asks how the numbers were produced."], plain="—", website="Chapter 7 provenance notes; docs/PROVENANCE.md in the repository.", ask=["—"]), kind="appendix"))
    heading(s, "Provenance and limitations", y=0.62, size=26)
    prov = [f"**Split.** Authors' official patient-level train/test split; validation = patient-disjoint ~10% of training patients. {n(SPL['train_images'])} / {n(SPL['val_images'])} / {n(SPL['test_images'])} images; patient overlap {SPL['patient_overlap']}.",
            f"**USF-MAE.** Official checkpoint usf_mae_500ep.pth (sha256 c659a061…b3a2533) initialised all 19 USF-MAE runs and produced every reconstruction in this deck. The historical 100-epoch checkpoint is not used here.",
            "**Pretraining overlap.** USF-MAE's unlabelled corpus (OpenUS-46) included FETAL_PLANES_DB images, possibly including test images: transductive exposure, not label leakage; stated by the project configuration, not independently verified.",
            "**DenseNet-161 checkpoint.** The original headline best.pt was overwritten by an interrupted rerun. Metrics come from the original run's saved predictions; maps, features and faithfulness use the rerun weights, which reproduce all 5,271 test and 704 validation predictions under the original inference settings but are not byte-identical.",
            "**Seeds.** USF-MAE: 3 seeds at every label fraction. DenseNet-161: 3 seeds below 100%, 1 seed at 100%. ViT-B/16: 100% labels, 1 seed only.",
            "**Statistics.** 95% CIs from a 1,000-resample patient-clustered bootstrap; exact McNemar with Holm correction (image-level pairing). No equivalence or low-label significance tests were run.",
            "**Faithfulness.** 30 random test images, 10 steps, Gaussian-blur baseline, true-label probability; descriptive only."]
    text(s, 0.6, 1.45, 12.1, 5.2, prov, size=12.5, color=INK2, space_after=7)


PM_PARAMS = {m["id"]: f"{m['params'] / 1e6:.1f} M" for m in MAN["models"]}

# ---------------------------------------------------------------- composite figures


def style_chart(ch, colors, legend=True, vmax=None, fmt="0.00", labels=False, gap=80, overlap=0):
    ch.font.name = SANS
    ch.font.size = Pt(12)
    ch.font.color.rgb = rgb(INK2)
    plot = ch.plots[0]
    plot.gap_width = gap
    plot.overlap = overlap
    for s, c in zip(plot.series, colors):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = rgb(c)
        s.format.line.fill.background()
    if labels:
        plot.has_data_labels = True
        dl = plot.data_labels
        dl.number_format = fmt
        dl.number_format_is_linked = False
        dl.position = XL_LABEL_POSITION.OUTSIDE_END
        dl.font.size = Pt(12)
        dl.font.color.rgb = rgb(INK)
    va = ch.value_axis
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = rgb(RULE)
    va.format.line.fill.background()
    va.tick_labels.number_format = fmt
    va.tick_labels.number_format_is_linked = False
    if vmax:
        va.maximum_scale = vmax
        va.minimum_scale = 0
    ca = ch.category_axis
    ca.format.line.color.rgb = rgb(RULE2)
    ca.tick_labels.font.size = Pt(13)
    ch.has_legend = legend
    if legend:
        ch.legend.position = XL_LEGEND_POSITION.TOP
        ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(12)


def split_diagram(s):
    # left: image-level split (leaky)
    for col, (title, sub, leaky) in enumerate([("Image-level split", "same patient on both sides: inflated scores", True),
                                               ("Patient-level split (used here)", "each patient entirely on one side", False)]):
        x0 = 0.6 + col * 6.25
        text(s, x0, 1.95, 5.9, 0.4, title, size=17, color=("8A3B3B" if leaky else "1D7A4E"), bold=True)
        text(s, x0, 2.32, 5.9, 0.35, sub, size=12.5, color=INK3)
        tr = box(s, x0, 2.8, 2.75, 1.85, fill="EEF1F3", line=RULE)
        te = box(s, x0 + 3.1, 2.8, 2.75, 1.85, fill="EEF1F3", line=RULE)
        text(s, x0, 4.68, 2.75, 0.3, "train", size=12, color=INK3, align="c")
        text(s, x0 + 3.1, 4.68, 2.75, 0.3, "test", size=12, color=INK3, align="c")
        pats = ["0072B2", "E69F00", "009E73", "8A3B3B"]
        for p, c in enumerate(pats):
            for k in range(3):
                if leaky:
                    side = (p + k) % 2
                else:
                    side = 0 if p < 2 else 1
                xx = x0 + side * 3.1 + 0.25 + (k * 0.8)
                yy = 2.95 + p * 0.42
                box(s, xx, yy, 0.62, 0.32, fill=c, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.2)
    text(s, 0.6, 5.12, 12.1, 0.3, "Colour = patient. Each rounded block is one image.", size=11, color=INK3)
    facts = [f"**{n(SPL['train_patients'])} / {n(SPL['val_patients'])} / {n(SPL['test_patients'])}** train / validation / test patients · overlap **{SPL['patient_overlap']}**",
             f"Still: **{n(ND['cross_split_pairs'])}** look-alike train/val↔test pairs (loose hash, different patients) · **{MBS['train']['Voluson S10']}** Voluson S10 images in training vs **{n(MBS['test']['Voluson S10'])}** in test"]
    text(s, 0.6, 5.5, 12.1, 0.9, facts, size=14.5, color=INK, space_after=4)


def densenet_diagram(s):
    # convolution: input grid with kernel -> output cell
    gx, gy, c = 0.7, 2.15, 0.27
    for i in range(7):
        for j in range(7):
            box(s, gx + j * c, gy + i * c, c, c, fill=("DCE3E7" if (i + j) % 3 else "C9D3D8"), line="FFFFFF", lw=0.5)
    box(s, gx + 2 * c, gy + 2 * c, 3 * c, 3 * c, line=M["densenet161"], lw=2.25)
    text(s, gx, gy + 7 * c + 0.08, 1.9, 0.3, "input", size=11, color=INK3, align="c")
    arrow(s, gx + 7 * c + 0.12, gy + 3.5 * c, gx + 7 * c + 0.8, gy + 3.5 * c)
    text(s, gx + 7 * c + 0.05, gy + 2.2 * c, 0.9, 0.3, "3×3 filter", size=10.5, color=INK3, align="c")
    ox = gx + 7 * c + 0.95
    for i in range(5):
        for j in range(5):
            box(s, ox + j * c, gy + c + i * c, c, c, fill=("E3EEF6" if (i, j) != (1, 1) else M["densenet161"]), line="FFFFFF", lw=0.5)
    text(s, ox, gy + 6 * c + 0.08, 5 * c, 0.3, "feature map", size=11, color=INK3, align="c")
    equation(s, "eq_conv", 0.7, 4.55, 0.48)
    text(s, 0.7, 5.1, 3.8, 0.9, "One small set of weights w, slid over every position: **locality** and **weight sharing**.", size=13, color=INK2)
    # dense block
    bx, by = 5.15, 2.2
    text(s, bx, 1.82 + 0.05, 4.5, 0.3, "Dense block: every layer gets all earlier outputs", size=13, color=INK, bold=True)
    xs = [bx + k * 1.0 for k in range(5)]
    for k, x in enumerate(xs):
        b = box(s, x, by + 0.75, 0.7, 1.15, fill=(M["densenet161"] if k else "8DB8D6"), shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.15)
        label_in(b, f"x{k}" if k else "x0", size=13, color="FFFFFF", bold=True)
    rail_col = ["8DB8D6", "5B95C2", "2F78AE", "0B5A8F"]
    for a in range(4):
        y = by + 0.6 - a * 0.14
        x1 = xs[a] + 0.35
        x2 = xs[4] + 0.12 + a * 0.15
        arrow(s, x1, by + 0.75, x1, y, color=rail_col[a], width=1.25, head=False)
        arrow(s, x1, y, x2, y, color=rail_col[a], width=1.25, head=False)
        for b_ in range(a + 1, 5):
            xd = xs[b_] + 0.12 + a * 0.15
            arrow(s, xd, y, xd, by + 0.75, color=rail_col[a], width=1.25, head=True)
    equation(s, "eq_dense", bx, 4.55, 0.48)
    text(s, bx, 5.1, 4.4, 0.9, "[ ] = concatenation. Features are reused, not relearned; 4 blocks of 6, 12, 36 and 24 layers.", size=13, color=INK2)
    # head
    hx = 10.25
    text(s, hx, 1.87, 2.6, 0.3, "Classification", size=13, color=INK, bold=True)
    for k, (lab, h) in enumerate([("7×7×2208 maps", 0.55), ("global average pool", 0.45), ("linear layer", 0.45), ("6 class probabilities", 0.45)]):
        y = 2.3 + k * 0.82
        b = box(s, hx, y, 2.45, h, fill=(RAISED if k < 3 else INK), line=RULE2, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.2)
        label_in(b, lab, size=12, color=(INK if k < 3 else "FFFFFF"))
        if k < 3:
            arrow(s, hx + 1.22, y + h, hx + 1.22, y + 0.82)
    text(s, hx, 5.6, 2.6, 0.8, "Pooling before the linear layer is what makes CAM possible (Chapter 5).", size=12, color=INK3)


def vit_diagram(s):
    picture(s, FIG / "vit_patch_grid.png", 0.6, 1.95, 3.95, 3.95)
    text(s, 0.6, 5.95, 3.95, 0.3, "224×224 input · 14×14 = 196 patches of 16×16 px", size=11.5, color=INK3, align="c")
    arrow(s, 4.7, 3.9, 5.25, 3.9)
    # token strip
    tx, ty = 5.4, 2.05
    text(s, tx, ty - 0.1, 3.4, 0.3, "patches → tokens", size=13, color=INK, bold=True)
    cls = box(s, tx, ty + 0.35, 0.62, 0.62, fill=INK, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.15)
    label_in(cls, "CLS", size=10.5, color="FFFFFF", bold=True)
    for j in range(5):
        picture(s, FIG / f"vit_patch_{j + 1}.png", tx + 0.72 * (j + 1), ty + 0.35, 0.62, 0.62)
    text(s, tx + 4.35, ty + 0.4, 0.5, 0.5, "…", size=20, color=INK3)
    for j in range(6):
        x = tx + 0.72 * j
        arrow(s, x + 0.31, ty + 1.02, x + 0.31, ty + 1.4, width=1.1)
        b = box(s, x, ty + 1.45, 0.62, 0.95, fill=("3D4B55" if j == 0 else "F3D9A0"), line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.15)
        p = box(s, x, ty + 2.45, 0.62, 0.22, fill="C9D3D8", shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.4)
        label_in(p, f"p{j}", size=8.5, color=INK)
    text(s, tx, ty + 2.75, 4.6, 0.65, ["linear projection to 768-d token", "+ learned position embedding (grey)"], size=12, color=INK2)
    arrow(s, 9.85, 3.9, 10.25, 3.9)
    # transformer stack
    sx = 10.35
    for k in range(4):
        b = box(s, sx + k * 0.06, 2.0 + k * 0.06, 2.35, 2.75, fill=("FFFFFF" if k < 3 else "FBF0D8"), line=M["vit_b16"], lw=1.0, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
    label_in(b, ["Transformer", "block × 12", "", "self-attention", "+ MLP"], size=13, color=INK)
    arrow(s, sx + 1.2, 4.95, sx + 1.2, 5.3)
    h = box(s, sx + 0.1, 5.35, 2.3, 0.5, fill=INK, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.2)
    label_in(h, "[CLS] → 6 classes", size=12, color="FFFFFF")
    text(s, 5.4, 5.55, 4.6, 0.8, "No convolution: patches relate only through attention, which is learned.", size=13, color=INK2)


def mae_pipeline(s):
    steps = [("ImageNet MAE", "self-supervised start\n(photographs)", "DCE3E7", INK),
             ("USF-MAE pretraining", "~370k unlabelled ultrasound\nimages · 500 epochs", "CCE9DF", INK),
             ("Keep encoder", "discard the decoder", "CCE9DF", INK),
             ("Fine-tune", "encoder + new 6-way head\non labelled FETAL_PLANES_DB", M["usf_mae"], "FFFFFF")]
    for i, (h_, b_, fill, fg) in enumerate(steps):
        x = 0.6 + i * 3.1
        b = box(s, x, 2.15, 2.75, 2.3, fill=fill, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.07)
        text(s, x + 0.2, 2.35, 2.35, 0.5, h_, size=17, color=fg, bold=True)
        text(s, x + 0.2, 2.95, 2.35, 1.4, b_.split("\n"), size=13.5, color=fg)
        text(s, x + 0.2, 4.0, 2.35, 0.35, ["no labels", "no labels", "", "labels"][i], size=11.5, color=(fg if i == 3 else INK3), bold=True)
        if i < 3:
            arrow(s, x + 2.78, 3.3, x + 3.07, 3.3, width=2)


def workflow_diagram(s):
    st = [("Pretrained encoder", "+ new 6-way head", RAISED, INK),
          ("Train", f"{n(SPL['train_images'])} images · {n(SPL['train_patients'])} patients", "E3EEF6", INK),
          ("Select", f"best epoch by validation macro-F1\n{n(SPL['val_images'])} images · {SPL['val_patients']} patients", "EEE6D2", INK),
          ("Test once", f"{n(SPL['test_images'])} images · {n(SPL['test_patients'])} patients", INK, "FFFFFF")]
    for i, (h_, b_, fill, fg) in enumerate(st):
        x = 0.6 + i * 3.1
        box(s, x, 2.0, 2.75, 2.45, fill=fill, line=RULE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.07)
        text(s, x + 0.2, 2.2, 2.35, 0.5, h_, size=19, color=fg, font=SERIF)
        text(s, x + 0.2, 2.85, 2.35, 1.5, b_.split("\n"), size=13, color=(fg if fg == "FFFFFF" else INK2))
        if i < 3:
            arrow(s, x + 2.78, 3.22, x + 3.07, 3.22, width=2)
    text(s, 3.7, 4.1, 6.0, 0.3, "class-weighted loss · augmentation · early stopping", size=11, color=INK3)


def forest_plot(s):
    rows = [("densenet161", "DenseNet-161", "single run", lambda k: (PM["densenet161"][k], *PM["densenet161"]["ci95"][k]), "ci"),
            ("vit_b16", "ViT-B/16", "single run", lambda k: (PM["vit_b16"][k], *PM["vit_b16"]["ci95"][k]), "ci"),
            ("usf_mae", "USF-MAE", "seed 0", lambda k: (PM["usf_mae"][k], *PM["usf_mae"]["ci95"][k]), "ci"),
            ("usf_mae", "USF-MAE", "3 seeds: mean ± sd", lambda k: (SEED["mean"][k], SEED["mean"][k] - SEED["sd"][k], SEED["mean"][k] + SEED["sd"][k]), "sd")]
    lo, hi = 0.925, 0.960
    for p, (key, title) in enumerate((("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"))):
        x0, w = 3.2 + p * 4.95, 3.7
        text(s, x0, 1.95, w, 0.35, title, size=16, color=INK, bold=True)
        for t in (0.93, 0.94, 0.95, 0.96):
            xx = x0 + (t - lo) / (hi - lo) * w
            box(s, xx, 2.4, 0.01, 3.05, fill=RULE)
            text(s, xx - 0.4, 5.48, 0.8, 0.3, f"{t:.2f}", size=11, color=INK3, align="c")
        for r, (m, nm, sub, fn, kind) in enumerate(rows):
            y = 2.65 + r * 0.72
            est, a, b = fn(key)
            xa, xb, xe = (x0 + (v - lo) / (hi - lo) * w for v in (a, b, est))
            box(s, xa, y + 0.13, xb - xa, 0.045, fill=M[m])
            shape = MSO_SHAPE.OVAL if kind == "ci" else MSO_SHAPE.DIAMOND
            box(s, xe - 0.1, y + 0.055, 0.2, 0.2, fill=(M[m] if kind == "ci" else RAISED), line=M[m], lw=1.75, shape=shape)
            text(s, x0 + w + 0.1, y - 0.02, 0.85, 0.35, f4(est), size=11.5, color=INK2)
            if p == 0:
                text(s, 0.6, y - 0.05, 2.5, 0.3, nm, size=14.5, color=M_TEXT[m], bold=True)
                text(s, 0.6, y + 0.22, 2.5, 0.3, sub, size=11, color=INK3)
    text(s, 0.6, 5.85, 12.1, 0.75, ["Pairwise exact McNemar: " + " · ".join(f"{NAME[c['a']]}–{NAME[c['b']]} p = {c['p_value']:.2f}" for c in EV["comparisons"]) + " · **Holm-adjusted p = 1** for all.",
                                    "Not significantly different ≠ equivalent: no equivalence test was run. Diamond row shows seed spread (sd), not a CI."], size=12.5, color=INK2, space_after=3)


def confusion(s, model):
    cm_ = PM[model]["confusion"]
    rows = [["Expert ↓ / predicted →"] + SHORT]
    fills, colors, bolds = [[None] * 7], [[None] * 7], [[None] * 7]
    for i, r in enumerate(cm_):
        tot = sum(r)
        row, fr, co, bo = [SHORT[i]], [None], [None], [True]
        for j, v in enumerate(r):
            f = v / tot
            row.append(f"{v:,}")
            fr.append(mix("FFFFFF", "00775A", min(1, f ** 0.6)) if v else None)
            co.append("FFFFFF" if f > 0.35 else (INK if v else "B5C0C6"))
            bo.append(i == j)
        rows.append(row); fills.append(fr); colors.append(co); bolds.append(bo)
    tb = table(s, 0.6, 1.95, 7.6, 4.55, rows, col_w=[2.2] + [0.9] * 6, size=14, fills=fills, colors=colors, bold_cells=bolds, row_h=0.64)
    for c in range(7):
        for r_ in tb.cell(0, c).text_frame.paragraphs[0].runs:
            r_.font.size = Pt(11.5)
    other = cm_[5]
    into_other = [cm_[i][5] for i in range(5)]
    errs = sum(sum(r) for r in cm_) - sum(cm_[i][i] for i in range(6))
    from_other = sum(other) - other[5]
    text(s, 8.6, 2.0, 4.2, 4.4, [f"**{errs}** errors in {n(sum(sum(r) for r in cm_))} test images.",
                                 f"**{from_other}** are 'Other' images given a specific plane (most often {SHORT[max(range(5), key=lambda j: other[j])].lower()}: {max(other[:5])}).",
                                 f"**{sum(into_other)}** are specific planes called 'Other'.",
                                 f"Only **{errs - from_other - sum(into_other)}** errors are between two specific planes.",
                                 "Same pattern for DenseNet-161 and ViT-B/16 (website, Chapter 4)."], size=15, color=INK2, space_after=10)


def gradcam_diagram(s):
    y = 2.05
    for k, col in enumerate(["5B7FA3", "7FA3C2", "A6C4DB"]):
        b = box(s, 0.7 + k * 0.18, y + k * 0.18, 1.6, 1.6, fill=col, line="FFFFFF")
    text(s, 0.6, y + 2.05, 2.4, 0.4, "feature maps A¹…Aᴷ", size=12, color=INK3, align="c")
    arrow(s, 2.8, y + 1.0, 3.5, y + 1.0)
    w_ = box(s, 3.6, y + 0.45, 2.1, 1.1, fill="EEE6D2", shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.15)
    label_in(w_, ["importance αₖ", "for class c"], size=13, color=INK)
    text(s, 3.5, y + 1.65, 2.3, 0.5, "from gradients of the class score", size=11, color=INK3, align="c")
    arrow(s, 5.8, y + 1.0, 6.5, y + 1.0)
    sm = box(s, 6.6, y + 0.45, 2.1, 1.1, fill="EEE6D2", shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.15)
    label_in(sm, ["Σ αₖ Aᵏ", "then ReLU"], size=13, color=INK)
    arrow(s, 8.8, y + 1.0, 9.5, y + 1.0)
    picture(s, FIG / "xai_densenet_gradcam.png", 9.6, y - 0.05, 3.1, 2.15)
    text(s, 9.6, y + 2.12, 3.1, 0.35, "upsampled 7×7 map (real, DenseNet)", size=11, color=INK3, align="c")


def perturbation_case(s, x, title, d):
    text(s, x, 1.95, 5.95, 0.35, title, size=16, color=INK, bold=True)
    tag = "femur" if "Femur" in title else "abdomen"
    picture(s, FIG / f"detective_{tag}_orig.png", x, 2.35, 2.9, 2.0)
    picture(s, FIG / f"detective_{tag}_blur.png", x + 3.05, 2.35, 2.9, 2.0)
    text(s, x, 4.37, 2.9, 0.3, "original", size=11, color=INK3, align="c")
    text(s, x + 3.05, 4.37, 2.9, 0.3, "DenseNet Grad-CAM top 10% blurred", size=11, color=INK3, align="c")
    true_c = SHORT[d["label"]]
    text(s, x, 4.72, 5.95, 0.3, f"P({true_c}) before → after · predicted class after", size=11, color=INK3)
    for i, m in enumerate(("densenet161", "vit_b16", "usf_mae")):
        y = 5.05 + i * 0.5
        b, a = d["before"][m]["p_true"], d["after"][m]["p_true"]
        flip = d["after"][m]["pred"] != d["label"]
        text(s, x, y, 1.6, 0.4, NAME[m], size=12.5, color=M_TEXT[m], bold=True, anchor="m")
        box(s, x + 1.65, y + 0.1, 2.2, 0.2, fill="E6EAED")
        box(s, x + 1.65, y + 0.1, 2.2 * b, 0.2, fill=RULE2)
        box(s, x + 1.65, y + 0.1, max(0.02, 2.2 * a), 0.2, fill=M[m])
        text(s, x + 3.95, y, 2.0, 0.4, f"{b:.0%} → {a:.0%} · {SHORT[d['after'][m]['pred']]}", size=12, color=("8A3B3B" if flip else INK2), bold=flip, anchor="m")


MARGIN = 0.02


def faith_verdict(r, rb):
    """Per-metric comparison with the model's random map; differences within MARGIN count as ≈."""
    def word(diff):  # diff > 0 means better than random
        return "better" if diff > MARGIN else ("worse" if diff < -MARGIN else "≈")
    wd = word(rb["deletion_auc_mean"] - r["deletion_auc_mean"])
    wi = word(r["insertion_auc_mean"] - rb["insertion_auc_mean"])
    if wd == wi:
        v = {"better": "better on both", "worse": "worse on both", "≈": "≈ random"}[wd]
    else:
        v = f"{wd} del. · {wi} ins."
    ws = {wd, wi}
    if "worse" not in ws and "better" in ws:
        return v, "1D7A4E", (None if ws == {"better"} else "EEF5EF")
    if "better" not in ws and "worse" in ws:
        return v, "8A3B3B", "F6E6E6"
    return v, "8A6D1F", "FBF3E2"


def faithfulness_table(s):
    rnd = FA["random_baseline"]
    rows = [["Model", "Method (layer)", "Deletion AUC ↓", "Insertion AUC ↑", "Compared with random"]]
    fills, colors, bolds = [[None] * 5], [[None] * 5], [[None] * 5]
    order = ["densenet161", "vit_b16", "usf_mae"]
    for m in order:
        rr = sorted([r for r in FA["rows"] if r["model"] == m], key=lambda r: (r["method"] == "random",))
        for r in rr:
            if r["method"] == "random":
                verdict, col, fill = "reference", INK3, "EEF1F3"
            else:
                verdict, col, fill = faith_verdict(r, rnd[m])
            lay = {"act": "final", "final": "all blocks", "block11": "block 11"}.get(r["layer"], r["layer"])
            meth = r["method_label"].replace(" (baseline)", "") + ("" if r["method"] == "random" else f" ({lay})")
            rows.append([NAME[m], meth, f"{r['deletion_auc_mean']:.2f} ± {r['deletion_auc_sd']:.2f}", f"{r['insertion_auc_mean']:.2f} ± {r['insertion_auc_sd']:.2f}", verdict])
            fills.append([fill] * 5)
            colors.append([M_TEXT[m], None, None, None, col])
            bolds.append([True, False, False, False, r["method"] != "random"])
    table(s, 0.6, 1.8, 12.1, 4.2, rows, col_w=[1.9, 4.4, 1.9, 1.9, 2.0], size=10.5, fills=fills, colors=colors, bold_cells=bolds, row_h=0.29, tight=True)
    text(s, 0.6, 6.32, 12.1, 0.35, f"n = {FA['n_images']} images · {FA['steps']} steps · blurred-image baseline · true-label probability · sds up to 0.29: most rankings are not statistically supported. ≈ = within 0.02 of random (descriptive).",
         size=10.5, color=INK3)


# ---------------------------------------------------------------- presenter guide

def write_guide():
    lines = ["# Presenter guide: From Pixels to Predictions",
             "",
             f"*{EVENT}* · {AUTHORS.replace('  ·  ', ' and ')} · 120 minutes including a 10-minute break.",
             "",
             "Generated by `presentation/src/build_deck.py` from the same specification as the slides; the speaker notes inside the .pptx are identical to the per-slide sections below. Edit the build script, not this file, if you want the two to stay in sync.",
             "",
             f"Companion website: {url_text()}",
             "",
             "## Run of show",
             "",
             "| Slide | Clock (min) | Minutes | Chapter | Slide title | Website activity cue |",
             "|---|---|---|---|---|---|"]
    t = 0.0
    for i, sp in enumerate(SPECS, 1):
        if sp.kind == "appendix":
            lines.append(f"| {i} | (appendix) | — | — | {sp.title} | — |")
            continue
        cue = (sp.notes.cue or ("Activity: " + (sp.activity or ""))) if sp.kind == "activity" else (sp.notes.cue or "")
        lines.append(f"| {i} | {t:g}–{t + sp.minutes:g} | {sp.minutes:g} | {CH[sp.chapter][1]} | {sp.title} | {cue} |")
        t += sp.minutes
    lines += ["", f"**Total planned time: {t:g} minutes.**", "", "### Chapter windows", "", "| Chapter | Window | Planned minutes |", "|---|---|---|"]
    for c in CHAPTERS:
        mins = sum(sp.minutes for sp in SPECS if sp.chapter == c[0] and sp.kind != "appendix")
        lines.append(f"| {c[2]} | {c[3]}–{c[4]} | {mins:g} |")
    lines += ["", "## Discussion prompts at a glance", ""]
    for i, sp in enumerate(SPECS, 1):
        if sp.kind == "appendix":
            continue
        lines.append(f"- **Slide {i}, {sp.title}:** " + " / ".join(sp.notes.ask))
    lines += ["", "## Slide-by-slide notes", ""]
    for i, sp in enumerate(SPECS, 1):
        lines += [f"### Slide {i}: {sp.title}", "", "```text", render_notes(sp, i), "```", ""]
    OUT_GUIDE.write_text("\n".join(lines), encoding="utf-8")
    return t


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="public workshop URL (generates QR codes); omit for placeholders")
    ap.add_argument("--out", default=str(OUT_PPTX))
    args = ap.parse_args()
    if args.url:
        if re.search(r"localhost|127\.0\.0\.1|\[::1\]", args.url):
            raise SystemExit("Refusing a localhost URL for participant-facing slides.")
        import qrcode
        QR.mkdir(parents=True, exist_ok=True)
        img = qrcode.make(args.url, border=2, box_size=12)
        STATE["qr"] = QR / "workshop_url_qr.png"
        img.save(STATE["qr"])
        STATE["url"] = args.url
    prs = Presentation()
    prs.slide_width, prs.slide_height = Emu(12192000), Emu(6858000)  # exact 16:9 (13.333 x 7.5 in)
    prs.core_properties.title = f"{TITLE}: {SUBTITLE}"
    prs.core_properties.author = "Kevin Dick; Youssef Megahed"
    prs.core_properties.subject = EVENT
    build(prs)
    for sl, sp, dk, idx in STATE["slides"]:
        chrome(sl, sp, dk, idx)
    total = write_guide()
    out = Path(args.out)
    prs.save(out)
    print(f"saved {out} · {len(prs.slides)} slides · planned {total:g} min · guide {OUT_GUIDE}")
    assert total == 120, total


if __name__ == "__main__":
    main()
