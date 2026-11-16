"""Tile preview/slides/slide_NN.png into preview/contact_sheet.png (5 columns, numbered)."""
from pathlib import Path

from PIL import Image, ImageDraw

PREV = Path(__file__).resolve().parents[1] / "preview"
files = sorted((PREV / "slides").glob("slide_*.png"))
W, H, COLS, PAD = 480, 270, 5, 28
rows = (len(files) + COLS - 1) // COLS
sheet = Image.new("RGB", (COLS * (W + 12) + 12, rows * (H + PAD + 12) + 12), "white")
d = ImageDraw.Draw(sheet)
for i, f in enumerate(files):
    x, y = 12 + (i % COLS) * (W + 12), 12 + (i // COLS) * (H + PAD + 12)
    sheet.paste(Image.open(f).convert("RGB").resize((W, H), Image.LANCZOS), (x, y + PAD))
    d.text((x, y + 6), f"{i + 1}", fill=(24, 35, 43))
    d.rectangle([x - 1, y + PAD - 1, x + W, y + PAD + H], outline=(181, 192, 198))
sheet.save(PREV / "contact_sheet.png", optimize=True)
print("contact sheet:", len(files), "slides")
