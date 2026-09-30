import random
from pathlib import Path

# -------------------------------------------------------------
# Generates res/header.svg: the square-pattern page header.
# Change SEED for a different layout; COLOR should match the
# --auxiliary colour in style.css.
# -------------------------------------------------------------

SEED = 572
W, H, BASE = 1760, 320, 160
COLOR = "#123B57"
OUTPUT = Path(__file__).resolve().parent.parent / "res" / "header.svg"

random.seed(SEED)
squares = []

def split(x, y, s):
    # Chance of subdividing a square of each size into 4 smaller ones
    p = {160: 0.6, 80: 0.45, 40: 0.25}.get(s, 0)
    if random.random() < p:
        h = s // 2
        for dx in (0, h):
            for dy in (0, h):
                split(x + dx, y + dy, h)
    elif random.random() < 0.7:
        squares.append((x, y, s))

# Touching squares: a quadtree subdivision of a grid (neighbours share edges)
for gx in range(0, W, BASE):
    for gy in range(0, H, BASE):
        split(gx, gy, BASE)

# Overlapping squares scattered on top, snapped to a 10px grid
for _ in range(22):
    s = random.choice([50, 60, 70, 90, 110, 130])
    squares.append((random.randrange(0, W - s, 10), random.randrange(0, H - s, 10), s))

# Offset by 0.5px so 1px strokes land on whole pixels and render crisply
rects = "\n".join(
    f'    <rect x="{x + 0.5}" y="{y + 0.5}" width="{s - 1}" height="{s - 1}"/>'
    for x, y, s in squares
)
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <g fill="none" stroke="{COLOR}" stroke-width="1">
{rects}
  </g>
</svg>
'''

OUTPUT.write_text(svg, encoding="utf-8", newline="\n")
print(f"Wrote {len(squares)} squares to {OUTPUT}")
