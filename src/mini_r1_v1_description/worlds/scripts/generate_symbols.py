#!/usr/bin/env python3
"""
generate_symbols.py

Render a set of distinct geometric symbols as PNGs for use as floor decals.

Each symbol is a high-contrast black-on-white 512x512 image written to
worlds/symbols/symbol_<id>.png. Symbol 1 (forward arrow) is expected to
be pre-existing and is not overwritten.
"""
import os
import cv2
import numpy as np


SIZE = 512
BG = (255, 255, 255)    # white background
FG = (30, 30, 30)       # near-black ink
MARGIN = 60


def canvas():
    img = np.full((SIZE, SIZE, 3), BG, dtype=np.uint8)
    return img


def triangle():
    img = canvas()
    pts = np.array([
        [SIZE // 2, MARGIN],
        [MARGIN, SIZE - MARGIN],
        [SIZE - MARGIN, SIZE - MARGIN],
    ], np.int32)
    cv2.fillPoly(img, [pts], FG)
    return img


def plus():
    img = canvas()
    bar_half = 55
    # vertical bar
    cv2.rectangle(
        img,
        (SIZE // 2 - bar_half, MARGIN),
        (SIZE // 2 + bar_half, SIZE - MARGIN),
        FG, -1,
    )
    # horizontal bar
    cv2.rectangle(
        img,
        (MARGIN, SIZE // 2 - bar_half),
        (SIZE - MARGIN, SIZE // 2 + bar_half),
        FG, -1,
    )
    return img


def x_mark():
    img = canvas()
    thick = 80
    cv2.line(img, (MARGIN + 20, MARGIN + 20),
             (SIZE - MARGIN - 20, SIZE - MARGIN - 20), FG, thick)
    cv2.line(img, (SIZE - MARGIN - 20, MARGIN + 20),
             (MARGIN + 20, SIZE - MARGIN - 20), FG, thick)
    return img


def diamond():
    img = canvas()
    pts = np.array([
        [SIZE // 2, MARGIN],
        [SIZE - MARGIN, SIZE // 2],
        [SIZE // 2, SIZE - MARGIN],
        [MARGIN, SIZE // 2],
    ], np.int32)
    cv2.fillPoly(img, [pts], FG)
    return img


def circle_dot():
    img = canvas()
    c = (SIZE // 2, SIZE // 2)
    cv2.circle(img, c, 190, FG, 28)  # outer ring
    cv2.circle(img, c, 55, FG, -1)   # inner dot
    return img


def chevron():
    """Double-chevron pointing up."""
    img = canvas()
    thick = 60
    mid = SIZE // 2
    for off in (0, 120):  # two stacked chevrons
        top_y = MARGIN + off
        cv2.line(img, (MARGIN + 40, top_y + 180),
                 (mid, top_y), FG, thick)
        cv2.line(img, (SIZE - MARGIN - 40, top_y + 180),
                 (mid, top_y), FG, thick)
    return img


SYMBOLS = {
    2: ('triangle', triangle),
    3: ('plus', plus),
    4: ('x_mark', x_mark),
    5: ('diamond', diamond),
    6: ('circle_dot', circle_dot),
    7: ('chevron', chevron),
}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.abspath(os.path.join(here, '..', 'symbols'))
    os.makedirs(out_dir, exist_ok=True)

    for idx, (name, fn) in SYMBOLS.items():
        path = os.path.join(out_dir, f'symbol_{idx}.png')
        cv2.imwrite(path, fn())
        print(f'[{name}] {path}')


if __name__ == '__main__':
    main()
