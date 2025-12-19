#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Replica-style subsampling with TEXT-ONLY COLMAP sparse filtering.
NO external imports, NO read_write_model.

Works with:
  sparse/0/images.txt
  sparse/0/points3D.txt
  sparse/0/cameras.txt

Naming:
  images/frame000123.jpg
  depth/depth000123.png
"""

import argparse
import shutil
import re
from pathlib import Path
from typing import List, Dict, Set

RGB_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".exr"}

# ---------------- utils ----------------

def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]

def extract_int_id(name: str) -> int:
    m = re.search(r"(\d+)$", Path(name).stem)
    if not m:
        raise RuntimeError(f"Cannot extract id from {name}")
    return int(m.group(1))

# ---------------- images ----------------

def read_images_txt(path: Path):
    lines = path.read_text().splitlines()
    items = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("#") or not lines[i].strip():
            i += 1
            continue
        header = lines[i]
        pts = lines[i+1]
        parts = header.split()
        image_id = int(parts[0])
        image_name = parts[9]
        items.append((image_id, header, pts, image_name))
        i += 2
    return items

def write_images_txt(path: Path, items):
    with path.open("w") as f:
        f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID IMAGE_NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for _, h, p, _ in items:
            f.write(h + "\n")
            f.write(p + "\n")

# ---------------- points3D ----------------

def read_points3D_txt(path: Path):
    pts = {}
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        pid = int(parts[0])
        xyzrgb = parts[1:7]
        error = parts[7]
        image_ids = list(map(int, parts[8::2]))
        point2D_idxs = list(map(int, parts[9::2]))
        pts[pid] = (xyzrgb, error, image_ids, point2D_idxs)
    return pts

def write_points3D_txt(path: Path, pts):
    with path.open("w") as f:
        f.write("# POINT3D_ID X Y Z R G B ERROR IMAGE_ID POINT2D_IDX ...\n")
        for pid, (xyzrgb, error, image_ids, point2D_idxs) in pts.items():
            line = [str(pid)] + xyzrgb + [error]
            for iid, idx in zip(image_ids, point2D_idxs):
                line += [str(iid), str(idx)]
            f.write(" ".join(line) + "\n")

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--images_dir", default="images")
    ap.add_argument("--depth_dir", default="depth")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    # ---- list images ----
    imgs = sorted(
        [p for p in (src / args.images_dir).iterdir()
         if p.suffix.lower() in RGB_EXTS],
        key=lambda p: natural_key(p.name)
    )
    kept_imgs = imgs[::args.step]
    kept_names = {p.name for p in kept_imgs}

    print(f"[RGB] keep {len(kept_imgs)} / {len(imgs)}")

    # ---- prepare dst ----
    if dst.exists():
        shutil.rmtree(dst)
    (dst / args.images_dir).mkdir(parents=True)
    (dst / args.depth_dir).mkdir(parents=True)
    (dst / "sparse" / "0").mkdir(parents=True)

    # ---- copy images & depth ----
    for p in kept_imgs:
        shutil.copy2(p, dst / args.images_dir / p.name)
        dp = src / args.depth_dir / f"depth{extract_int_id(p.name):06d}.png"
        if dp.exists():
            shutil.copy2(dp, dst / args.depth_dir / dp.name)

    # ---- sparse: images.txt ----
    images_txt = src / "sparse" / "0" / "images.txt"
    points_txt = src / "sparse" / "0" / "points3D.txt"
    cameras_txt = src / "sparse" / "0" / "cameras.txt"

    imgs_items = read_images_txt(images_txt)
    kept_items = [it for it in imgs_items if it[3] in kept_names]

    write_images_txt(dst / "sparse" / "0" / "images.txt", kept_items)

    kept_image_ids = {it[0] for it in kept_items}

    # ---- sparse: points3D.txt ----
    pts = read_points3D_txt(points_txt)
    pts_f = {}

    for pid, (xyzrgb, error, image_ids, point2D_idxs) in pts.items():
        new_iids = []
        new_idxs = []
        for iid, idx in zip(image_ids, point2D_idxs):
            if iid in kept_image_ids:
                new_iids.append(iid)
                new_idxs.append(idx)
        if len(new_iids) >= 2:
            pts_f[pid] = (xyzrgb, error, new_iids, new_idxs)

    write_points3D_txt(dst / "sparse" / "0" / "points3D.txt", pts_f)

    # ---- copy cameras.txt ----
    shutil.copy2(cameras_txt, dst / "sparse" / "0" / "cameras.txt")

    print(f"[sparse] images {len(kept_items)}, points3D {len(pts_f)}")
    print("[DONE]", dst)

if __name__ == "__main__":
    main()
