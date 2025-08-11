# scripts/prepare_ccpd_yolo.py
"""
将 CCPD2019 各子集转换为 YOLOv8 检测格式：
- 解析文件名中的车牌外接矩形 (xmin,ymin,xmax,ymax)
- 生成 labels/*.txt (class cx cy w h, 归一化)
- 划分 train/val/test

示例：
python scripts/prepare_ccpd_yolo.py --ccpd_root /path/to/CCPD2019 \
  --out_root data/lpr/ccpd_yolo \
  --subsets ccpd_base ccpd_blur ccpd_db \
  --val_ratio 0.05 --test_ratio 0.05
"""
import argparse
import random
import re
from pathlib import Path
import shutil
import cv2

random.seed(42)

# 匹配 ...-x1&y1_x2&y2-... 片段
BBOX_REGEX = re.compile(r"-(\d+)&(\d+)_(\d+)&(\d+)-")

def parse_bbox_from_name(name: str):
    m = BBOX_REGEX.search(name)
    if not m:
        return None
    x1, y1, x2, y2 = map(int, m.groups())
    xmin, xmax = sorted([x1, x2])
    ymin, ymax = sorted([y1, y2])
    return xmin, ymin, xmax, ymax

def yolo_line_from_bbox(img_w, img_h, bbox, cls_id=0):
    xmin, ymin, xmax, ymax = bbox
    cx = (xmin + xmax) / 2.0 / img_w
    cy = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    w = min(max(w, 1e-6), 1.0)
    h = min(max(h, 1e-6), 1.0)
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"

def collect_images(ccpd_root: Path, subsets):
    imgs = []
    for sub in subsets:
        d = ccpd_root / sub
        if not d.exists():
            print(f"[WARN] subset not found: {d}")
            continue
        imgs.extend(list(d.rglob("*.jpg")))
    return imgs

def split_list(items, val_ratio=0.05, test_ratio=0.05):
    random.shuffle(items)
    n = len(items)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    val = items[:n_val]
    test = items[n_val:n_val+n_test]
    train = items[n_val+n_test:]
    return train, val, test

def convert(ccpd_root: Path, out_root: Path, subsets, val_ratio=0.05, test_ratio=0.05):
    images_dir = out_root / "images"
    labels_dir = out_root / "labels"
    for split in ["train", "val", "test"]:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)

    imgs = collect_images(ccpd_root, subsets)
    train, val, test = split_list(imgs, val_ratio, test_ratio)
    splits = {"train": train, "val": val, "test": test}

    skipped = 0
    total = 0

    for split, paths in splits.items():
        for src_img in paths:
            total += 1
            rel_name = src_img.name
            bbox = parse_bbox_from_name(rel_name)
            if bbox is None:
                skipped += 1
                continue

            img = cv2.imread(str(src_img))
            if img is None:
                skipped += 1
                continue
            h, w = img.shape[:2]

            yolo_line = yolo_line_from_bbox(w, h, bbox, cls_id=0)
            dst_img = images_dir / split / rel_name
            dst_lbl = labels_dir / split / (src_img.stem + ".txt")

            shutil.copyfile(src_img, dst_img)
            with open(dst_lbl, "w", encoding="utf-8") as f:
                f.write(yolo_line)

    print(f"Done. total={total}, skipped(no bbox or read fail)={skipped}")
    print(f"images: {images_dir}")
    print(f"labels: {labels_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ccpd_root", type=str, required=True, help="CCPD2019 根目录，包含各子集")
    ap.add_argument("--out_root", type=str, default="data/lpr/ccpd_yolo", help="输出 YOLO 数据根目录")
    ap.add_argument("--subsets", nargs="+", default=["ccpd_base"], help="参与转换的子集列表")
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--test_ratio", type=float, default=0.05)
    args = ap.parse_args()

    convert(Path(args.ccpd_root), Path(args.out_root), args.subsets, args.val_ratio, args.test_ratio)

if __name__ == "__main__":
    main()
