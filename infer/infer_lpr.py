from utils.paths import abspath, DEVICE, RUNS_DIR
import argparse
import os
import sys
import csv
import json
from pathlib import Path

import cv2
import torch

# 允许作为独立脚本运行
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models.lpr_model import LicensePlateRecognizer


def is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def draw_bbox(img, bbox, text=None):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if text:
        cv2.putText(img, text, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img


def process_image(lpr: LicensePlateRecognizer, img_path: Path, save_vis_dir: Path = None):
    res = lpr.predict(str(img_path))
    vis_img = cv2.imread(str(img_path))
    if res and res.get("bbox"):
        label = f"{res.get('plate_text','')} {res.get('score',0):.2f}"
        vis_img = draw_bbox(vis_img, res["bbox"], label)
    if save_vis_dir is not None:
        save_vis_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_vis_dir / img_path.name
        cv2.imwrite(str(out_path), vis_img)
    return res


def process_dir(lpr: LicensePlateRecognizer, src: Path, save_dir: Path, csv_name: str = "results.csv"):
    images = [p for p in src.rglob("*") if p.is_file() and is_image(p)]
    results = []
    vis_dir = save_dir / "vis"
    for p in images:
        res = process_image(lpr, p, vis_dir)
        if res is None:
            res = {"image_path": str(p), "plate_text": "", "score": 0.0, "bbox": []}
        results.append(res)
    # 写 CSV
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / csv_name
    with open(csv_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "plate_text", "score", "x1", "y1", "x2", "y2"])
        for r in results:
            bbox = r.get("bbox") or ["", "", "", ""]
            writer.writerow([
                r.get("image_path", ""), r.get("plate_text", ""), f"{r.get('score',0):.4f}",
                *(bbox if len(bbox)==4 else ["", "", "", ""])])
    # 也保存 JSON
    with open(save_dir / "results.json", "w", encoding="utf-8") as jf:
        json.dump(results, jf, ensure_ascii=False, indent=2)
    return csv_path


def process_video(lpr: LicensePlateRecognizer, video_path: Path, save_dir: Path = None, write_video: bool = True):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return
    writer = None
    if write_video and save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(save_dir / (video_path.stem + "_vis.mp4")), fourcc, fps, (w, h))
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        # 临时保存一帧到内存路径并调用内部流程（也可改造成直接传 ndarray）
        # 这里直接复用 detect + ocr：
        res = lpr.det_model(frame)
        boxes = res[0].boxes
        if boxes.shape[0] > 0:
            # 只展示分数最高的一个
            bbox = boxes[0].xyxy[0].cpu().numpy().astype(int).tolist()
            x1, y1, x2, y2 = bbox
            crop = frame[y1:y2, x1:x2]
            ocr_res = lpr.recognize_plate(crop)
            text = f"{ocr_res[0]} {ocr_res[1]:.2f}" if ocr_res[0] else ""
            draw_bbox(frame, bbox, text)
        if writer is not None:
            writer.write(frame)
        # 若需要实时可视化，取消注释：
        # cv2.imshow('LPR', frame)
        # if cv2.waitKey(1) & 0xFF == 27:
        #     break
    cap.release()
    if writer is not None:
        writer.release()


def parse_args():
    parser = argparse.ArgumentParser(description="License Plate Recognition Inference")
    parser.add_argument("--source", type=str, required=True,
                        help="image file | directory | video file")
    parser.add_argument("--det_model", type=str, default="yolov8n.pt",
                        help="YOLOv8 detection model path (trained for plates)")
    parser.add_argument("--ocr_lang", type=str, default="ch", help="PaddleOCR language model, e.g. ch")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--save_dir", type=str, default="runs/lpr_infer", help="output directory")
    parser.add_argument("--mode", type=str, choices=["auto", "image", "dir", "video"], default="auto",
                        help="auto-detect source type or force a mode")
    return parser.parse_args()


def main():
    args.det_model = str(abspath(args.det_model))
    args.save_dir  = str(abspath(args.save_dir or (RUNS_DIR / 'lpr_infer')))
    device = args.device if args.device else DEVICE
    lpr = LicensePlateRecognizer(det_model_path=args.det_model, ocr_lang=args.ocr_lang, device=device)
    mode = args.mode
    if mode == "auto":
        if src.is_dir():
            mode = "dir"
        elif is_image(src):
            mode = "image"
        else:
            mode = "video"

    if mode == "image":
        vis_dir = save_dir / "vis"
        res = process_image(lpr, src, vis_dir)
        print(json.dumps(res or {}, ensure_ascii=False, indent=2))
    elif mode == "dir":
        csv_path = process_dir(lpr, src, save_dir)
        print(f"Saved results to: {csv_path}")
    else:  # video
        process_video(lpr, src, save_dir)
        print(f"Video processed. Outputs (if enabled) saved to: {save_dir}")


if __name__ == "__main__":
    main()