import torch
import cv2
from paddleocr import PaddleOCR
from ultralytics import YOLO
import numpy as np


class LicensePlateRecognizer:
    def __init__(self, det_model_path='yolov8n.pt', ocr_lang='ch', device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.det_model = YOLO(det_model_path).to(self.device)
        self.ocr = PaddleOCR(use_angle_cls=True, lang=ocr_lang)

    def detect_plate(self, image):
        """
        使用 YOLOv8 检测车牌位置
        返回 [x1, y1, x2, y2] 的列表（只取最高分的一个）
        """
        results = self.det_model(image)
        boxes = results[0].boxes
        if boxes.shape[0] == 0:
            return None
        # 取置信度最高的一个车牌框
        best_box = boxes[0].xyxy[0].cpu().numpy().astype(int)
        return best_box  # [x1, y1, x2, y2]

    def recognize_plate(self, plate_crop):
        """
        使用 OCR 识别裁剪后的车牌区域
        返回：车牌文本和置信度
        """
        result = self.ocr.ocr(plate_crop, cls=True)
        if not result or not result[0]:
            return "", 0.0
        text, score = result[0][0][1]
        return text, score

    def predict(self, image_path):
        """
        整体流程：检测车牌 → 裁剪区域 → OCR识别
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Failed to load image: {image_path}")
            return None

        bbox = self.detect_plate(image)
        if bbox is None:
            print("[INFO] No license plate detected.")
            return None

        x1, y1, x2, y2 = bbox
        plate_crop = image[y1:y2, x1:x2]

        plate_text, score = self.recognize_plate(plate_crop)

        return {
            'image_path': image_path,
            'plate_text': plate_text,
            'score': score,
            'bbox': [int(x1), int(y1), int(x2), int(y2)]
        }
