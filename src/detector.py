from ultralytics import YOLO
import cv2
import torch
import numpy as np

class HelmetDetector:
    def __init__(self, config):
        self.config = config
        self.model = YOLO(config['model']['weights_path'])
        self.device = config['model']['device']
        self.conf_threshold = config['model']['confidence_threshold']
        self.iou_threshold = config['model']['iou_threshold']
        
    def detect(self, frame):
        """
        Detect persons and helmets in frame
        Returns: List of detections [x1, y1, x2, y2, confidence, class_id]
        """
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'class_id': cls,
                    'class_name': result.names[cls]
                })
        
        return detections
    
    def detect_batch(self, frames):
        """Batch detection for multiple frames"""
        results = self.model(frames, conf=self.conf_threshold, device=self.device)
        return results