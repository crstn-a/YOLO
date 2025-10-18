"""
Enhanced YOLOv11 detection module for fingerspelling recognition
"""
from ultralytics import YOLO
import cv2
import numpy as np
import time
import json
import os
from typing import List, Dict, Tuple, Any
from config import config, ASL_CLASSES, STATIC_LETTERS, DYNAMIC_LETTERS

class YOLODetector:
    """Enhanced YOLOv11 detector for fingerspelling recognition"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or config.model_path
        self.model = YOLO(self.model_path)
        self.confidence_threshold = config.confidence_threshold
        self.iou_threshold = config.iou_threshold
    
    async def run_inference(self, file, letter_type: str = "static") -> Dict[str, Any]:
        """Run YOLOv11 inference on uploaded image/video frame"""
        
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"error": "Failed to decode image"}
        
        # Run inference with timing
        start_time = time.time()
        results = self.model(img, conf=self.confidence_threshold, iou=self.iou_threshold)
        end_time = time.time()
        
        # Calculate performance metrics
        inference_time = end_time - start_time
        fps = 1 / inference_time if inference_time > 0 else 0
        
        # Process detections
        detections = self._process_detections(results[0], letter_type)
        
        # Create output
        output = {
            "detections": detections,
            "fps": round(fps, 2),
            "latency": round(inference_time, 4),
            "letter_type": letter_type,
            "timestamp": time.time(),
            "image_shape": img.shape
        }
        
        # Log results
        self._log_results(output, letter_type)
        
        return output
    
    def _process_detections(self, result, letter_type: str) -> List[Dict[str, Any]]:
        """Process YOLO detection results"""
        detections = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.data.cpu().numpy()
            
            for box in boxes:
                x1, y1, x2, y2, conf, cls = box
                
                # Get class name
                class_id = int(cls)
                class_name = ASL_CLASSES.get(class_id, f"Unknown_{class_id}")
                
                # Filter by letter type
                if letter_type == "static" and class_name not in STATIC_LETTERS:
                    continue
                elif letter_type == "dynamic" and class_name not in DYNAMIC_LETTERS:
                    continue
                
                detection = {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(conf),
                    "class_id": class_id,
                    "class_name": class_name,
                    "area": float((x2 - x1) * (y2 - y1))
                }
                detections.append(detection)
        
        return detections
    
    def _log_results(self, output: Dict[str, Any], letter_type: str) -> None:
        """Log results to JSON file"""
        timestamp = int(time.time())
        log_path = f"{config.logs_dir}/{letter_type}_{timestamp}.json"
        
        with open(log_path, 'w') as f:
            json.dump(output, f, indent=4)
    
    def batch_inference(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Run batch inference on multiple images"""
        results = []
        
        for i, img in enumerate(images):
            start_time = time.time()
            yolo_results = self.model(img, conf=self.confidence_threshold)
            end_time = time.time()
            
            inference_time = end_time - start_time
            fps = 1 / inference_time if inference_time > 0 else 0
            
            detections = self._process_detections(yolo_results[0], "static")
            
            results.append({
                "image_index": i,
                "detections": detections,
                "fps": round(fps, 2),
                "latency": round(inference_time, 4)
            })
        
        return results

# Global detector instance
detector = YOLODetector()

# Backward compatibility functions
async def run_yolo_inference(file):
    """Backward compatibility function"""
    return await detector.run_inference(file, "static")
