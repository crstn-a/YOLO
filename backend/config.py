"""
Configuration settings for YOLOv11 Fingerspelling Benchmark Framework
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluation"""
    
    # Model settings
    model_path: str = "models/yolov11_fingerspelling.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    
    # Static evaluation settings
    static_letters: List[str] = None  # A-I, K-Y (excluding J and Z)
    static_threshold: float = 0.90  # 90% accuracy threshold for static letters
    
    # Dynamic evaluation settings
    dynamic_letters: List[str] = None  # J and Z
    dynamic_threshold: float = 0.85  # 85% accuracy threshold for dynamic gestures
    dtw_threshold: float = 0.7  # DTW similarity threshold
    
    # Performance settings
    target_fps: float = 25.0  # Target FPS for real-time performance
    min_fps: float = 20.0  # Minimum acceptable FPS
    max_latency: float = 0.04  # Maximum acceptable latency (40ms)
    
    # Dataset settings
    dataset_path: str = "datasets/test_data"
    batch_size: int = 1
    num_workers: int = 4
    
    # Output settings
    results_dir: str = "results"
    logs_dir: str = "results/logs"
    reports_dir: str = "results/reports"
    
    # Benchmark criteria for research-grade deployment
    research_grade_criteria: Dict[str, float] = None
    
    def __post_init__(self):
        if self.static_letters is None:
            self.static_letters = [chr(i) for i in range(ord('A'), ord('I')+1)] + \
                                 [chr(i) for i in range(ord('K'), ord('Y')+1)]
        
        if self.dynamic_letters is None:
            self.dynamic_letters = ['J', 'Z']
        
        if self.research_grade_criteria is None:
            self.research_grade_criteria = {
                "static_accuracy": 0.90,
                "dynamic_accuracy": 0.85,
                "fps": 25.0,
                "dtw_similarity": 0.7
            }
        
        # Create directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

# Global configuration instance
config = BenchmarkConfig()

# ASL Fingerspelling class mapping
ASL_CLASSES = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
    17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

# Reverse mapping for class names to indices
CLASS_TO_IDX = {v: k for k, v in ASL_CLASSES.items()}

# Static vs Dynamic letter classification
STATIC_LETTERS = [chr(i) for i in range(ord('A'), ord('I')+1)] + \
                [chr(i) for i in range(ord('K'), ord('Y')+1)]
DYNAMIC_LETTERS = ['J', 'Z']

