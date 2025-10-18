"""
Comprehensive metrics calculation for YOLOv11 Fingerspelling Benchmark
"""
import numpy as np
import json
import time
from typing import Dict, List, Tuple, Any
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from dtw import dtw
import cv2

class BenchmarkMetrics:
    """Comprehensive metrics calculation for fingerspelling evaluation"""
    
    def __init__(self, config):
        self.config = config
        self.static_results = []
        self.dynamic_results = []
        self.performance_results = []
    
    def calculate_static_metrics(self, y_true: List[int], y_pred: List[int], 
                               class_names: List[str]) -> Dict[str, Any]:
        """Calculate static fingerspelling metrics (A-I, K-Y)"""
        
        # Filter for static letters only
        static_indices = [i for i, name in enumerate(class_names) if name in self.config.static_letters]
        
        # Filter predictions and ground truth for static letters
        static_mask = np.isin(y_true, static_indices)
        static_y_true = np.array(y_true)[static_mask]
        static_y_pred = np.array(y_pred)[static_mask]
        
        if len(static_y_true) == 0:
            return {"error": "No static letter predictions found"}
        
        # Calculate metrics
        precision = precision_score(static_y_true, static_y_pred, average='macro', zero_division=0)
        recall = recall_score(static_y_true, static_y_pred, average='macro', zero_division=0)
        f1 = f1_score(static_y_true, static_y_pred, average='macro', zero_division=0)
        accuracy = accuracy_score(static_y_true, static_y_pred)
        
        # Per-class metrics
        precision_per_class = precision_score(static_y_true, static_y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(static_y_true, static_y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(static_y_true, static_y_pred, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(static_y_true, static_y_pred)
        
        # Per-class detailed metrics
        per_class_metrics = {}
        for i, class_name in enumerate(class_names):
            if class_name in self.config.static_letters:
                class_idx = static_indices.index(i) if i in static_indices else None
                if class_idx is not None and class_idx < len(precision_per_class):
                    per_class_metrics[class_name] = {
                        "precision": float(precision_per_class[class_idx]),
                        "recall": float(recall_per_class[class_idx]),
                        "f1_score": float(f1_per_class[class_idx])
                    }
        
        return {
            "overall_accuracy": float(accuracy),
            "overall_precision": float(precision),
            "overall_recall": float(recall),
            "overall_f1_score": float(f1),
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": cm.tolist(),
            "static_letters_evaluated": len(static_y_true),
            "meets_threshold": accuracy >= self.config.static_threshold
        }
    
    def calculate_dynamic_metrics(self, trajectories: List[Dict], 
                                ground_truth_trajectories: List[Dict]) -> Dict[str, Any]:
        """Calculate dynamic gesture metrics (J, Z) using DTW"""
        
        if not trajectories or not ground_truth_trajectories:
            return {"error": "No dynamic gesture data provided"}
        
        dtw_scores = []
        sequence_accuracies = []
        completion_rates = []
        
        for i, (pred_traj, gt_traj) in enumerate(zip(trajectories, ground_truth_trajectories)):
            # Extract trajectory points
            pred_points = np.array(pred_traj.get('points', []))
            gt_points = np.array(gt_traj.get('points', []))
            
            if len(pred_points) == 0 or len(gt_points) == 0:
                continue
            
            # Calculate DTW distance and similarity
            try:
                distance, path, _, _ = dtw(pred_points, gt_points, 
                                         dist=lambda x, y: np.linalg.norm(x - y))
                similarity = 1 / (1 + distance)
                dtw_scores.append(similarity)
                
                # Sequence accuracy (binary: correct gesture or not)
                is_correct = similarity >= self.config.dtw_threshold
                sequence_accuracies.append(float(is_correct))
                
                # Completion rate (whether the full gesture was performed)
                completion_rate = min(1.0, len(pred_points) / len(gt_points))
                completion_rates.append(completion_rate)
                
            except Exception as e:
                print(f"DTW calculation error for trajectory {i}: {e}")
                continue
        
        if not dtw_scores:
            return {"error": "No valid dynamic gesture calculations"}
        
        # Calculate aggregate metrics
        avg_dtw_similarity = np.mean(dtw_scores)
        avg_sequence_accuracy = np.mean(sequence_accuracies)
        avg_completion_rate = np.mean(completion_rates)
        
        return {
            "dtw_similarity": float(avg_dtw_similarity),
            "sequence_accuracy": float(avg_sequence_accuracy),
            "completion_rate": float(avg_completion_rate),
            "individual_scores": [float(score) for score in dtw_scores],
            "meets_threshold": avg_sequence_accuracy >= self.config.dynamic_threshold,
            "gestures_evaluated": len(dtw_scores)
        }
    
    def calculate_performance_metrics(self, fps_measurements: List[float], 
                                   latency_measurements: List[float]) -> Dict[str, Any]:
        """Calculate real-time performance metrics"""
        
        if not fps_measurements:
            return {"error": "No performance measurements provided"}
        
        avg_fps = np.mean(fps_measurements)
        min_fps = np.min(fps_measurements)
        max_fps = np.max(fps_measurements)
        fps_std = np.std(fps_measurements)
        
        avg_latency = np.mean(latency_measurements) if latency_measurements else 0
        min_latency = np.min(latency_measurements) if latency_measurements else 0
        max_latency = np.max(latency_measurements) if latency_measurements else 0
        
        # Real-time performance assessment
        meets_target_fps = avg_fps >= self.config.target_fps
        meets_min_fps = avg_fps >= self.config.min_fps
        meets_latency = avg_latency <= self.config.max_latency
        
        return {
            "average_fps": float(avg_fps),
            "min_fps": float(min_fps),
            "max_fps": float(max_fps),
            "fps_std": float(fps_std),
            "average_latency": float(avg_latency),
            "min_latency": float(min_latency),
            "max_latency": float(max_latency),
            "meets_target_fps": meets_target_fps,
            "meets_min_fps": meets_min_fps,
            "meets_latency": meets_latency,
            "real_time_ready": meets_min_fps and meets_latency,
            "measurements_count": len(fps_measurements)
        }
    
    def evaluate_research_grade_criteria(self, static_metrics: Dict, 
                                       dynamic_metrics: Dict, 
                                       performance_metrics: Dict) -> Dict[str, Any]:
        """Evaluate if the model meets research-grade deployment criteria"""
        
        criteria = self.config.research_grade_criteria
        
        # Check static accuracy
        static_passes = static_metrics.get('overall_accuracy', 0) >= criteria['static_accuracy']
        
        # Check dynamic accuracy
        dynamic_passes = dynamic_metrics.get('sequence_accuracy', 0) >= criteria['dynamic_accuracy']
        
        # Check FPS performance
        fps_passes = performance_metrics.get('average_fps', 0) >= criteria['fps']
        
        # Check DTW similarity
        dtw_passes = dynamic_metrics.get('dtw_similarity', 0) >= criteria['dtw_similarity']
        
        # Overall assessment
        all_criteria_met = static_passes and dynamic_passes and fps_passes and dtw_passes
        
        return {
            "static_accuracy_passes": static_passes,
            "dynamic_accuracy_passes": dynamic_passes,
            "fps_passes": fps_passes,
            "dtw_similarity_passes": dtw_passes,
            "research_grade_ready": all_criteria_met,
            "criteria_details": {
                "static_accuracy": {
                    "required": criteria['static_accuracy'],
                    "achieved": static_metrics.get('overall_accuracy', 0),
                    "passes": static_passes
                },
                "dynamic_accuracy": {
                    "required": criteria['dynamic_accuracy'],
                    "achieved": dynamic_metrics.get('sequence_accuracy', 0),
                    "passes": dynamic_passes
                },
                "fps": {
                    "required": criteria['fps'],
                    "achieved": performance_metrics.get('average_fps', 0),
                    "passes": fps_passes
                },
                "dtw_similarity": {
                    "required": criteria['dtw_similarity'],
                    "achieved": dynamic_metrics.get('dtw_similarity', 0),
                    "passes": dtw_passes
                }
            }
        }
    
    def save_metrics(self, filename: str = None) -> str:
        """Save all metrics to JSON file"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"{self.config.logs_dir}/benchmark_metrics_{timestamp}.json"
        
        metrics_data = {
            "timestamp": time.time(),
            "config": {
                "static_threshold": self.config.static_threshold,
                "dynamic_threshold": self.config.dynamic_threshold,
                "target_fps": self.config.target_fps,
                "dtw_threshold": self.config.dtw_threshold
            },
            "static_results": self.static_results,
            "dynamic_results": self.dynamic_results,
            "performance_results": self.performance_results
        }
        
        with open(filename, 'w') as f:
            json.dump(metrics_data, f, indent=4)
        
        return filename

