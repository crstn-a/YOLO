"""
Enhanced benchmarking module for YOLOv11 fingerspelling evaluation
"""
import json
import time
import os
import numpy as np
from typing import List, Dict, Any, Tuple
from ultralytics import YOLO
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from utils.detection import YOLODetector
from utils.dtw_evaluation import DynamicGestureEvaluator
from utils.metrics import BenchmarkMetrics
from config import config, ASL_CLASSES, STATIC_LETTERS, DYNAMIC_LETTERS

class FingerspellingBenchmark:
    """Comprehensive benchmark for YOLOv11 fingerspelling recognition"""
    
    def __init__(self):
        self.detector = YOLODetector()
        self.dynamic_evaluator = DynamicGestureEvaluator()
        self.metrics_calculator = BenchmarkMetrics(config)
        self.results = {}
    
    def run_static_benchmark(self, dataset_path: str = None) -> Dict[str, Any]:
        """Run static fingerspelling benchmark (A-I, K-Y)"""
        
        print("🔍 Running Static Fingerspelling Benchmark...")
        
        dataset_path = dataset_path or config.dataset_path
        if not os.path.exists(dataset_path):
            return {"error": f"Dataset path not found: {dataset_path}"}
        
        # Load test images and labels
        test_data = self._load_static_dataset(dataset_path)
        if not test_data:
            return {"error": "No static test data found"}
        
        y_true, y_pred = [], []
        fps_measurements = []
        latency_measurements = []
        
        # Run inference on test data
        for img_path, true_label in test_data:
            try:
                # Load image
                import cv2
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # Run inference with timing
                start_time = time.time()
                results = self.detector.model(img, conf=config.confidence_threshold)
                end_time = time.time()
                
                # Calculate performance metrics
                inference_time = end_time - start_time
                fps = 1 / inference_time if inference_time > 0 else 0
                
                fps_measurements.append(fps)
                latency_measurements.append(inference_time)
                
                # Extract prediction
                pred_label = None
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    pred_class_id = int(results[0].boxes.cls[0].item())
                    pred_label = ASL_CLASSES.get(pred_class_id, "Unknown")
                
                y_true.append(true_label)
                y_pred.append(pred_label)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        if not y_true:
            return {"error": "No valid predictions generated"}
        
        # Calculate static metrics
        static_metrics = self.metrics_calculator.calculate_static_metrics(
            y_true, y_pred, list(ASL_CLASSES.values())
        )
        
        # Calculate performance metrics
        performance_metrics = self.metrics_calculator.calculate_performance_metrics(
            fps_measurements, latency_measurements
        )
        
        # Store results
        self.results['static'] = {
            'metrics': static_metrics,
            'performance': performance_metrics,
            'samples_evaluated': len(y_true)
        }
        
        print(f"✅ Static Benchmark Complete - Accuracy: {static_metrics.get('overall_accuracy', 0):.3f}")
        return self.results['static']
    
    def run_dynamic_benchmark(self, dataset_path: str = None) -> Dict[str, Any]:
        """Run dynamic gesture benchmark (J, Z)"""
        
        print("🔄 Running Dynamic Gesture Benchmark...")
        
        dataset_path = dataset_path or config.dataset_path
        if not os.path.exists(dataset_path):
            return {"error": f"Dataset path not found: {dataset_path}"}
        
        # Load dynamic test sequences
        dynamic_sequences = self._load_dynamic_dataset(dataset_path)
        if not dynamic_sequences:
            return {"error": "No dynamic test sequences found"}
        
        all_dtw_scores = []
        all_sequence_accuracies = []
        all_completion_rates = []
        
        # Evaluate each dynamic sequence
        for sequence_name, frame_paths in dynamic_sequences.items():
            try:
                # Load sequence frames
                trajectories = []
                for frame_path in frame_paths:
                    import cv2
                    img = cv2.imread(frame_path)
                    if img is not None:
                        # Extract trajectory (simplified)
                        trajectory = self._extract_trajectory_from_image(img)
                        if trajectory:
                            trajectories.append(trajectory)
                
                if len(trajectories) < 2:
                    continue
                
                # Calculate DTW metrics
                dtw_results = self.dynamic_evaluator._calculate_dtw_metrics(trajectories)
                if 'error' not in dtw_results:
                    all_dtw_scores.append(dtw_results['similarity'])
                
                # Calculate sequence accuracy (simplified)
                # In practice, you'd compare against ground truth sequences
                sequence_accuracy = dtw_results.get('similarity', 0) >= config.dtw_threshold
                all_sequence_accuracies.append(float(sequence_accuracy))
                
                # Calculate completion rate
                completion_analysis = self.dynamic_evaluator._analyze_gesture_completion(trajectories)
                all_completion_rates.append(completion_analysis['completion_rate'])
                
            except Exception as e:
                print(f"Error processing sequence {sequence_name}: {e}")
                continue
        
        if not all_dtw_scores:
            return {"error": "No valid dynamic sequences processed"}
        
        # Calculate aggregate dynamic metrics
        dynamic_metrics = {
            "dtw_similarity": float(np.mean(all_dtw_scores)),
            "sequence_accuracy": float(np.mean(all_sequence_accuracies)),
            "completion_rate": float(np.mean(all_completion_rates)),
            "sequences_evaluated": len(all_dtw_scores),
            "meets_threshold": np.mean(all_sequence_accuracies) >= config.dynamic_threshold
        }
        
        # Store results
        self.results['dynamic'] = {
            'metrics': dynamic_metrics,
            'samples_evaluated': len(all_dtw_scores)
        }
        
        print(f"✅ Dynamic Benchmark Complete - Sequence Accuracy: {dynamic_metrics['sequence_accuracy']:.3f}")
        return self.results['dynamic']
    
    def run_efficiency_benchmark(self, num_iterations: int = 100) -> Dict[str, Any]:
        """Run efficiency benchmark for real-time performance"""
        
        print("⚡ Running Efficiency Benchmark...")
        
        # Create test images of different sizes
        test_images = self._create_test_images()
        
        fps_measurements = []
        latency_measurements = []
        
        # Run benchmark iterations
        for i in range(num_iterations):
            for img in test_images:
                try:
                    start_time = time.time()
                    results = self.detector.model(img, conf=config.confidence_threshold)
                    end_time = time.time()
                    
                    inference_time = end_time - start_time
                    fps = 1 / inference_time if inference_time > 0 else 0
                    
                    fps_measurements.append(fps)
                    latency_measurements.append(inference_time)
                    
                except Exception as e:
                    print(f"Error in efficiency benchmark iteration {i}: {e}")
                    continue
        
        if not fps_measurements:
            return {"error": "No efficiency measurements collected"}
        
        # Calculate performance metrics
        performance_metrics = self.metrics_calculator.calculate_performance_metrics(
            fps_measurements, latency_measurements
        )
        
        # Store results
        self.results['efficiency'] = {
            'metrics': performance_metrics,
            'iterations': num_iterations
        }
        
        print(f"✅ Efficiency Benchmark Complete - Avg FPS: {performance_metrics['average_fps']:.2f}")
        return self.results['efficiency']
    
    def run_comprehensive_benchmark(self, dataset_path: str = None) -> Dict[str, Any]:
        """Run complete benchmark suite"""
        
        print("🚀 Starting Comprehensive YOLOv11 Fingerspelling Benchmark...")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all benchmark components
        static_results = self.run_static_benchmark(dataset_path)
        dynamic_results = self.run_dynamic_benchmark(dataset_path)
        efficiency_results = self.run_efficiency_benchmark()
        
        # Evaluate research-grade criteria
        research_grade_evaluation = self.metrics_calculator.evaluate_research_grade_criteria(
            static_results.get('metrics', {}),
            dynamic_results.get('metrics', {}),
            efficiency_results.get('metrics', {})
        )
        
        # Compile comprehensive results
        comprehensive_results = {
            "benchmark_info": {
                "timestamp": time.time(),
                "duration": time.time() - start_time,
                "config": {
                    "static_threshold": config.static_threshold,
                    "dynamic_threshold": config.dynamic_threshold,
                    "target_fps": config.target_fps,
                    "dtw_threshold": config.dtw_threshold
                }
            },
            "static_evaluation": static_results,
            "dynamic_evaluation": dynamic_results,
            "efficiency_evaluation": efficiency_results,
            "research_grade_assessment": research_grade_evaluation
        }
        
        # Save comprehensive results
        self._save_comprehensive_results(comprehensive_results)
        
        print("=" * 60)
        print("🎯 BENCHMARK SUMMARY")
        print("=" * 60)
        
        if 'error' not in static_results:
            print(f"📊 Static Accuracy: {static_results['metrics'].get('overall_accuracy', 0):.3f}")
        
        if 'error' not in dynamic_results:
            print(f"🔄 Dynamic Sequence Accuracy: {dynamic_results['metrics'].get('sequence_accuracy', 0):.3f}")
        
        if 'error' not in efficiency_results:
            print(f"⚡ Average FPS: {efficiency_results['metrics'].get('average_fps', 0):.2f}")
        
        print(f"🏆 Research Grade Ready: {research_grade_evaluation.get('research_grade_ready', False)}")
        print("=" * 60)
        
        return comprehensive_results
    
    def _load_static_dataset(self, dataset_path: str) -> List[Tuple[str, str]]:
        """Load static fingerspelling dataset"""
        # Simplified dataset loading - in practice, you'd have a proper dataset structure
        test_data = []
        
        # Look for static letter images
        for letter in STATIC_LETTERS:
            letter_dir = os.path.join(dataset_path, letter)
            if os.path.exists(letter_dir):
                for img_file in os.listdir(letter_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(letter_dir, img_file)
                        test_data.append((img_path, letter))
        
        return test_data
    
    def _load_dynamic_dataset(self, dataset_path: str) -> Dict[str, List[str]]:
        """Load dynamic gesture sequences"""
        sequences = {}
        
        # Look for dynamic gesture sequences
        for letter in DYNAMIC_LETTERS:
            letter_dir = os.path.join(dataset_path, letter)
            if os.path.exists(letter_dir):
                # Look for sequence directories
                for seq_dir in os.listdir(letter_dir):
                    seq_path = os.path.join(letter_dir, seq_dir)
                    if os.path.isdir(seq_path):
                        frame_paths = []
                        for frame_file in sorted(os.listdir(seq_path)):
                            if frame_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                                frame_paths.append(os.path.join(seq_path, frame_file))
                        if frame_paths:
                            sequences[f"{letter}_{seq_dir}"] = frame_paths
        
        return sequences
    
    def _create_test_images(self) -> List[np.ndarray]:
        """Create test images for efficiency benchmarking"""
        import cv2
        
        test_images = []
        
        # Create images of different sizes
        sizes = [(224, 224), (416, 416), (640, 640)]
        
        for size in sizes:
            # Create random test image
            img = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
            test_images.append(img)
        
        return test_images
    
    def _extract_trajectory_from_image(self, img: np.ndarray) -> Dict[str, Any]:
        """Extract trajectory from single image (simplified)"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find contours
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate centroid
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return {"centroid": [cx, cy], "area": cv2.contourArea(largest_contour)}
        
        return None
    
    def _save_comprehensive_results(self, results: Dict[str, Any]) -> str:
        """Save comprehensive benchmark results"""
        timestamp = int(time.time())
        filename = f"{config.results_dir}/comprehensive_benchmark_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"📄 Results saved to: {filename}")
        return filename

# Global benchmark instance
benchmark = FingerspellingBenchmark()

# Backward compatibility functions
def evaluate_static(model_path, dataset_loader):
    """Backward compatibility function"""
    return benchmark.run_static_benchmark()

def measure_efficiency():
    """Backward compatibility function"""
    return benchmark.run_efficiency_benchmark()
