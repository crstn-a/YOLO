"""
Enhanced Dynamic Time Warping evaluation for dynamic fingerspelling gestures (J, Z)
"""
from dtw import dtw
import numpy as np
import cv2
import time
import json
from typing import List, Dict, Tuple, Any
from config import config, DYNAMIC_LETTERS

class DynamicGestureEvaluator:
    """Evaluator for dynamic fingerspelling gestures using DTW"""
    
    def __init__(self):
        self.dtw_threshold = config.dtw_threshold
        self.dynamic_letters = DYNAMIC_LETTERS
    
    async def evaluate_dynamic(self, files: List) -> Dict[str, Any]:
        """Evaluate dynamic gestures from sequence of frames"""
        
        if not files or len(files) < 2:
            return {"error": "At least 2 frames required for dynamic gesture evaluation"}
        
        # Extract trajectories from frames
        trajectories = []
        for file in files:
            trajectory = await self._extract_trajectory(file)
            if trajectory is not None:
                trajectories.append(trajectory)
        
        if len(trajectories) < 2:
            return {"error": "Insufficient valid trajectories extracted"}
        
        # Calculate DTW metrics
        dtw_results = self._calculate_dtw_metrics(trajectories)
        
        # Determine gesture type and completion
        gesture_analysis = self._analyze_gesture_completion(trajectories)
        
        # Combine results
        results = {
            "dtw_metrics": dtw_results,
            "gesture_analysis": gesture_analysis,
            "trajectory_count": len(trajectories),
            "timestamp": time.time()
        }
        
        # Log results
        self._log_dynamic_results(results)
        
        return results
    
    async def _extract_trajectory(self, file) -> Dict[str, Any]:
        """Extract hand trajectory from a single frame"""
        
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return None
        
        # Convert to grayscale for hand detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Simple hand detection using contour analysis
        # In a real implementation, you'd use more sophisticated hand tracking
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find the largest contour (likely the hand)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate centroid
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            return None
        
        # Calculate additional trajectory features
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        return {
            "centroid": [cx, cy],
            "area": float(area),
            "perimeter": float(perimeter),
            "timestamp": time.time()
        }
    
    def _calculate_dtw_metrics(self, trajectories: List[Dict]) -> Dict[str, Any]:
        """Calculate DTW metrics for trajectory comparison"""
        
        if len(trajectories) < 2:
            return {"error": "Insufficient trajectories for DTW calculation"}
        
        # Extract centroid points for DTW
        points = [traj["centroid"] for traj in trajectories]
        points_array = np.array(points)
        
        # Create a reference trajectory (straight line for J, zigzag for Z)
        # This is a simplified approach - in practice, you'd have ground truth trajectories
        ref_trajectory = self._create_reference_trajectory(len(points), "J")  # Default to J
        
        try:
            # Calculate DTW distance
            distance, path, _, _ = dtw(points_array, ref_trajectory, 
                                     dist=lambda x, y: np.linalg.norm(x - y))
            
            # Calculate similarity score
            similarity = 1 / (1 + distance)
            
            # Calculate path statistics
            path_length = len(path[0])
            avg_distance = distance / path_length if path_length > 0 else 0
            
            return {
                "dtw_distance": float(distance),
                "similarity": float(similarity),
                "path_length": path_length,
                "average_distance": float(avg_distance),
                "meets_threshold": similarity >= self.dtw_threshold
            }
            
        except Exception as e:
            return {"error": f"DTW calculation failed: {str(e)}"}
    
    def _create_reference_trajectory(self, length: int, gesture_type: str) -> np.ndarray:
        """Create reference trajectory for gesture type"""
        
        if gesture_type == "J":
            # J gesture: downward curve
            x = np.linspace(0, 1, length)
            y = np.sin(np.pi * x) * 0.5 + x * 0.5
            return np.column_stack([x, y])
        
        elif gesture_type == "Z":
            # Z gesture: zigzag pattern
            x = np.linspace(0, 1, length)
            y = np.sin(4 * np.pi * x) * 0.3 + 0.5
            return np.column_stack([x, y])
        
        else:
            # Default: straight line
            x = np.linspace(0, 1, length)
            y = np.linspace(0, 1, length)
            return np.column_stack([x, y])
    
    def _analyze_gesture_completion(self, trajectories: List[Dict]) -> Dict[str, Any]:
        """Analyze if the gesture was completed properly"""
        
        if len(trajectories) < 2:
            return {"completion_rate": 0.0, "is_complete": False}
        
        # Calculate movement metrics
        total_movement = 0
        for i in range(1, len(trajectories)):
            prev_point = np.array(trajectories[i-1]["centroid"])
            curr_point = np.array(trajectories[i]["centroid"])
            movement = np.linalg.norm(curr_point - prev_point)
            total_movement += movement
        
        # Calculate completion rate based on movement
        # This is a simplified metric - in practice, you'd use more sophisticated analysis
        expected_movement = 100  # pixels (adjust based on your requirements)
        completion_rate = min(1.0, total_movement / expected_movement)
        
        # Check if gesture is complete
        is_complete = completion_rate >= 0.8  # 80% completion threshold
        
        return {
            "completion_rate": float(completion_rate),
            "is_complete": is_complete,
            "total_movement": float(total_movement),
            "trajectory_length": len(trajectories)
        }
    
    def _log_dynamic_results(self, results: Dict[str, Any]) -> None:
        """Log dynamic evaluation results"""
        timestamp = int(time.time())
        log_path = f"{config.logs_dir}/dynamic_{timestamp}.json"
        
        with open(log_path, 'w') as f:
            json.dump(results, f, indent=4)
    
    def evaluate_sequence_accuracy(self, predicted_sequences: List[str], 
                                 ground_truth_sequences: List[str]) -> Dict[str, Any]:
        """Evaluate sequence-level accuracy for dynamic gestures"""
        
        if len(predicted_sequences) != len(ground_truth_sequences):
            return {"error": "Sequence length mismatch"}
        
        correct_sequences = 0
        total_sequences = len(predicted_sequences)
        
        for pred, gt in zip(predicted_sequences, ground_truth_sequences):
            if pred == gt:
                correct_sequences += 1
        
        accuracy = correct_sequences / total_sequences if total_sequences > 0 else 0
        
        return {
            "sequence_accuracy": float(accuracy),
            "correct_sequences": correct_sequences,
            "total_sequences": total_sequences,
            "meets_threshold": accuracy >= config.dynamic_threshold
        }

# Global evaluator instance
dynamic_evaluator = DynamicGestureEvaluator()

# Backward compatibility function
async def evaluate_dynamic(files):
    """Backward compatibility function"""
    return await dynamic_evaluator.evaluate_dynamic(files)
