"""
Enhanced FastAPI application for YOLOv11 Fingerspelling Benchmark
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from utils.detection import YOLODetector
from utils.dtw_evaluation import DynamicGestureEvaluator
from utils.benchmarking import FingerspellingBenchmark
from utils.report_generator import generate_pdf_report, generate_summary_report
from config import config
import json
import os
from typing import List, Dict, Any

# Initialize FastAPI app
app = FastAPI(
    title="YOLOv11 Fingerspelling Benchmark API",
    description="Comprehensive benchmark framework for evaluating YOLOv11 fingerspelling recognition",
    version="1.0.0"
)

# Initialize components
detector = YOLODetector()
dynamic_evaluator = DynamicGestureEvaluator()
benchmark = FingerspellingBenchmark()

@app.get("/")
async def root():
    """API root endpoint with information"""
    return {
        "message": "YOLOv11 Fingerspelling Benchmark API",
        "version": "1.0.0",
        "endpoints": {
            "static_prediction": "/predict/static/",
            "dynamic_prediction": "/predict/dynamic/",
            "benchmark_static": "/benchmark/static/",
            "benchmark_dynamic": "/benchmark/dynamic/",
            "benchmark_efficiency": "/benchmark/efficiency/",
            "benchmark_comprehensive": "/benchmark/comprehensive/",
            "config": "/config/"
        }
    }

@app.get("/config/")
async def get_config():
    """Get current benchmark configuration"""
    return {
        "static_threshold": config.static_threshold,
        "dynamic_threshold": config.dynamic_threshold,
        "target_fps": config.target_fps,
        "dtw_threshold": config.dtw_threshold,
        "static_letters": config.static_letters,
        "dynamic_letters": config.dynamic_letters,
        "research_grade_criteria": config.research_grade_criteria
    }

@app.post("/predict/static/")
async def predict_static(file: UploadFile = File(...)):
    """Predict static fingerspelling letter (A-I, K-Y)"""
    try:
        results = await detector.run_inference(file, "static")
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Static prediction failed: {str(e)}")

@app.post("/predict/dynamic/")
async def predict_dynamic(files: List[UploadFile] = File(...)):
    """Predict dynamic fingerspelling gesture (J, Z) from sequence of frames"""
    try:
        if len(files) < 2:
            raise HTTPException(status_code=400, detail="At least 2 frames required for dynamic prediction")
        
        results = await dynamic_evaluator.evaluate_dynamic(files)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dynamic prediction failed: {str(e)}")

@app.post("/benchmark/static/")
async def run_static_benchmark(dataset_path: str = None):
    """Run static fingerspelling benchmark (A-I, K-Y)"""
    try:
        results = benchmark.run_static_benchmark(dataset_path)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Static benchmark failed: {str(e)}")

@app.post("/benchmark/dynamic/")
async def run_dynamic_benchmark(dataset_path: str = None):
    """Run dynamic gesture benchmark (J, Z)"""
    try:
        results = benchmark.run_dynamic_benchmark(dataset_path)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dynamic benchmark failed: {str(e)}")

@app.post("/benchmark/efficiency/")
async def run_efficiency_benchmark(iterations: int = 100):
    """Run efficiency benchmark for real-time performance"""
    try:
        results = benchmark.run_efficiency_benchmark(iterations)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Efficiency benchmark failed: {str(e)}")

@app.post("/benchmark/comprehensive/")
async def run_comprehensive_benchmark(dataset_path: str = None):
    """Run comprehensive benchmark suite"""
    try:
        results = benchmark.run_comprehensive_benchmark(dataset_path)
        
        # Generate PDF reports
        full_report = generate_pdf_report(results)
        summary_report = generate_summary_report(results)
        
        return {
            "benchmark_results": results,
            "reports": {
                "full_report": full_report,
                "summary_report": summary_report
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comprehensive benchmark failed: {str(e)}")

@app.get("/benchmark/status/")
async def get_benchmark_status():
    """Get current benchmark status and results"""
    try:
        # Check if results exist
        results_dir = config.results_dir
        if not os.path.exists(results_dir):
            return {"status": "no_results", "message": "No benchmark results found"}
        
        # Find latest results
        result_files = [f for f in os.listdir(results_dir) if f.startswith("comprehensive_benchmark_")]
        if not result_files:
            return {"status": "no_results", "message": "No comprehensive benchmark results found"}
        
        latest_file = max(result_files, key=lambda x: os.path.getctime(os.path.join(results_dir, x)))
        
        with open(os.path.join(results_dir, latest_file), 'r') as f:
            results = json.load(f)
        
        return {
            "status": "results_available",
            "latest_results": latest_file,
            "summary": {
                "static_accuracy": results.get("static_evaluation", {}).get("metrics", {}).get("overall_accuracy", 0),
                "dynamic_accuracy": results.get("dynamic_evaluation", {}).get("metrics", {}).get("sequence_accuracy", 0),
                "average_fps": results.get("efficiency_evaluation", {}).get("metrics", {}).get("average_fps", 0),
                "research_grade_ready": results.get("research_grade_assessment", {}).get("research_grade_ready", False)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get benchmark status: {str(e)}")

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": os.path.exists(config.model_path),
        "config_loaded": True
    }

# Legacy endpoints for backward compatibility
@app.get("/benchmark/run/")
def run_benchmark_legacy():
    """Legacy benchmark endpoint for backward compatibility"""
    try:
        results = benchmark.run_efficiency_benchmark()
        report_path = generate_pdf_report(results)
        return {"status": "Benchmark Completed", "report": report_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Legacy benchmark failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
