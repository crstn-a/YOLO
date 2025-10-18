"""
Main benchmark runner for YOLOv11 Fingerspelling Evaluation
"""
import sys
import os
import argparse
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.benchmarking import FingerspellingBenchmark
from utils.report_generator import generate_pdf_report
from config import config

def main():
    """Main benchmark runner with command-line interface"""
    
    parser = argparse.ArgumentParser(description="YOLOv11 Fingerspelling Benchmark Runner")
    parser.add_argument("--mode", choices=["static", "dynamic", "efficiency", "comprehensive"], 
                       default="comprehensive", help="Benchmark mode to run")
    parser.add_argument("--dataset", type=str, default=None, 
                       help="Path to dataset directory")
    parser.add_argument("--iterations", type=int, default=100,
                       help="Number of iterations for efficiency benchmark")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for results")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to custom config file")
    
    args = parser.parse_args()
    
    # Update config if custom path provided
    if args.dataset:
        config.dataset_path = args.dataset
    if args.output:
        config.results_dir = args.output
    
    print("🚀 YOLOv11 Fingerspelling Benchmark Runner")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Dataset: {config.dataset_path}")
    print(f"Output: {config.results_dir}")
    print("=" * 50)
    
    # Initialize benchmark
    benchmark = FingerspellingBenchmark()
    
    try:
        if args.mode == "static":
            results = benchmark.run_static_benchmark()
        elif args.mode == "dynamic":
            results = benchmark.run_dynamic_benchmark()
        elif args.mode == "efficiency":
            results = benchmark.run_efficiency_benchmark(args.iterations)
        elif args.mode == "comprehensive":
            results = benchmark.run_comprehensive_benchmark()
        
        # Generate PDF report
        if 'error' not in str(results):
            print("\n📊 Generating PDF Report...")
            report_path = generate_pdf_report(results)
            print(f"📄 Report generated: {report_path}")
        
        print("\n✅ Benchmark completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

