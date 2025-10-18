"""
Example usage of YOLOv11 Fingerspelling Benchmark Framework
"""
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from utils.benchmarking import FingerspellingBenchmark
from utils.report_generator import generate_pdf_report, generate_summary_report
from config import config

def example_static_benchmark():
    """Example: Run static fingerspelling benchmark"""
    print("🔍 Running Static Fingerspelling Benchmark Example...")
    
    # Initialize benchmark
    benchmark = FingerspellingBenchmark()
    
    # Run static benchmark
    results = benchmark.run_static_benchmark()
    
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        return
    
    # Print results
    metrics = results.get('metrics', {})
    print(f"✅ Static Benchmark Complete!")
    print(f"📊 Accuracy: {metrics.get('overall_accuracy', 0):.3f}")
    print(f"📊 Precision: {metrics.get('overall_precision', 0):.3f}")
    print(f"📊 Recall: {metrics.get('overall_recall', 0):.3f}")
    print(f"📊 F1-Score: {metrics.get('overall_f1_score', 0):.3f}")
    print(f"📊 Samples: {results.get('samples_evaluated', 0)}")
    print(f"🎯 Meets Threshold (90%): {'✅ YES' if metrics.get('meets_threshold', False) else '❌ NO'}")

def example_dynamic_benchmark():
    """Example: Run dynamic gesture benchmark"""
    print("\n🔄 Running Dynamic Gesture Benchmark Example...")
    
    # Initialize benchmark
    benchmark = FingerspellingBenchmark()
    
    # Run dynamic benchmark
    results = benchmark.run_dynamic_benchmark()
    
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        return
    
    # Print results
    metrics = results.get('metrics', {})
    print(f"✅ Dynamic Benchmark Complete!")
    print(f"📊 Sequence Accuracy: {metrics.get('sequence_accuracy', 0):.3f}")
    print(f"📊 DTW Similarity: {metrics.get('dtw_similarity', 0):.3f}")
    print(f"📊 Completion Rate: {metrics.get('completion_rate', 0):.3f}")
    print(f"📊 Sequences: {metrics.get('sequences_evaluated', 0)}")
    print(f"🎯 Meets Threshold (85%): {'✅ YES' if metrics.get('meets_threshold', False) else '❌ NO'}")

def example_efficiency_benchmark():
    """Example: Run efficiency benchmark"""
    print("\n⚡ Running Efficiency Benchmark Example...")
    
    # Initialize benchmark
    benchmark = FingerspellingBenchmark()
    
    # Run efficiency benchmark
    results = benchmark.run_efficiency_benchmark(iterations=50)  # Reduced for example
    
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        return
    
    # Print results
    metrics = results.get('metrics', {})
    print(f"✅ Efficiency Benchmark Complete!")
    print(f"📊 Average FPS: {metrics.get('average_fps', 0):.2f}")
    print(f"📊 Min FPS: {metrics.get('min_fps', 0):.2f}")
    print(f"📊 Max FPS: {metrics.get('max_fps', 0):.2f}")
    print(f"📊 Average Latency: {metrics.get('average_latency', 0)*1000:.2f} ms")
    print(f"🎯 Real-time Ready: {'✅ YES' if metrics.get('real_time_ready', False) else '❌ NO'}")
    print(f"🎯 Meets Target FPS (25): {'✅ YES' if metrics.get('meets_target_fps', False) else '❌ NO'}")

def example_comprehensive_benchmark():
    """Example: Run comprehensive benchmark suite"""
    print("\n🚀 Running Comprehensive Benchmark Example...")
    print("=" * 60)
    
    # Initialize benchmark
    benchmark = FingerspellingBenchmark()
    
    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark()
    
    # Print summary
    print("\n🎯 BENCHMARK SUMMARY")
    print("=" * 60)
    
    # Static results
    if "static_evaluation" in results and "error" not in results["static_evaluation"]:
        static_metrics = results["static_evaluation"]["metrics"]
        print(f"📊 Static Accuracy: {static_metrics.get('overall_accuracy', 0):.3f}")
    
    # Dynamic results
    if "dynamic_evaluation" in results and "error" not in results["dynamic_evaluation"]:
        dynamic_metrics = results["dynamic_evaluation"]["metrics"]
        print(f"🔄 Dynamic Sequence Accuracy: {dynamic_metrics.get('sequence_accuracy', 0):.3f}")
    
    # Efficiency results
    if "efficiency_evaluation" in results and "error" not in results["efficiency_evaluation"]:
        efficiency_metrics = results["efficiency_evaluation"]["metrics"]
        print(f"⚡ Average FPS: {efficiency_metrics.get('average_fps', 0):.2f}")
    
    # Research grade assessment
    if "research_grade_assessment" in results:
        assessment = results["research_grade_assessment"]
        ready = assessment.get('research_grade_ready', False)
        print(f"🏆 Research Grade Ready: {'✅ YES' if ready else '❌ NO'}")
        
        # Detailed criteria
        criteria = assessment.get('criteria_details', {})
        print("\n📋 Detailed Criteria:")
        for criterion, details in criteria.items():
            status = "✅ PASS" if details.get('passes', False) else "❌ FAIL"
            print(f"  • {criterion}: {details.get('achieved', 0):.1%} (Required: {details.get('required', 0):.1%}) {status}")
    
    # Generate reports
    print("\n📄 Generating Reports...")
    try:
        full_report = generate_pdf_report(results)
        summary_report = generate_summary_report(results)
        print(f"📄 Full Report: {full_report}")
        print(f"📄 Summary Report: {summary_report}")
    except Exception as e:
        print(f"❌ Report generation failed: {e}")

def example_configuration():
    """Example: Show current configuration"""
    print("\n⚙️ Current Configuration:")
    print("=" * 40)
    print(f"Static Threshold: {config.static_threshold:.1%}")
    print(f"Dynamic Threshold: {config.dynamic_threshold:.1%}")
    print(f"Target FPS: {config.target_fps}")
    print(f"DTW Threshold: {config.dtw_threshold:.1%}")
    print(f"Static Letters: {', '.join(config.static_letters)}")
    print(f"Dynamic Letters: {', '.join(config.dynamic_letters)}")
    print(f"Model Path: {config.model_path}")
    print(f"Dataset Path: {config.dataset_path}")

def main():
    """Main example function"""
    print("🚀 YOLOv11 Fingerspelling Benchmark Framework - Example Usage")
    print("=" * 70)
    
    # Show configuration
    example_configuration()
    
    # Note: These examples will show errors if no dataset is available
    # In a real scenario, you would have a proper dataset structure
    
    print("\n" + "=" * 70)
    print("📝 Note: These examples require a properly structured dataset.")
    print("📝 See README.md for dataset structure requirements.")
    print("📝 Run with actual data to see full benchmark results.")
    print("=" * 70)
    
    # Uncomment these lines when you have a dataset:
    # example_static_benchmark()
    # example_dynamic_benchmark()
    # example_efficiency_benchmark()
    # example_comprehensive_benchmark()

if __name__ == "__main__":
    main()

