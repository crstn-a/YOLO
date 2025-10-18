"""
Test script to verify YOLOv11 Fingerspelling Benchmark Framework
"""
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from config import config, ASL_CLASSES, STATIC_LETTERS, DYNAMIC_LETTERS
        print("✅ Config module imported successfully")
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        from utils.detection import YOLODetector
        print("✅ Detection module imported successfully")
    except Exception as e:
        print(f"❌ Detection import failed: {e}")
        return False
    
    try:
        from utils.dtw_evaluation import DynamicGestureEvaluator
        print("✅ DTW evaluation module imported successfully")
    except Exception as e:
        print(f"❌ DTW evaluation import failed: {e}")
        return False
    
    try:
        from utils.benchmarking import FingerspellingBenchmark
        print("✅ Benchmarking module imported successfully")
    except Exception as e:
        print(f"❌ Benchmarking import failed: {e}")
        return False
    
    try:
        from utils.metrics import BenchmarkMetrics
        print("✅ Metrics module imported successfully")
    except Exception as e:
        print(f"❌ Metrics import failed: {e}")
        return False
    
    try:
        from utils.report_generator import generate_pdf_report, generate_summary_report
        print("✅ Report generator module imported successfully")
    except Exception as e:
        print(f"❌ Report generator import failed: {e}")
        return False
    
    return True

def test_configuration():
    """Test configuration loading"""
    print("\n⚙️ Testing configuration...")
    
    try:
        from config import config
        
        # Test basic configuration
        assert hasattr(config, 'static_threshold'), "Missing static_threshold"
        assert hasattr(config, 'dynamic_threshold'), "Missing dynamic_threshold"
        assert hasattr(config, 'target_fps'), "Missing target_fps"
        assert hasattr(config, 'dtw_threshold'), "Missing dtw_threshold"
        
        print("✅ Configuration loaded successfully")
        print(f"  • Static Threshold: {config.static_threshold:.1%}")
        print(f"  • Dynamic Threshold: {config.dynamic_threshold:.1%}")
        print(f"  • Target FPS: {config.target_fps}")
        print(f"  • DTW Threshold: {config.dtw_threshold:.1%}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_class_definitions():
    """Test class definitions"""
    print("\n🏗️ Testing class definitions...")
    
    try:
        from utils.detection import YOLODetector
        from utils.dtw_evaluation import DynamicGestureEvaluator
        from utils.benchmarking import FingerspellingBenchmark
        from utils.metrics import BenchmarkMetrics
        
        # Test class instantiation (without model loading)
        print("✅ All classes defined successfully")
        
        return True
    except Exception as e:
        print(f"❌ Class definition test failed: {e}")
        return False

def test_directory_structure():
    """Test directory structure"""
    print("\n📁 Testing directory structure...")
    
    required_dirs = [
        "results",
        "results/logs", 
        "results/reports",
        "models",
        "datasets",
        "datasets/test_data"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path} exists")
        else:
            print(f"⚠️  {dir_path} missing (will be created when needed)")
            # Create directory if it doesn't exist
            os.makedirs(dir_path, exist_ok=True)
            print(f"✅ {dir_path} created")
    
    return all_exist

def test_model_file():
    """Test if model file exists"""
    print("\n🤖 Testing model file...")
    
    model_path = "models/yolov11_fingerspelling.pt"
    if os.path.exists(model_path):
        print(f"✅ Model file found: {model_path}")
        return True
    else:
        print(f"⚠️  Model file not found: {model_path}")
        print("   This is expected if you haven't downloaded the model yet.")
        print("   The framework will work but benchmarks will fail without the model.")
        return False

def test_requirements():
    """Test if required packages are available"""
    print("\n📦 Testing requirements...")
    
    required_packages = [
        "ultralytics",
        "opencv-python", 
        "numpy",
        "scikit-learn",
        "dtw-python",
        "reportlab",
        "fastapi",
        "uvicorn"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} available")
        except ImportError:
            print(f"❌ {package} missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Install with: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main test function"""
    print("🧪 YOLOv11 Fingerspelling Benchmark Framework - Test Suite")
    print("=" * 70)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_configuration),
        ("Class Definition Test", test_class_definitions),
        ("Directory Structure Test", test_directory_structure),
        ("Model File Test", test_model_file),
        ("Requirements Test", test_requirements)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 70)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Framework is ready to use.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    print("\n📝 Next steps:")
    print("1. Install missing packages: pip install -r requirements.txt")
    print("2. Download YOLOv11 model to models/ directory")
    print("3. Prepare test dataset according to README.md structure")
    print("4. Run benchmarks: python benchmarks/benchmark_runner.py --mode comprehensive")

if __name__ == "__main__":
    main()

