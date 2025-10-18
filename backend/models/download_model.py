"""
Model download script for YOLOv11 Fingerspelling Recognition
"""
import os
import requests
from pathlib import Path
from ultralytics import YOLO

def download_base_model(model_size='n'):
    """Download YOLOv11 base model"""
    model_name = f'yolo11{model_size}.pt'
    model_path = f'models/{model_name}'
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    if not os.path.exists(model_path):
        print(f"Downloading {model_name}...")
        model = YOLO(model_name)
        model.save(model_path)
        print(f"✅ {model_name} downloaded successfully!")
    else:
        print(f"✅ {model_name} already exists!")
    
    return model_path

def download_fingerspelling_model():
    """Download or create fingerspelling model"""
    model_path = 'models/yolov11_fingerspelling.pt'
    
    if os.path.exists(model_path):
        print(f"✅ Fingerspelling model already exists at {model_path}")
        return model_path
    
    print("📝 No pre-trained fingerspelling model found.")
    print("🔧 You have several options:")
    print("   1. Train your own model (see DATASETS_AND_MODELS.md)")
    print("   2. Use a base YOLOv11 model for testing")
    print("   3. Download from community sources")
    
    # Download base model for testing
    base_model = download_base_model('s')  # Use 's' for good balance
    print(f"📦 Using base model {base_model} for testing")
    
    return base_model

def main():
    """Main download function"""
    print("🚀 YOLOv11 Fingerspelling Model Downloader")
    print("=" * 50)
    
    # Download base models
    models = ['n', 's', 'm', 'l', 'x']
    for model_size in models:
        download_base_model(model_size)
    
    # Try to get fingerspelling model
    fingerspelling_model = download_fingerspelling_model()
    
    print("\n📚 Next Steps:")
    print("1. Check DATASETS_AND_MODELS.md for dataset information")
    print("2. Train your own model on fingerspelling data")
    print("3. Test with the benchmark framework")
    
    return fingerspelling_model

if __name__ == "__main__":
    main()
