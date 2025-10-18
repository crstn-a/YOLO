# Datasets and Pre-trained Models for YOLOv11 Fingerspelling Recognition

This guide provides comprehensive information about datasets and pre-trained models for fingerspelling recognition, helping you get started with your own model training or use existing solutions.

## 🎯 Quick Start Options

### **Option 1: Use Pre-trained Models (Recommended for Quick Testing)**
- Download existing YOLOv11 models trained on fingerspelling
- Test immediately with your benchmark framework
- No training required, just inference

### **Option 2: Train Your Own Model**
- Use public datasets to train custom models
- Fine-tune for your specific use case
- Better accuracy for your specific conditions

### **Option 3: Hybrid Approach**
- Start with pre-trained models
- Fine-tune on your own data
- Best of both worlds

## 📊 Recommended Datasets

### **1. ASL Alphabet Dataset (Most Popular)**
- **Size**: 87,000 RGB images
- **Classes**: 29 classes (A-Z + SPACE, DELETE, NOTHING)
- **Resolution**: 200×200 pixels
- **Features**: Multiple backgrounds, lighting conditions
- **Download**: [Kaggle ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Format**: Ready for YOLO training
- **Best for**: General fingerspelling recognition

### **2. ASLYset Dataset**
- **Size**: 5,200 images
- **Classes**: 24 ASL alphabet signs (A-I, K-Y, excluding J and Z)
- **Features**: Complex backgrounds, multiple individuals
- **Download**: [ASLYset Dataset](https://github.com/yayayru/sign-lanuage-datasets)
- **Format**: Requires conversion to YOLO format
- **Best for**: Static letter recognition with robustness

### **3. Roboflow ASL Letters Dataset**
- **Size**: Variable (multiple versions available)
- **Classes**: A-Z fingerspelling
- **Features**: YOLO-ready format, multiple augmentations
- **Download**: [Roboflow ASL Letters](https://universe.roboflow.com/american-sign-language-letters)
- **Format**: Already in YOLO format
- **Best for**: Quick training with minimal preprocessing

### **4. Custom Dataset Creation**
- **Tools**: Use your own camera/phone
- **Size**: Start with 100-500 images per class
- **Classes**: Focus on A-I, K-Y for static, J-Z for dynamic
- **Format**: Convert to YOLO format using tools
- **Best for**: Specific use cases and conditions

## 🤖 Pre-trained Models

### **1. YOLOv11 Base Models**
```bash
# Download YOLOv11 base models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11s.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11m.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11l.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11x.pt
```

### **2. Community Trained Models**
- **ASL Fingerspelling Models**: Check Hugging Face, GitHub
- **Custom Trained Models**: Look for community contributions
- **Research Models**: Academic papers often share models

### **3. Model Download Script**
Create a script to download models automatically:

```python
# models/download_model.py
import os
import requests
from pathlib import Path

def download_model(model_url, save_path):
    """Download model from URL"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if not os.path.exists(save_path):
        print(f"Downloading model from {model_url}...")
        response = requests.get(model_url)
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Model saved to {save_path}")
    else:
        print(f"Model already exists at {save_path}")

# Example usage
if __name__ == "__main__":
    # Download a pre-trained fingerspelling model
    model_url = "https://example.com/yolov11_fingerspelling.pt"
    save_path = "models/yolov11_fingerspelling.pt"
    download_model(model_url, save_path)
```

## 🚀 Getting Started with Datasets

### **Step 1: Download a Dataset**

#### **Option A: ASL Alphabet Dataset (Recommended)**
```bash
# Download from Kaggle (requires Kaggle API)
pip install kaggle
kaggle datasets download -d grassknoted/asl-alphabet
unzip asl-alphabet.zip

# Or download manually from:
# https://www.kaggle.com/datasets/grassknoted/asl-alphabet
```

#### **Option B: Roboflow Dataset**
```bash
# Install Roboflow
pip install roboflow

# Download dataset
from roboflow import Roboflow
rf = Roboflow(api_key="your_api_key")
project = rf.workspace("workspace").project("asl-letters")
dataset = project.version(1).download("yolov11")
```

### **Step 2: Prepare Dataset for Training**

#### **Convert to YOLO Format**
```python
# utils/dataset_converter.py
import os
import json
from pathlib import Path
import cv2

def convert_to_yolo_format(dataset_path, output_path):
    """Convert dataset to YOLO format"""
    
    # Create output directories
    os.makedirs(f"{output_path}/images/train", exist_ok=True)
    os.makedirs(f"{output_path}/images/val", exist_ok=True)
    os.makedirs(f"{output_path}/labels/train", exist_ok=True)
    os.makedirs(f"{output_path}/labels/val", exist_ok=True)
    
    # Class mapping
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
    # Process images and create labels
    # Implementation depends on your dataset format
    
    # Create data.yaml
    data_yaml = {
        'path': str(Path(output_path).absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(classes),
        'names': classes
    }
    
    with open(f"{output_path}/data.yaml", 'w') as f:
        import yaml
        yaml.dump(data_yaml, f)
    
    print(f"Dataset converted to YOLO format in {output_path}")

if __name__ == "__main__":
    convert_to_yolo_format("datasets/asl_alphabet", "datasets/yolo_format")
```

### **Step 3: Train Your Model**

```python
# train_model.py
from ultralytics import YOLO
import os

def train_fingerspelling_model():
    """Train YOLOv11 model on fingerspelling dataset"""
    
    # Load a pre-trained model
    model = YOLO('yolo11n.pt')  # or yolo11s.pt, yolo11m.pt, etc.
    
    # Train the model
    results = model.train(
        data='datasets/yolo_format/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device='cuda',  # or 'cpu'
        project='runs/train',
        name='fingerspelling_model'
    )
    
    # Save the trained model
    model.save('models/yolov11_fingerspelling.pt')
    
    print("Training completed!")
    return results

if __name__ == "__main__":
    train_fingerspelling_model()
```

## 🎯 Dataset Preparation for Your Benchmark

### **Create Test Dataset Structure**
```bash
# Create the expected directory structure
mkdir -p datasets/test_data/{A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z}

# For static letters (A-I, K-Y)
# Place images in respective directories:
# datasets/test_data/A/image1.jpg
# datasets/test_data/A/image2.jpg
# etc.

# For dynamic letters (J, Z)
# Create sequence directories:
# datasets/test_data/J/sequence1/frame1.jpg
# datasets/test_data/J/sequence1/frame2.jpg
# datasets/test_data/J/sequence1/frame3.jpg
# etc.
```

### **Dataset Validation Script**
```python
# validate_dataset.py
import os
from pathlib import Path

def validate_dataset_structure(dataset_path="datasets/test_data"):
    """Validate dataset structure for benchmark"""
    
    static_letters = [chr(i) for i in range(ord('A'), ord('I')+1)] + \
                    [chr(i) for i in range(ord('K'), ord('Y')+1)]
    dynamic_letters = ['J', 'Z']
    
    print("🔍 Validating dataset structure...")
    
    # Check static letters
    for letter in static_letters:
        letter_dir = Path(dataset_path) / letter
        if letter_dir.exists():
            images = list(letter_dir.glob("*.jpg")) + list(letter_dir.glob("*.png"))
            print(f"✅ {letter}: {len(images)} images")
        else:
            print(f"❌ {letter}: Missing directory")
    
    # Check dynamic letters
    for letter in dynamic_letters:
        letter_dir = Path(dataset_path) / letter
        if letter_dir.exists():
            sequences = [d for d in letter_dir.iterdir() if d.is_dir()]
            print(f"✅ {letter}: {len(sequences)} sequences")
        else:
            print(f"❌ {letter}: Missing directory")
    
    print("Dataset validation complete!")

if __name__ == "__main__":
    validate_dataset_structure()
```

## 📊 Recommended Dataset Sizes

### **Minimum Requirements**
- **Static Letters**: 50-100 images per letter (A-I, K-Y)
- **Dynamic Letters**: 10-20 sequences per letter (J, Z)
- **Total Images**: ~2,000-4,000 images

### **Recommended Sizes**
- **Static Letters**: 200-500 images per letter
- **Dynamic Letters**: 50-100 sequences per letter
- **Total Images**: ~10,000-20,000 images

### **Research-Grade Sizes**
- **Static Letters**: 500-1000 images per letter
- **Dynamic Letters**: 100-200 sequences per letter
- **Total Images**: ~25,000-50,000 images

## 🔧 Data Augmentation

### **Recommended Augmentations**
```python
# augmentation_config.py
AUGMENTATION_CONFIG = {
    'hsv_h': 0.015,  # HSV hue augmentation
    'hsv_s': 0.7,     # HSV saturation augmentation
    'hsv_v': 0.4,     # HSV value augmentation
    'degrees': 0.0,   # Rotation degrees
    'translate': 0.1, # Translation fraction
    'scale': 0.5,     # Scale range
    'shear': 0.0,     # Shear degrees
    'perspective': 0.0, # Perspective augmentation
    'flipud': 0.0,    # Vertical flip probability
    'fliplr': 0.5,    # Horizontal flip probability
    'mosaic': 1.0,    # Mosaic augmentation probability
    'mixup': 0.0,     # Mixup augmentation probability
}
```

## 🎯 Model Selection Guide

### **For Quick Testing**
- **YOLOv11n**: Fastest, lowest accuracy
- **YOLOv11s**: Good balance of speed and accuracy
- **Use case**: Prototyping, mobile deployment

### **For Production**
- **YOLOv11m**: Good accuracy, reasonable speed
- **YOLOv11l**: High accuracy, moderate speed
- **Use case**: Production applications

### **For Research**
- **YOLOv11x**: Highest accuracy, slower speed
- **Custom architectures**: Research-specific modifications
- **Use case**: Academic research, maximum accuracy

## 🚀 Quick Start Commands

### **Download and Test Pre-trained Model**
```bash
# 1. Download a pre-trained model
python models/download_model.py

# 2. Test with your benchmark
python test_framework.py

# 3. Run benchmarks
python benchmarks/benchmark_runner.py --mode comprehensive
```

### **Train Your Own Model**
```bash
# 1. Download dataset
# (Follow dataset download instructions above)

# 2. Convert to YOLO format
python utils/dataset_converter.py

# 3. Train model
python train_model.py

# 4. Test with benchmark
python benchmarks/benchmark_runner.py --mode comprehensive
```

## 📚 Additional Resources

### **Dataset Sources**
- [Kaggle ASL Datasets](https://www.kaggle.com/search?q=asl+sign+language)
- [Roboflow Universe](https://universe.roboflow.com/search?q=sign+language)
- [GitHub Sign Language Datasets](https://github.com/yayayru/sign-lanuage-datasets)
- [Papers with Code Datasets](https://paperswithcode.com/datasets?q=sign+language)

### **Model Sources**
- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- [Hugging Face Models](https://huggingface.co/models?search=yolo)
- [GitHub Community Models](https://github.com/search?q=yolo+fingerspelling)

### **Training Resources**
- [YOLOv11 Documentation](https://docs.ultralytics.com/)
- [Computer Vision Tutorials](https://github.com/ultralytics/yolov5)
- [ASL Recognition Papers](https://paperswithcode.com/task/sign-language-recognition)

## 🎯 Next Steps

1. **Choose your approach**: Pre-trained model or custom training
2. **Download dataset**: Start with ASL Alphabet Dataset
3. **Prepare data**: Convert to YOLO format
4. **Train/Download model**: Use provided scripts
5. **Test with benchmark**: Validate performance
6. **Iterate and improve**: Fine-tune based on results

Remember: Start simple with pre-trained models, then move to custom training as you gain experience!
