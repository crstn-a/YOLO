"""
Dataset preparation script for YOLOv11 Fingerspelling Recognition
"""
import os
import shutil
from pathlib import Path
import requests
import zipfile

def create_dataset_structure():
    """Create the expected dataset directory structure"""
    print("📁 Creating dataset directory structure...")
    
    # Create main dataset directory
    dataset_path = Path("datasets/test_data")
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    # Create directories for static letters (A-I, K-Y)
    static_letters = [chr(i) for i in range(ord('A'), ord('I')+1)] + \
                    [chr(i) for i in range(ord('K'), ord('Y')+1)]
    
    for letter in static_letters:
        letter_dir = dataset_path / letter
        letter_dir.mkdir(exist_ok=True)
        print(f"✅ Created directory: {letter_dir}")
    
    # Create directories for dynamic letters (J, Z)
    dynamic_letters = ['J', 'Z']
    for letter in dynamic_letters:
        letter_dir = dataset_path / letter
        letter_dir.mkdir(exist_ok=True)
        
        # Create sequence directories
        for i in range(1, 4):  # Create 3 example sequences
            seq_dir = letter_dir / f"sequence{i}"
            seq_dir.mkdir(exist_ok=True)
            print(f"✅ Created directory: {seq_dir}")
    
    print(f"\n📁 Dataset structure created at: {dataset_path}")
    print("📝 Next steps:")
    print("   1. Add your images to the respective letter directories")
    print("   2. For dynamic letters, add frame sequences to sequence directories")
    print("   3. Run the benchmark to test your dataset")

def download_sample_dataset():
    """Download a sample dataset for testing"""
    print("📥 Downloading sample dataset...")
    
    # This would download a sample dataset
    # For now, we'll create placeholder files
    dataset_path = Path("datasets/test_data")
    
    # Create some placeholder images (empty files for demonstration)
    for letter in ['A', 'B', 'C']:  # Just a few examples
        letter_dir = dataset_path / letter
        for i in range(3):
            placeholder_file = letter_dir / f"sample_{i}.txt"
            placeholder_file.write_text(f"Placeholder image for letter {letter}")
    
    print("✅ Sample dataset structure created")
    print("📝 Note: Replace placeholder files with actual images")

def validate_dataset():
    """Validate the dataset structure"""
    print("🔍 Validating dataset structure...")
    
    dataset_path = Path("datasets/test_data")
    if not dataset_path.exists():
        print("❌ Dataset directory not found!")
        return False
    
    static_letters = [chr(i) for i in range(ord('A'), ord('I')+1)] + \
                    [chr(i) for i in range(ord('K'), ord('Y')+1)]
    dynamic_letters = ['J', 'Z']
    
    all_valid = True
    
    # Check static letters
    for letter in static_letters:
        letter_dir = dataset_path / letter
        if letter_dir.exists():
            images = list(letter_dir.glob("*.jpg")) + list(letter_dir.glob("*.png")) + list(letter_dir.glob("*.jpeg"))
            print(f"✅ {letter}: {len(images)} images")
            if len(images) == 0:
                print(f"⚠️  {letter}: No images found")
        else:
            print(f"❌ {letter}: Missing directory")
            all_valid = False
    
    # Check dynamic letters
    for letter in dynamic_letters:
        letter_dir = dataset_path / letter
        if letter_dir.exists():
            sequences = [d for d in letter_dir.iterdir() if d.is_dir()]
            print(f"✅ {letter}: {len(sequences)} sequences")
            if len(sequences) == 0:
                print(f"⚠️  {letter}: No sequences found")
        else:
            print(f"❌ {letter}: Missing directory")
            all_valid = False
    
    if all_valid:
        print("✅ Dataset structure is valid!")
    else:
        print("❌ Dataset structure has issues. Please check the output above.")
    
    return all_valid

def show_dataset_info():
    """Show information about available datasets"""
    print("📚 Available Datasets for Fingerspelling Recognition:")
    print("=" * 60)
    
    datasets = [
        {
            "name": "ASL Alphabet Dataset",
            "size": "87,000 images",
            "classes": "A-Z + SPACE, DELETE, NOTHING",
            "source": "Kaggle",
            "url": "https://www.kaggle.com/datasets/grassknoted/asl-alphabet",
            "format": "Ready for YOLO"
        },
        {
            "name": "ASLYset Dataset",
            "size": "5,200 images",
            "classes": "A-I, K-Y (24 letters)",
            "source": "GitHub",
            "url": "https://github.com/yayayru/sign-lanuage-datasets",
            "format": "Requires conversion"
        },
        {
            "name": "Roboflow ASL Letters",
            "size": "Variable",
            "classes": "A-Z fingerspelling",
            "source": "Roboflow",
            "url": "https://universe.roboflow.com/american-sign-language-letters",
            "format": "YOLO ready"
        }
    ]
    
    for i, dataset in enumerate(datasets, 1):
        print(f"{i}. {dataset['name']}")
        print(f"   Size: {dataset['size']}")
        print(f"   Classes: {dataset['classes']}")
        print(f"   Source: {dataset['source']}")
        print(f"   URL: {dataset['url']}")
        print(f"   Format: {dataset['format']}")
        print()
    
    print("📖 For detailed information, see DATASETS_AND_MODELS.md")

def main():
    """Main dataset preparation function"""
    print("🚀 YOLOv11 Fingerspelling Dataset Preparation")
    print("=" * 60)
    
    while True:
        print("\n📋 Choose an option:")
        print("1. Create dataset structure")
        print("2. Download sample dataset")
        print("3. Validate dataset")
        print("4. Show dataset information")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            create_dataset_structure()
        elif choice == "2":
            download_sample_dataset()
        elif choice == "3":
            validate_dataset()
        elif choice == "4":
            show_dataset_info()
        elif choice == "5":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

