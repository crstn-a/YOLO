# YOLOv11 Fingerspelling Recognition Web App

A comprehensive web application and benchmark framework for evaluating YOLOv11's accuracy and performance in recognizing ASL fingerspelling letters, including both static (A-I, K-Y) and dynamic (J, Z) gestures.

## 🌟 Project Overview

This project combines a **Flutter frontend** with a **FastAPI backend** to create an interactive fingerspelling recognition system with comprehensive benchmarking capabilities.

### **Key Features**
- 🎯 **Real-time Recognition**: Live fingerspelling detection through camera
- 📊 **Comprehensive Benchmarking**: Research-grade evaluation framework
- 📱 **Cross-platform Support**: Works on web, mobile, and desktop
- 🤖 **AI Integration**: YOLOv11 for state-of-the-art object detection
- 📈 **Performance Analytics**: Real-time FPS, latency, and accuracy metrics
- 📄 **Automated Reporting**: Professional PDF reports with visualizations

## 🏗️ Project Structure

```
YOLO/                          # Root project directory
├── backend/                   # Python FastAPI backend
│   ├── config.py             # Configuration management
│   ├── main.py               # FastAPI application server
│   ├── requirements.txt      # Python dependencies
│   ├── README.md             # Backend documentation
│   ├── DATASETS_AND_MODELS.md # Dataset and model guide
│   ├── test_framework.py     # Framework testing script
│   ├── example_usage.py      # Usage examples
│   ├── prepare_dataset.py    # Dataset preparation tool
│   ├── benchmarks/
│   │   └── benchmark_runner.py  # Command-line benchmark runner
│   ├── utils/
│   │   ├── detection.py      # YOLOv11 detection module
│   │   ├── dtw_evaluation.py # Dynamic gesture evaluation
│   │   ├── benchmarking.py   # Comprehensive benchmark suite
│   │   ├── metrics.py        # Metrics calculation
│   │   └── report_generator.py # PDF report generation
│   ├── models/
│   │   ├── download_model.py # Model download script
│   │   └── yolov11_fingerspelling.pt  # YOLOv11 model (download required)
│   ├── datasets/
│   │   └── test_data/        # Test dataset directory (create your own)
│   └── results/
│       ├── logs/             # JSON log files (auto-generated)
│       └── reports/          # PDF reports (auto-generated)
└── frontend/                 # Flutter web/mobile app
    ├── lib/
    │   ├── main.dart         # Flutter app entry point
    │   ├── pages/
    │   │   └── camera_page.dart  # Camera interface
    │   └── services/
    │       └── api_services.dart # API communication
    ├── pubspec.yaml          # Flutter dependencies
    ├── android/              # Android-specific files
    ├── ios/                  # iOS-specific files
    ├── web/                  # Web-specific files
    └── windows/              # Windows-specific files
```

## 🚀 Quick Start

### **Prerequisites**
- Python 3.8+ installed
- Flutter SDK installed (for frontend)
- Webcam or camera device
- At least 4GB RAM (8GB+ recommended for optimal performance)
- GPU support (optional but recommended for better performance)

### **1. Clone the Repository**
```bash
git clone <your-repo-url>
cd YOLO
```

### **2. Setup Backend**
```bash
cd backend
pip install -r requirements.txt
python test_framework.py
```

### **3. Download Models**
```bash
python models/download_model.py
```

### **4. Start Backend Server**
```bash
python main.py
```

### **5. Setup Frontend**
```bash
cd ../frontend
flutter pub get
flutter run -d chrome  # For web
# or
flutter run  # For mobile/desktop
```

### **6. Run Benchmarks**
```bash
cd ../backend
python benchmarks/benchmark_runner.py --mode comprehensive
```

## 📊 What You Can Expect

### **Interactive Web Application**
- **Real-time Camera Interface**: Capture and analyze fingerspelling gestures
- **Live Recognition**: Get instant feedback on static letters (A-I, K-Y) and dynamic gestures (J, Z)
- **Performance Metrics**: See real-time FPS, latency, and accuracy measurements
- **Cross-platform Support**: Works on desktop, mobile, and web browsers

### **Comprehensive Benchmark Suite**
- **Static Accuracy Testing**: Evaluate recognition of 24 static letters with precision, recall, and F1-score
- **Dynamic Gesture Analysis**: Test J and Z gestures using Dynamic Time Warping (DTW)
- **Real-time Performance**: Measure FPS and latency for deployment readiness
- **Research Grade Assessment**: Determine if your model meets research-grade deployment criteria

### **Professional Reporting**
- **Automated PDF Reports**: Generate comprehensive benchmark reports
- **Visual Analytics**: Charts and tables showing performance metrics
- **Research Documentation**: Detailed analysis for academic and commercial use

## 🎯 Research Grade Criteria

A model is considered **research-grade and deployment-ready** if it meets:
- **Static Accuracy**: ≥ 90% on static letters (A-I, K-Y)
- **Dynamic Accuracy**: ≥ 85% on dynamic gestures (J, Z)
- **Real-time Performance**: ≥ 25 FPS
- **DTW Similarity**: ≥ 70% for dynamic gesture recognition

## 📚 Documentation

### **Backend Documentation**
- **[Backend README](backend/README.md)**: Comprehensive backend documentation
- **[Datasets & Models Guide](backend/DATASETS_AND_MODELS.md)**: Complete guide to datasets and pre-trained models
- **[API Documentation](http://localhost:8000)**: Interactive API documentation (when server is running)

### **Key Files**
- `backend/config.py`: Configuration management
- `backend/main.py`: FastAPI application server
- `backend/benchmarks/benchmark_runner.py`: Command-line benchmark runner
- `frontend/lib/main.dart`: Flutter app entry point
- `frontend/lib/pages/camera_page.dart`: Camera interface

## 🛠️ Development

### **Backend Development**
```bash
cd backend
pip install -r requirements.txt
python main.py  # Start development server
```

### **Frontend Development**
```bash
cd frontend
flutter pub get
flutter run -d chrome  # Hot reload enabled
```

### **Testing**
```bash
# Test backend framework
cd backend
python test_framework.py

# Run benchmarks
python benchmarks/benchmark_runner.py --mode comprehensive
```

## 📊 Performance Benchmarks

### **Typical Results (Good Model)**
- **Static Accuracy**: 85-95%
- **Dynamic Accuracy**: 80-90%
- **Real-time FPS**: 20-35 (varies by hardware)
- **Latency**: 30-80ms per inference

### **Hardware Requirements**
- **Minimum**: CPU-only, 4GB RAM, webcam
- **Recommended**: GPU, 8GB+ RAM, high-quality camera
- **Optimal**: RTX 3060+ GPU, 16GB+ RAM, 1080p camera

## 🎯 Target Audiences

### **👨‍🎓 Researchers & Academics**
- Gesture recognition research
- ASL studies and analysis
- Computer vision research
- Publication support with automated reports

### **👨‍💻 Developers & Engineers**
- AI integration learning
- Web development with FastAPI and Flutter
- API design and architecture
- Performance optimization

### **🎓 Students & Learners**
- Computer vision hands-on experience
- Machine learning model evaluation
- Full-stack application development
- Research methods and reporting

### **🏢 Industry Professionals**
- Accessibility tools development
- Gesture interface systems
- AI model deployment testing
- Research and development prototyping

## 🚀 Deployment

### **Local Development**
```bash
# Backend
cd backend && python main.py

# Frontend
cd frontend && flutter run -d chrome
```

### **Production Deployment**
- **Backend**: Deploy FastAPI to cloud platforms (AWS, GCP, Azure)
- **Frontend**: Build Flutter web app for production
- **Database**: Add persistent storage for results
- **Monitoring**: Implement logging and monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### **Common Issues**
- **Model not found**: Run `python models/download_model.py`
- **Camera permissions**: Ensure HTTPS or localhost for web access
- **Performance issues**: Check GPU availability and system resources
- **Dataset errors**: Validate dataset structure with `python prepare_dataset.py`

### **Getting Help**
- Check the [Backend README](backend/README.md) for detailed documentation
- Review log files in `backend/results/logs/` directory
- Test individual components using the API endpoints
- Check the Flutter console for frontend errors

## 🙏 Acknowledgments

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) for the object detection framework
- [Flutter](https://flutter.dev/) for the cross-platform frontend framework
- [FastAPI](https://fastapi.tiangolo.com/) for the high-performance backend framework
- ASL community for fingerspelling datasets and research

---

**Note**: This project is designed for research and development purposes. For production deployment, additional testing and validation may be required.
