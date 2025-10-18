@echo off
REM YOLOv11 Fingerspelling Project - Git Setup Script for Windows
echo 🚀 Setting up Git repository for YOLOv11 Fingerspelling Project
echo ==================================================================

REM Initialize Git repository
echo 📁 Initializing Git repository...
git init

REM Add all files to Git
echo 📝 Adding files to Git...
git add .

REM Create initial commit
echo 💾 Creating initial commit...
git commit -m "Initial commit: YOLOv11 Fingerspelling Recognition Web App

- Complete Flutter frontend with camera interface
- FastAPI backend with comprehensive benchmarking
- YOLOv11 integration for fingerspelling recognition
- Static (A-I, K-Y) and dynamic (J, Z) gesture evaluation
- Research-grade assessment framework
- Automated PDF report generation
- Cross-platform support (web, mobile, desktop)
- Comprehensive documentation and examples"

REM Set up main branch
echo 🌿 Setting up main branch...
git branch -M main

REM Display repository status
echo 📊 Repository status:
git status

echo.
echo ✅ Git repository initialized successfully!
echo.
echo 📋 Next steps:
echo 1. Create a new repository on GitHub
echo 2. Add the remote origin:
echo    git remote add origin https://github.com/yourusername/your-repo-name.git
echo 3. Push to GitHub:
echo    git push -u origin main
echo.
echo 📚 Files included in this repository:
echo - Complete Flutter frontend application
echo - FastAPI backend with benchmarking framework
echo - Comprehensive documentation (README.md, DATASETS_AND_MODELS.md)
echo - Example scripts and utilities
echo - Proper .gitignore files for Python and Flutter
echo.
echo 📁 Files excluded (as per .gitignore):
echo - Model files (*.pt, *.onnx) - too large for Git
echo - Dataset files (*.jpg, *.png, *.mp4) - too large for Git
echo - Results and logs - auto-generated
echo - Virtual environments and cache files
echo - OS-specific files (.DS_Store, Thumbs.db)
echo.
echo 🎯 Your project is ready for GitHub!
pause
