# Git Setup Guide for YOLOv11 Fingerspelling Project

This guide will help you initialize Git and prepare your project for GitHub upload.

## 🚀 Quick Setup

### **Option 1: Automated Setup (Recommended)**

#### **For Windows:**
```cmd
# Run the batch file
setup_git.bat
```

#### **For Mac/Linux:**
```bash
# Make the script executable and run
chmod +x setup_git.sh
./setup_git.sh
```

### **Option 2: Manual Setup**

```bash
# 1. Initialize Git repository
git init

# 2. Add all files
git add .

# 3. Create initial commit
git commit -m "Initial commit: YOLOv11 Fingerspelling Recognition Web App

- Complete Flutter frontend with camera interface
- FastAPI backend with comprehensive benchmarking
- YOLOv11 integration for fingerspelling recognition
- Static (A-I, K-Y) and dynamic (J, Z) gesture evaluation
- Research-grade assessment framework
- Automated PDF report generation
- Cross-platform support (web, mobile, desktop)
- Comprehensive documentation and examples"

# 4. Set main branch
git branch -M main
```

## 📁 What's Included in Git

### **✅ Files Included:**
- **Source Code**: All Python and Dart source files
- **Configuration**: All configuration files and scripts
- **Documentation**: README.md, DATASETS_AND_MODELS.md, GIT_SETUP.md
- **Dependencies**: requirements.txt, pubspec.yaml
- **Scripts**: Setup, testing, and utility scripts
- **Project Structure**: Complete directory structure

### **❌ Files Excluded (via .gitignore):**
- **Model Files**: `*.pt`, `*.onnx`, `*.engine` (too large for Git)
- **Dataset Files**: `*.jpg`, `*.png`, `*.mp4`, `*.avi` (too large for Git)
- **Results**: `results/`, `logs/`, `*.pdf` (auto-generated)
- **Virtual Environments**: `venv/`, `env/`, `.venv/`
- **Cache Files**: `__pycache__/`, `.cache/`, `build/`
- **OS Files**: `.DS_Store`, `Thumbs.db`, `desktop.ini`
- **IDE Files**: `.vscode/`, `.idea/`, `*.swp`
- **Temporary Files**: `*.tmp`, `*.temp`, `*.log`

## 🔧 Git Configuration

### **Set Up Git User (if not already done):**
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### **Check Git Status:**
```bash
git status
```

### **View What Will Be Committed:**
```bash
git diff --cached
```

## 📤 Uploading to GitHub

### **Step 1: Create GitHub Repository**
1. Go to [GitHub.com](https://github.com)
2. Click "New repository"
3. Name it: `yolov11-fingerspelling` (or your preferred name)
4. **Don't** initialize with README, .gitignore, or license (we already have these)
5. Click "Create repository"

### **Step 2: Connect Local Repository to GitHub**
```bash
# Add remote origin (replace with your repository URL)
git remote add origin https://github.com/yourusername/your-repo-name.git

# Verify remote was added
git remote -v
```

### **Step 3: Push to GitHub**
```bash
# Push to GitHub
git push -u origin main
```

## 📊 Repository Structure on GitHub

Your GitHub repository will contain:

```
yolov11-fingerspelling/
├── README.md                    # Main project documentation
├── GIT_SETUP.md                # This Git setup guide
├── .gitignore                  # Git ignore rules
├── setup_git.sh               # Git setup script (Linux/Mac)
├── setup_git.bat              # Git setup script (Windows)
├── backend/                    # Python FastAPI backend
│   ├── README.md              # Backend documentation
│   ├── DATASETS_AND_MODELS.md # Dataset and model guide
│   ├── config.py              # Configuration
│   ├── main.py                # FastAPI server
│   ├── requirements.txt       # Python dependencies
│   ├── test_framework.py      # Testing script
│   ├── example_usage.py       # Usage examples
│   ├── prepare_dataset.py     # Dataset preparation
│   ├── .gitignore             # Backend-specific ignore rules
│   ├── benchmarks/            # Benchmark framework
│   ├── utils/                 # Utility modules
│   ├── models/                # Model scripts (no actual models)
│   └── datasets/              # Dataset scripts (no actual data)
└── frontend/                  # Flutter application
    ├── lib/                   # Dart source code
    ├── pubspec.yaml          # Flutter dependencies
    ├── android/              # Android-specific files
    ├── ios/                  # iOS-specific files
    ├── web/                  # Web-specific files
    ├── windows/              # Windows-specific files
    └── .gitignore            # Flutter-specific ignore rules
```

## 🔄 Ongoing Git Workflow

### **Daily Development:**
```bash
# Check status
git status

# Add changes
git add .

# Commit changes
git commit -m "Description of changes"

# Push to GitHub
git push
```

### **Feature Development:**
```bash
# Create feature branch
git checkout -b feature/new-feature

# Make changes and commit
git add .
git commit -m "Add new feature"

# Push feature branch
git push origin feature/new-feature

# Create pull request on GitHub
```

### **Updating from GitHub:**
```bash
# Pull latest changes
git pull origin main
```

## 📋 Pre-Upload Checklist

Before uploading to GitHub, ensure:

- [ ] All sensitive information removed (API keys, passwords)
- [ ] Large files excluded (models, datasets, results)
- [ ] Documentation complete and up-to-date
- [ ] Code properly commented
- [ ] Tests working (`python test_framework.py`)
- [ ] README files comprehensive
- [ ] .gitignore files properly configured

## 🚨 Important Notes

### **File Size Limits:**
- **GitHub**: 100MB per file, 1GB per repository
- **Large files**: Use Git LFS for models and datasets
- **Best practice**: Keep repository under 100MB total

### **Security Considerations:**
- Never commit API keys or passwords
- Use environment variables for sensitive data
- Review .gitignore files carefully
- Consider using GitHub Secrets for CI/CD

### **Model and Dataset Management:**
- Store large models in cloud storage (Google Drive, Dropbox)
- Provide download scripts for models
- Document dataset sources and requirements
- Use Git LFS for essential large files

## 🛠️ Troubleshooting

### **Common Issues:**

#### **"Repository not found"**
```bash
# Check remote URL
git remote -v

# Update remote URL if needed
git remote set-url origin https://github.com/yourusername/your-repo-name.git
```

#### **"Permission denied"**
```bash
# Use SSH instead of HTTPS
git remote set-url origin git@github.com:yourusername/your-repo-name.git

# Or configure GitHub credentials
git config --global credential.helper store
```

#### **"Large file detected"**
```bash
# Remove large files from Git history
git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch path/to/large/file' --prune-empty --tag-name-filter cat -- --all

# Force push to update GitHub
git push origin --force --all
```

## 📚 Additional Resources

- [Git Documentation](https://git-scm.com/doc)
- [GitHub Documentation](https://docs.github.com/)
- [Git LFS Documentation](https://git-lfs.github.io/)
- [Flutter Git Best Practices](https://docs.flutter.dev/development/tools/version-control)
- [Python Git Best Practices](https://docs.python.org/3/tutorial/venv.html)

## 🎯 Next Steps After Upload

1. **Set up GitHub Pages** for documentation
2. **Configure GitHub Actions** for CI/CD
3. **Add issue templates** for bug reports
4. **Set up branch protection** for main branch
5. **Add contributors** and collaborators
6. **Create releases** for stable versions

---

**Your YOLOv11 Fingerspelling project is now ready for GitHub! 🎉**
