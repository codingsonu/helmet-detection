# **README.md** (Short Version)

```markdown
# 🦺 Helmet Compliance Detection System

AI-powered real-time helmet detection system for industrial safety monitoring using YOLOv8 and Streamlit.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B)

---

## 📋 Overview

Automatically detect and monitor helmet usage in industrial environments. Process video feeds, track workers, classify compliance, and generate safety analytics.

**Key Features:**
- ✅ Real-time helmet detection using YOLOv8
- ✅ Worker tracking across frames (SORT algorithm)
- ✅ Interactive web dashboard (Streamlit)
- ✅ Automated CSV reports and violation snapshots
- ✅ Compliance analytics and metrics

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/helmet-detection.git
cd helmet-detection

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

**Method 1: Web Dashboard (Recommended)**
```bash
streamlit run app.py
```
- Upload video → Adjust settings → Process → View results

**Method 2: Command Line**
```bash
python main.py --input data/input/video.mp4 --output output/result.mp4
```

**Method 3: Python Script**
```python
from main import HelmetComplianceSystem

system = HelmetComplianceSystem()
system.process_video('input.mp4', 'output.mp4')
```

---

## 📁 Project Structure

```
helmet-detection/
├── app.py                      # Streamlit dashboard
├── main.py                     # Main pipeline
├── config.yaml                 # Configuration
├── requirements.txt            # Dependencies
│
├── src/                        # Source modules
│   ├── detector.py            # YOLOv8 detector
│   ├── tracker.py             # SORT tracker
│   ├── compliance.py          # Compliance logic
│   └── analytics.py           # Analytics manager
│
├── utils/
│   └── visualization.py       # Visualization tools
│
├── data/
│   ├── input/                 # Input videos
│   └── models/
│       └── helmet_best.pt     # YOLOv8 model
│
└── output/
    ├── videos/                # Processed videos
    ├── logs/                  # CSV reports
    └── snapshots/             # Violation images
```

---

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
model:
  weights_path: "data/models/helmet_best.pt"
  confidence_threshold: 0.5     # Detection confidence
  device: "cpu"                 # "cpu" or "cuda"

tracking:
  max_age: 30                   # Max frames without detection
  min_hits: 3                   # Min detections before confirmed

video:
  skip_frames: 1                # Process every Nth frame
  
analytics:
  save_violations: true         # Save violation snapshots
```

---

## 📊 Outputs

### 1. Annotated Video
`output/videos/annotated_output.mp4`
- Bounding boxes with IDs
- Compliance status labels
- Real-time statistics overlay

### 2. CSV Report
`output/logs/compliance_log.csv`
```csv
timestamp,frame_number,person_id,status,bbox_x1,bbox_y1,bbox_x2,bbox_y2
2024-01-15 10:30:45,1,0,Compliant,120,80,220,280
```

### 3. Summary Report
`output/logs/compliance_log_summary.txt`
```
Total Detections: 1,234
Compliant: 1,180
Violations: 54
Compliance Rate: 95.6%
```

### 4. Violation Snapshots
`output/snapshots/violation_ID{id}_Frame{frame}.jpg`

---

## 🎯 System Architecture

```
Input Video → YOLOv8 Detection → SORT Tracking → Compliance Check → Analytics
                                                          ↓
                                                  Annotated Video + Reports
```

**Components:**
1. **Detector** (YOLOv8): Detects helmets in frames
2. **Tracker** (SORT): Tracks objects across frames
3. **Compliance Checker**: Determines compliance status
4. **Analytics**: Generates reports and metrics
5. **Visualizer**: Annotates video output

---

## 📦 Dependencies

```txt
opencv-python>=4.8.0
ultralytics>=8.0.0
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
streamlit>=1.28.0
filterpy>=1.4.5
scipy>=1.10.0
PyYAML>=6.0
tqdm>=4.65.0
```

---

## 🔧 Troubleshooting

**Model not found:**
```bash
# Place your YOLOv8 model in data/models/
# Update config.yaml with correct path
```

**Slow processing:**
```yaml
# Enable GPU
model:
  device: "cuda"

# Or skip frames
video:
  skip_frames: 2
```

**Import errors:**
```bash
pip install -r requirements.txt --upgrade
```

---

## 📈 Performance

| Hardware | Resolution | FPS |
|----------|-----------|-----|
| CPU (i5) | 720p | 5-10 |
| GPU (GTX 1660) | 720p | 40-60 |
| GPU (RTX 3060) | 1080p | 60-80 |

**Optimization Tips:**
- Use GPU for 10-20x speedup
- Skip frames for faster processing
- Lower confidence threshold for more detections

---

## ⚠️ Limitations

- Struggles with heavily occluded objects
- Performance degrades in poor lighting
- Works best with frontal/side camera angles
- May miss very distant workers (<20 pixels)
- Trained on specific helmet styles

---

## 🤝 Contributing

Contributions welcome! 

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file

---

## 📞 Contact

**GitHub**: [@yourusername](https://github.com/yourusername)  
**Email**: your.email@example.com

---

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [SORT Algorithm](https://github.com/abewley/sort)
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)

---

## 📚 Documentation

For detailed documentation, see:
- [Installation Guide](docs/installation.md)
- [API Reference](docs/api.md)
- [Deployment Guide](docs/deployment.md)

---

<div align="center">

**⭐ Star this repo if you find it useful!**

[Report Bug](https://github.com/yourusername/helmet-detection/issues) · 
[Request Feature](https://github.com/yourusername/helmet-detection/issues)

Made with ❤️ for Worker Safety

</div>
```

---

This short version includes:
- ✅ Quick overview
- ✅ Fast installation steps
- ✅ Simple usage examples
- ✅ Key features
- ✅ Essential configuration
- ✅ Output descriptions
- ✅ Basic troubleshooting
- ✅ Contact info

Perfect for GitHub and quick reference! 🚀