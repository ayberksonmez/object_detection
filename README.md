# 🎯 Object Detection Project
![image alt](https://github.com/ayberksonmez/object_detection/blob/main/result4.jpeg?raw=true)



A computer vision project implementing YOLO (You Only Look Once) for real-time object detection and classification.

For Live Camera

Mac

yolo predict device=mps model=my_model.pt source=0 show=True on terminal.


Windows

python yolo_detect.py --model my_model.pt --source usb0 --resolution 1280x720 on anaconda terminal.

## 🚀 Features

- **Real-time Object Detection**: Fast and accurate object detection using YOLOv11
- **Custom Model Training**: Train models on custom datasets
- **Video Processing**: Process video files for object detection
- **Multiple Output Formats**: Support for various image and video formats
- **Performance Metrics**: Comprehensive training and validation metrics

## 📊 Model Performance

- **Architecture**: YOLOv11s but prob works better on 11n didn't try it :((YOLO version 11)
- **Training Data**: Custom dataset with labeled objects
- **Validation Accuracy**: High precision and recall rates
- **Inference Speed**: Optimized for real-time processing

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Setup
```bash
# Clone the repository
git clone https://github.com/ayberksonmez/object_detection.git
cd object_detection

# Install dependencies
pip install ultralytics
pip install opencv-python
pip install torch torchvision

# Install additional requirements
pip install -r requirements.txt
```

## 🎮 Usage

### Training a Custom Model
```python
from ultralytics import YOLO

# Load a pre-trained model
model = YOLO('yolov8n.pt')

# Train on custom dataset
results = model.train(data='data.yaml', epochs=100, imgsz=640)
```

### Running Inference
```python
# Load trained model
model = YOLO('my_model/best.pt')

# Run detection on image
results = model('path/to/image.jpg')

# Run detection on video
results = model('path/to/video.mp4')
```

### Command Line Interface
```bash
# Detect objects in image
python my_model/yolo_detect.py --source image.jpg

# Detect objects in video
python my_model/yolo_detect.py --source video.mp4

# Detect objects in webcam
python my_model/yolo_detect.py --source 0
```

## 📁 Project Structure

```
object_detection/
├── my_model/                 # Trained model and scripts
│   ├── best.pt              # Best model weights
│   ├── yolo_detect.py       # Detection script
│   ├── train/               # Training results
│   └── runs/                # Detection outputs
├── data.zip                 # Training dataset
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 📈 Training Results

The model training includes:
- **Confusion Matrix**: Classification accuracy visualization
- **Precision-Recall Curves**: Performance metrics
- **Training Loss**: Learning progress tracking
- **Validation Metrics**: Model generalization assessment

## 🔧 Configuration

### Data Format
- Images: JPG, PNG formats
- Annotations: YOLO format (.txt files)
- Dataset structure: Organized in train/val/test splits

### Model Parameters
- Input size: 640x640 pixels
- Batch size: 16 (adjustable based on GPU memory)
- Learning rate: 0.01 (with cosine annealing)
- Optimizer: AdamW

## 🎯 Supported Object Classes

The model can detect and classify various object categories including:
- People
- Vehicles
- Animals
- Common objects
- Custom categories (based on training data)

## 📊 Performance Metrics

- **mAP@0.5**: Mean Average Precision at IoU 0.5
- **mAP@0.5:0.95**: Mean Average Precision across IoU thresholds
- **Precision**: True positive rate
- **Recall**: Detection rate
- **F1-Score**: Harmonic mean of precision and recall

## 🚀 Future Enhancements

- [ ] Real-time webcam detection
- [ ] Mobile app integration
- [ ] Cloud deployment
- [ ] Multi-object tracking
- [ ] Custom dataset annotation tools

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request




## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8 implementation
- [OpenCV](https://opencv.org/) for computer vision tools
- [PyTorch](https://pytorch.org/) for deep learning framework

---

⭐ **Star this repository if you found it helpful!**
