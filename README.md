# Silent Face Anti-Spoofing TFLite

## Overview
This repository provides an end-to-end solution for face anti-spoofing using MiniFASNet models. The project enables the conversion of PyTorch models (`.pth`) to TFLite, making them suitable for mobile and edge devices.

## Features
- Face anti-spoofing using MiniFASNet
- Model conversion pipeline: PyTorch -> ONNX -> TensorFlow -> TFLite
- Optimized for real-time inference
- Pretrained models included

## Repository Structure
```
Silent-Face-Anti-Spoofing-TFLite/
│── models/             # Pretrained MiniFASNet models
│── scripts/            # Scripts for model conversion
│── inference/          # Sample inference scripts
│── requirements.txt    # Dependencies
│── README.md           # Project documentation
```

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/Silent-Face-Anti-Spoofing-TFLite.git
cd Silent-Face-Anti-Spoofing-TFLite
pip install -r requirements.txt
```

## Model Conversion Pipeline
1. **PyTorch to ONNX**  
   ```bash
   python scripts/convert_to_onnx.py --model_path models/2.7_80x80_MiniFASNetV2.pth --output models/model.onnx
   ```
2. **ONNX to TensorFlow**  
   ```bash
   python scripts/convert_to_tf.py --onnx_path models/model.onnx --output models/model.pb
   ```
3. **TensorFlow to TFLite**  
   ```bash
   python scripts/convert_to_tflite.py --tf_model models/model.pb --output models/model.tflite
   ```

## Running Inference
To test the converted TFLite model:
```bash
python inference/run_tflite_inference.py --model_path models/model.tflite --input_image sample.jpg
```

## Contributing
Feel free to submit pull requests to improve the pipeline or add new features!

## License
MIT License

