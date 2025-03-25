# Silent-Face-Anti-Spoofing-TFLite

## 🚀 Project Overview
Silent-Face-Anti-Spoofing-TFLite is an optimized implementation of MiniFASNet-based face anti-spoofing, designed for real-time applications on mobile and edge devices. It leverages deep learning models to detect fake faces and prevent spoofing attacks, ensuring secure authentication.

## ✨ Features
- **Lightweight & Efficient**: Optimized model for real-time inference
- **Multi-Stage Model Conversion**: PyTorch → ONNX → TensorFlow → TFLite
- **Works with Various Input Sizes (80x80)**
- **Mobile & Edge-Friendly**: Designed for Android/iOS deployment
- **Easy Integration**: SDK-compatible

---

## 📂 Project Structure
```
Silent-Face-Anti-Spoofing-TFLite/
│── models/                  # Pretrained models and converted formats
│   ├── 2.7_80x80_MiniFASNetV2.pth  # PyTorch model
│   ├── model.onnx                  # ONNX model
│   ├── tf_model/                    # TensorFlow model directory
│   ├── model.tflite                 # Final TFLite model
│
│── scripts/                 # Conversion scripts
│   ├── convert_to_onnx.py   # PyTorch → ONNX
│   ├── onnx_to_tf.py        # ONNX → TensorFlow
│   ├── tf_to_tflite.py      # TensorFlow → TFLite
│
│── data/                    # Sample test images
│── inference/               # Inference scripts and testing tools
│── README.md                # Documentation
│── requirements.txt         # Dependencies
```

---

## 🛠 Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/feni-katharotiya/Silent-Face-Anti-Spoofing-TFLite.git
cd Silent-Face-Anti-Spoofing-TFLite
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔄 Model Conversion Pipeline
### 🔹 Step 1: Convert PyTorch to ONNX
```bash
python scripts/convert_to_onnx.py --model_path models/2.7_80x80_MiniFASNetV2.pth --output models/model.onnx
```

### 🔹 Step 2: Convert ONNX to TensorFlow
```bash
python scripts/onnx_to_tf.py --onnx_model models/model.onnx --output_dir models/tf_model
```

### 🔹 Step 3: Convert TensorFlow to TFLite
```bash
python scripts/tf_to_tflite.py --tf_model models/tf_model --output models/model.tflite
```

---

## 🖥 Running Inference with TFLite
```python
import tensorflow as tf
import numpy as np

interpreter = tf.lite.Interpreter(model_path="models/model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Example input (preprocessed image)
input_data = np.random.rand(1, 80, 80, 3).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
print("Inference Output:", output)
```

---

## 📌 Notes
- The model is trained on 80x80 input images.
- Ensure your input images are normalized properly before inference.
- The final TFLite model is optimized for mobile and edge devices.

---

## 📜 License
This project follows the **MIT License**. See [LICENSE](LICENSE) for details.

## 👨‍💻 Contributors
- **Feni Katharotiya** - Project Lead & Developer

---

## 📝 Future Enhancements
- ✅ Improve model efficiency for real-time applications
- ✅ Add more dataset preprocessing techniques
- ✅ Enhance inference speed on mobile devices

---

## 🌎 Connect with Us
Have suggestions or issues? Feel free to **open an issue** or **contribute** to this repository!

