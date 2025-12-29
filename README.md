# CAPTCHA Recognition Server

AI-powered CAPTCHA recognition API using FastAPI and TensorFlow for Golestan university system.

---

## 📋 Overview

This is a FastAPI-based REST API that recognizes 5-character CAPTCHAs from the Golestan university portal using a trained TensorFlow/Keras CNN model.

**Key Features:**
- ✅ Fast CAPTCHA recognition (5-10 seconds)
- ✅ ~95% accuracy on Golestan CAPTCHAs
- ✅ RESTful API with CORS support
- ✅ Automatic character segmentation
- ✅ Production-ready with health checks

---

## 🏗️ Architecture

```
┌─────────────┐
│   Client    │
│ (Extension) │
└──────┬──────┘
       │ POST /predict
       │ (CAPTCHA image)
       ▼
┌─────────────────────────────────┐
│      FastAPI Server             │
├─────────────────────────────────┤
│  1. Image preprocessing         │
│  2. Mask application            │
│  3. Character segmentation      │
│  4. CNN prediction              │
└──────┬──────────────────────────┘
       │
       ▼
  Predicted Text
```

---

## 📁 Project Structure

```
captcha-recognition-server/
├── app.py                  # Main FastAPI application
├── requirements.txt        # Python dependencies
├── captcha_model_h5.h5    # Trained CNN model
├── char_to_idx.json       # Character mapping
├── mask.png               # Image mask for preprocessing
└── README.md              # This file
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- pip

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/HTIcodes/captcha-recognition-server.git
   cd captcha-recognition-server
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure required files exist**
   - `captcha_model_h5.h5` - The trained model
   - `char_to_idx.json` - Character index mapping
   - `mask.png` - Image preprocessing mask

4. **Run the server**
   ```bash
   python app.py
   ```

5. **Test the API**
   ```bash
   curl http://localhost:8000/
   ```

---

## 🌐 Deployment

### Deploy to Render.com (Recommended)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Create Web Service on Render**
   - Go to [render.com](https://render.com)
   - Connect your GitHub repository
   - Configure:
     - **Build Command:** `pip install -r requirements.txt`
     - **Start Command:** `python app.py`
     - **Instance Type:** Free

3. **Environment Variables**
   - `PORT` - Auto-set by Render (default: 10000)
   - `PYTHON_VERSION` - `3.12.0` (optional)

4. **Deploy!**
   - Render will automatically deploy your app
   - You'll get a URL like: `https://your-app.onrender.com`

---

## 📡 API Reference

### `GET /`

Health check endpoint.

**Response:**
```json
{
  "message": "CAPTCHA Recognition API is running!",
  "status": "ready",
  "model": "loaded",
  "mask": "loaded",
  "characters": 54
}
```

---

### `POST /predict`

Predict CAPTCHA text from an image.

**Request:**
- **Content-Type:** `multipart/form-data`
- **Body:** 
  - `file` (required) - CAPTCHA image file (PNG/JPG)

**Example (cURL):**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@captcha.png"
```

**Example (Python):**
```python
import requests

with open('captcha.png', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f}
    )
    print(response.json())
```

**Success Response (200 OK):**
```json
{
  "success": true,
  "captcha": "A3B7K",
  "length": 5,
  "confidence": 0.943,
  "characters_detected": 5
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "Cannot decode image"
}
```

---

### `GET /health`

Detailed health check with model status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "mask_loaded": true,
  "mask_shape": [50, 140],
  "num_classes": 54
}
```

---

## 🧪 Testing

### Test with sample image

```bash
# Download a test CAPTCHA
curl -o test_captcha.png https://golestan.ikiu.ac.ir/.../captcha.aspx

# Test prediction
curl -X POST http://localhost:8000/predict \
  -F "file=@test_captcha.png"
```

### Expected output

```json
{
  "success": true,
  "captcha": "XYZ12",
  "length": 5,
  "confidence": 0.95,
  "characters_detected": 5
}
```

---

## 🔧 Configuration

### Model Details

- **Architecture:** CNN (Convolutional Neural Network)
- **Input Shape:** 28×28 grayscale images
- **Output Classes:** 54 characters (digits 1-9, letters A-Z excluding I,J,L,O)
- **Training Accuracy:** ~95%

### Character Set

```
Digits: 1, 2, 3, 4, 5, 6, 7, 8, 9
Uppercase: A, B, C, D, E, F, G, H, K, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z
Lowercase: a, b, c, d, e, f, g, h, k, m, n, p, q, r, s, t, u, v, w, x, y, z

Excluded: 0, I, J, L, O (to avoid confusion)
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Average Response Time | 5-10 seconds |
| First Request (Cold Start) | 30-60 seconds |
| Model Accuracy | ~95% |
| Supported Image Formats | PNG, JPG |
| Max Image Size | 10 MB |

---

## 🐛 Troubleshooting

### Issue: Model fails to load

**Error:** `❌ Error loading model`

**Solution:**
- Ensure `captcha_model_h5.h5` exists in the root directory
- Check file isn't corrupted
- Verify TensorFlow version compatibility

---

### Issue: Characters not detected

**Error:** `"characters_detected": 0`

**Solution:**
- Ensure CAPTCHA image is clear and not corrupted
- Check image dimensions (should be close to 140×50)
- Verify mask.png is correct

---

### Issue: Wrong predictions

**Possible causes:**
- Image size mismatch (automatically resized)
- Poor image quality
- Different CAPTCHA format

**Solution:**
- Check server logs for segmentation details
- Ensure CAPTCHA is from Golestan system
- Review `debug_images/` folder for processed images

---

### Issue: Server timeout on Render

**Error:** Cold start takes too long

**Solution:**
- This is normal for free tier
- First request after 15 min idle: 30-60s
- Subsequent requests: 5-10s
- Use UptimeRobot to keep server warm

---

## 🔐 Security Notes

- ⚠️ This API has CORS enabled for all origins (`*`)
- ⚠️ No authentication required (suitable for internal use)
- ⚠️ For production, consider adding API keys
- ⚠️ Rate limiting not implemented

---

## 📈 Monitoring

### Server Logs

Check Render dashboard for:
- Request logs
- Error traces
- Model predictions
- Performance metrics

### Key Metrics to Watch

```
📥 Image received: (50, 140, 3)
🔍 Detected 5 characters
  Char 1: 'A' (confidence: 0.98)
  ...
✅ Final prediction: ABC12 (avg confidence: 0.94)
```

---

## 🛠️ Development

### Running Locally

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run with auto-reload
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Adding New Features

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

---

## 📝 Requirements

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
opencv-python-headless==4.8.1.78
numpy==1.26.4
tensorflow==2.16.2
Pillow==10.1.0
protobuf==3.20.3
h5py==3.10.0
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is for educational purposes. Use responsibly.

---

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- ML powered by [TensorFlow](https://www.tensorflow.org/)
- Image processing with [OpenCV](https://opencv.org/)
- Deployed on [Render](https://render.com/)

---

## 📧 Contact

For questions or issues:
- Create an issue on GitHub
- Contact: mahyarhemati84@gmail.com

---

## 🔗 Related Projects

- [Chrome Extension](https://github.com/HTIcodes/golestan-captcha-extension) - User-facing extension

---

**Built with ❤️ for students**

*Last updated: December 2024*
