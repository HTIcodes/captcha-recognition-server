from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import tensorflow as tf
import json
from io import BytesIO
from PIL import Image
import os
import sys

app = FastAPI()

# اجازه دسترسی از اکستنشن Chrome
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# مسیرهای فایل‌ها
MODEL_PATH = "captcha_model_h5.h5"
MAPPING_PATH = "char_to_idx.json"
MASK_PATH = "mask.png"

# بررسی وجود فایل‌های ضروری
required_files = {
    "Model": MODEL_PATH,
    "Mapping": MAPPING_PATH,
    "Mask": MASK_PATH
}

for name, path in required_files.items():
    if not os.path.exists(path):
        print(f"❌ ERROR: {name} file not found: {path}")
        print(f"Please make sure '{path}' exists in the server directory.")
        sys.exit(1)

# بارگذاری مدل
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# بارگذاری mapping
try:
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)
    
    if all(isinstance(v, int) for v in mapping_data.values()):
        idx_to_char = {v: k for k, v in mapping_data.items()}
    else:
        idx_to_char = {int(k): v for k, v in mapping_data.items()}
    
    print(f"✅ Character mapping loaded ({len(idx_to_char)} classes)")
except Exception as e:
    print(f"❌ Error loading mapping: {e}")
    sys.exit(1)

# بارگذاری mask - این فایل حتماً باید وجود داشته باشه
try:
    mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise Exception("Could not read mask.png file")
    print(f"✅ Mask loaded successfully! Shape: {mask.shape}")
except Exception as e:
    print(f"❌ Error loading mask: {e}")
    sys.exit(1)


def segment_captcha_hybrid(img_array, mask, size=(28, 28)):
    """
    تقسیم تصویر CAPTCHA به کاراکترهای جداگانه
    توجه: mask حتماً باید اعمال بشه برای حذف نویزها
    """
    
    # بررسی سازگاری اندازه mask با تصویر
    if img_array.shape[:2] != mask.shape:
        # اگر اندازه‌ها یکسان نیستند، mask رو resize می‌کنیم
        mask_resized = cv2.resize(mask, (img_array.shape[1], img_array.shape[0]))
    else:
        mask_resized = mask
    
    # اعمال mask - قسمت‌های سفید mask رو سفید می‌کنیم توی تصویر اصلی
    img_array[mask_resized == 255] = (255, 255, 255)

    # تبدیل به grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    # threshold تطبیقی
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # عملیات مورفولوژی برای بهبود کیفیت
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel, iterations=1)

    # پیدا کردن contour ها
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 2 and h > 8:
            boxes.append((x, y, w, h))
    
    # مرتب‌سازی box ها از چپ به راست
    boxes = sorted(boxes, key=lambda b: b[0])

    def split_wide_box(img, box, expected_splits=2):
        """تقسیم box های عریض که احتمالاً چند کاراکتر توش هست"""
        x, y, w, h = box
        roi = img[y:y+h, x:x+w]
        vertical_sum = np.sum(roi, axis=0)
        thresh = np.max(vertical_sum) * 0.5
        
        split_indices = []
        in_space = False
        for i, val in enumerate(vertical_sum):
            if val < thresh and not in_space:
                split_indices.append(i)
                in_space = True
            elif val >= thresh:
                in_space = False
        
        if len(split_indices) == 0:
            split_indices = np.linspace(0, w, expected_splits+1, dtype=int)[1:-1]
        
        new_boxes = []
        x_prev = 0
        for sx in split_indices:
            new_boxes.append((x + x_prev, y, sx - x_prev, h))
            x_prev = sx
        new_boxes.append((x + x_prev, y, w - x_prev, h))
        return new_boxes

    # اگر box های کمی داریم، احتمالاً باید تقسیم بشن
    if len(boxes) <= 2:
        new_boxes = []
        for b in boxes:
            if b[2] > 20:  # اگر عرض بیش از 20 پیکسل بود
                new_boxes.extend(split_wide_box(processed, b, expected_splits=2))
            else:
                new_boxes.append(b)
        boxes = sorted(new_boxes, key=lambda b: b[0])
    
    # اگر هنوز کمتر از 5 کاراکتر داریم، تقسیم بیشتر
    if len(boxes) < 5:
        new_boxes = []
        for b in boxes:
            if b[2] > 15:  # threshold کمتر
                new_boxes.extend(split_wide_box(processed, b, expected_splits=2))
            else:
                new_boxes.append(b)
        boxes = sorted(new_boxes, key=lambda b: b[0])
    
    # فقط 5 box اول رو بردار (چون CAPTCHA 5 کاراکتریه)
    boxes = boxes[:5]

    # استخراج کاراکترها
    letters = []
    for (x, y, w, h) in boxes:
        roi = processed[y:y+h, x:x+w]
        roi = cv2.resize(roi, size, interpolation=cv2.INTER_AREA)
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=-1)
        letters.append(roi)
    
    return letters


@app.get("/")
async def root():
    return {
        "message": "CAPTCHA Recognition API is running!",
        "status": "ready",
        "model": "loaded",
        "mask": "loaded",
        "characters": len(idx_to_char)
    }


@app.post("/predict")
async def predict_captcha(file: UploadFile = File(...)):
    """دریافت تصویر CAPTCHA و پیش‌بینی متن"""
    try:
        # خواندن تصویر
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {
                "success": False,
                "error": "Cannot decode image"
            }

        print(f"📥 Image received: {img.shape}")
        print(f"📐 Expected mask shape: {mask.shape}")
        
        # اگر اندازه تصویر با mask یکسان نیست، resize کن
        if img.shape[:2] != mask.shape:
            print(f"⚠️ Resizing image from {img.shape[:2]} to {mask.shape}")
            img = cv2.resize(img, (mask.shape[1], mask.shape[0]))
        
        # DEBUG: ذخیره تصویر اصلی
        debug_dir = "debug_images"
        os.makedirs(debug_dir, exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"{debug_dir}/original_{timestamp}.png", img)

        # تقسیم‌بندی کاراکترها با استفاده از mask
        letters = segment_captcha_hybrid(img, mask)
        
        if len(letters) == 0:
            return {
                "success": False,
                "error": "No characters detected"
            }

        print(f"🔍 Detected {len(letters)} characters")

        # پیش‌بینی هر کاراکتر
        predicted_text = ""
        confidences = []
        
        for i, roi in enumerate(letters):
            roi_input = np.expand_dims(roi, axis=0)
            preds = model.predict(roi_input, verbose=0)
            pred_idx = np.argmax(preds)
            confidence = float(np.max(preds))
            char = idx_to_char.get(pred_idx, "?")
            
            predicted_text += char
            confidences.append(confidence)
            print(f"  Char {i+1}: '{char}' (confidence: {confidence:.2f})")

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        print(f"✅ Final prediction: {predicted_text} (avg confidence: {avg_confidence:.2f})")
        
        return {
            "success": True,
            "captcha": predicted_text,
            "length": len(predicted_text),
            "confidence": round(avg_confidence, 3),
            "characters_detected": len(letters)
        }

    except Exception as e:
        print(f"❌ Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/health")
async def health_check():
    """بررسی سلامت سرور"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "mask_loaded": mask is not None,
        "mask_shape": mask.shape if mask is not None else None,
        "num_classes": len(idx_to_char)
    }


if __name__ == "__main__":
    import uvicorn
    import os
    
    # دریافت port از environment variable (برای Render)
    port = int(os.environ.get("PORT", 8000))
    
    print("\n" + "="*50)
    print("🚀 Starting CAPTCHA Recognition Server")
    print("="*50)
    print(f"📊 Model: {MODEL_PATH}")
    print(f"🎭 Mask: {MASK_PATH}")
    print(f"🔤 Characters: {len(idx_to_char)}")
    print(f"🌐 Server running on port: {port}")
    print("="*50 + "\n")
    
    # برای production از 0.0.0.0 استفاده می‌کنیم
    uvicorn.run(app, host="0.0.0.0", port=port)