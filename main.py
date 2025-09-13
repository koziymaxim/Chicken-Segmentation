import numpy as np
import cv2
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import io

model = tf.keras.models.load_model('models/unet_chicken_best.h5', compile=False)
IMG_SIZE = 256

app = FastAPI(title="Chicken Segmentation API")

@app.post("/segment/")
async def predict(image_file: UploadFile = File(...)):
    # Чтение и декодирование изображения
    contents = await image_file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Предобработка изображения для модели
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)

    # Предсказание
    pred_mask = model.predict(img_batch)[0]
    
    # Постобработка: создание контуров и наложение на оригинал
    pred_mask_thresh = (pred_mask > 0.5).astype(np.uint8) * 255
    pred_mask_orig_size = cv2.resize(pred_mask_thresh, (img.shape[1], img.shape[0]))
    
    contours, _ = cv2.findContours(pred_mask_orig_size, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    _, img_encoded = cv2.imencode('.jpg', img)
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")