import os
os.environ["SM_FRAMEWORK"] = "tf.keras" 

import math
from pathlib import Path
import cv2
import numpy as np
import albumentations as A
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import segmentation_models as sm

RANDOM_SEED = 42
IMG_SIZE = 256
N_CHANNELS = 3
BATCH_SIZE = 8
AUTOTUNE = tf.data.AUTOTUNE

os.makedirs("models", exist_ok=True)
checkpoint_path = "models/unet_chicken_best.h5"

BASE_DIR = Path('data')
IMAGE_DIR = BASE_DIR / 'images'
LABEL_DIR = BASE_DIR / 'labels'

def get_file_map(image_dir, label_dir):
    file_map = {}
    for image_path in image_dir.glob('*.jpg'):
        file_stem = image_path.stem
        file_map[file_stem] = {'image': str(image_path)}
    for label_path in label_dir.glob('*.txt'):
        file_stem = label_path.stem
        if file_stem in file_map:
            file_map[file_stem]['label'] = str(label_path)
    paired_files = [(data['image'], data['label']) for key, data in file_map.items()]
    return paired_files

def polygons_to_mask(label_path, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    if not Path(label_path).exists(): return mask
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            coords_normalized = np.array(parts[1:], dtype=np.float32)
            coords = coords_normalized.reshape(-1, 2)
            coords[:, 0] *= width
            coords[:, 1] *= height
            pts = coords.astype(np.int32)
            cv2.fillPoly(mask, [pts], color=(255))
    return mask

def data_generator(file_pairs):
    for img_path, mask_path in file_pairs:
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        mask = polygons_to_mask(mask_path, h, w)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        yield image, mask

def augment(image, mask, transform):
    def apply_aug(img, msk):
        augmented = transform(image=img, mask=msk)
        return augmented['image'], augmented['mask']
    image_aug, mask_aug = tf.numpy_function(func=apply_aug, inp=[image, mask], Tout=[tf.uint8, tf.uint8])
    image_aug.set_shape([IMG_SIZE, IMG_SIZE, N_CHANNELS])
    mask_aug.set_shape([IMG_SIZE, IMG_SIZE])
    image_aug = tf.cast(image_aug, tf.float32) / 255.0
    mask_aug = tf.cast(mask_aug, tf.float32) / 255.0
    mask_aug = tf.expand_dims(mask_aug, axis=-1)
    return image_aug, mask_aug

def normalize(image, mask):
    image = tf.cast(image, tf.float32) / 255.0
    mask = tf.cast(mask, tf.float32) / 255.0
    mask = tf.expand_dims(mask, axis=-1)
    return image, mask


if __name__ == "__main__":
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("1. Загрузка и разделение данных")
    file_pairs = get_file_map(IMAGE_DIR, LABEL_DIR)
    train_files, val_test_files = train_test_split(file_pairs, test_size=0.2, random_state=RANDOM_SEED)
    val_files, test_files = train_test_split(val_test_files, test_size=0.5, random_state=RANDOM_SEED)

    print("2. Создание tf.data пайплайнов")
    train_dataset = tf.data.Dataset.from_generator(lambda: data_generator(train_files), output_signature=(tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, N_CHANNELS), dtype=tf.uint8), tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE), dtype=tf.uint8)))
    val_dataset = tf.data.Dataset.from_generator(lambda: data_generator(val_files), output_signature=(tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, N_CHANNELS), dtype=tf.uint8), tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE), dtype=tf.uint8)))
    
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=25, p=0.7),
        A.OneOf([
            A.ElasticTransform(p=0.8, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1)                  
        ], p=0.8),
        A.RandomBrightnessContrast(p=0.8),    
        A.RandomGamma(p=0.8)
    ]) 
    
    train_batches = train_dataset.cache().shuffle(len(train_files)).repeat().map(lambda x, y: augment(x, y, transform), num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    val_batches = val_dataset.cache().map(normalize, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

    print("3. Сборка модели U-Net + MobileNetV2")
    model = sm.Unet('mobilenetv2', input_shape=(IMG_SIZE, IMG_SIZE, N_CHANNELS), classes=1, activation='sigmoid', encoder_weights='imagenet')
    bce_dice_loss = sm.losses.BinaryCELoss() + sm.losses.DiceLoss()
    iou_metric = sm.metrics.iou_score

    print("4. Обучение модели")
    callbacks = [
        ModelCheckpoint(filepath=checkpoint_path, monitor='val_iou_score', save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_iou_score', patience=10, mode='max', restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_iou_score', factor=0.2, patience=3, mode='max', min_lr=1e-6, verbose=1)
    ]
    
    steps_per_epoch = math.ceil(len(train_files) / BATCH_SIZE)
    validation_steps = math.ceil(len(val_files) / BATCH_SIZE)

    for layer in model.layers:
        if 'decoder' not in layer.name: layer.trainable = False
    model.compile(optimizer=Adam(learning_rate=1e-3), loss=bce_dice_loss, metrics=['accuracy', iou_metric])
    history_frozen = model.fit(train_batches, epochs=15, steps_per_epoch=steps_per_epoch, validation_data=val_batches, validation_steps=validation_steps, callbacks=callbacks)

    # Fine-tuning
    for layer in model.layers: layer.trainable = True
    model.compile(optimizer=Adam(learning_rate=1e-5), loss=bce_dice_loss, metrics=['accuracy', iou_metric])
    model.fit(train_batches, epochs=40, initial_epoch=history_frozen.epoch[-1] + 1, steps_per_epoch=steps_per_epoch, validation_data=val_batches, validation_steps=validation_steps, callbacks=callbacks)
    
    print(f"Обучение завершено. Лучшая модель сохранена в {checkpoint_path}")