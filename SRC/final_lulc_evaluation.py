# Install required library
!pip install git+https://github.com/qubvel/classification_models.git

# -----------

# Import libraries
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanIoU, Precision, Recall
from classification_models.tfkeras import Classifiers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# -----------

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# -----------

# Check dataset directories
val_img_dir = "/kaggle/input/lulc-data/data_for_training_and_testing/val/images/"
val_mask_dir = "/kaggle/input/lulc-data/data_for_training_and_testing/val/masks/"
test_img_dir = "/kaggle/input/lulc-data/data_for_training_and_testing/test/images/"
test_mask_dir = "/kaggle/input/lulc-data/data_for_training_and_testing/test/masks/"

print(f"Validation images: {len(os.listdir(val_img_dir))}")
print(f"Validation masks: {len(os.listdir(val_mask_dir))}")
print(f"Test images: {len(os.listdir(test_img_dir))}")
print(f"Test masks: {len(os.listdir(test_mask_dir))}")

# -----------

model = load_model("/kaggle/input/unet-lulc/landcover_RESNET_backbone_batch16.hdf5", compile=False)

# -----------

# Define constants
batch_size = 16
n_classes = 4
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3

# -----------

# Define preprocessing function
def preprocess_data(img, mask, num_class):
    ResNet34, preprocess_input = Classifiers.get('resnet34')
    img = img.astype(np.float32) / 255.0
    img = preprocess_input(img)
    mask = np.clip(mask, 0, num_class - 1).astype(np.uint8)
    mask = to_categorical(mask, num_class)
    return img, mask

# -----------

# Define custom generator
def custom_generator(img_dir, mask_dir, batch_size, num_class, shuffle=True):
    img_list = sorted(os.listdir(img_dir))
    mask_list = sorted(os.listdir(mask_dir))
    print(f"Found {len(img_list)} images and {len(mask_list)} masks in {img_dir}")
    if len(img_list) != len(mask_list):
        raise ValueError("Mismatch between image and mask counts")
    if len(img_list) == 0:
        raise ValueError("No images found in directory")
    while True:
        indices = list(range(len(img_list)))
        if shuffle:
            random.shuffle(indices)
        for start in range(0, len(img_list), batch_size):
            img_batch = []
            mask_batch = []
            end = min(start + batch_size, len(img_list))
            for idx in indices[start:end]:
                img_path = os.path.join(img_dir, img_list[idx])
                mask_path = os.path.join(mask_dir, mask_list[idx])
                if not os.path.exists(img_path):
                    print(f"Image file does not exist: {img_path}")
                    continue
                if not os.path.exists(mask_path):
                    print(f"Mask file does not exist: {mask_path}")
                    continue
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Failed to load image: {img_path}")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mask = cv2.imread(mask_path, 0)
                if mask is None:
                    print(f"Failed to load mask: {mask_path}")
                    continue
                img, mask = preprocess_data(img, mask, num_class)
                img_batch.append(img)
                mask_batch.append(mask)
            if img_batch:
                yield np.array(img_batch), np.array(mask_batch)
            else:
                print(f"Empty batch at start index {start}")

# Create generators
val_img_gen = custom_generator(val_img_dir, val_mask_dir, batch_size, n_classes, shuffle=True)
test_img_gen = custom_generator(test_img_dir, test_mask_dir, batch_size, n_classes, shuffle=True)

# -----------

# Verify data loading

num_samples = 2
print("Validation Samples:")
try:
    for _ in range(num_samples):
        x, y = next(val_img_gen)
        print(f"Batch shape: Images {x.shape}, Masks {y.shape}")
        image = x[0]
        mask = np.argmax(y[0], axis=2)
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.imshow(image)
        plt.title('Validation Image')
        plt.subplot(122)
        plt.imshow(mask, cmap='gray')
        plt.title('Validation Mask')
        plt.show()
except Exception as e:
    print(f"Error in validation generator: {e}")

print("Test Samples:")
try:
    for _ in range(num_samples):
        x, y = next(test_img_gen)
        print(f"Batch shape: Images {x.shape}, Masks {y.shape}")
        image = x[0]
        mask = np.argmax(y[0], axis=2)
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.imshow(image)
        plt.title('Test Image')
        plt.subplot(122)
        plt.imshow(mask, cmap='gray')
        plt.title('Test Mask')
        plt.show()
except Exception as e:
    print(f"Error in test generator: {e}")

# -----------

# Function to evaluate IoU over entire dataset
def evaluate_iou(model, generator, num_images, batch_size, n_classes):
    IOU_keras = MeanIoU(num_classes=n_classes)
    steps = (num_images + batch_size - 1) // batch_size  # Ceiling division
    for _ in range(steps):
        try:
            images, masks = next(generator)
            preds = model.predict(images, verbose=0)
            masks_argmaxed = np.argmax(masks, axis=3)
            preds_argmaxed = np.argmax(preds, axis=3)
            IOU_keras.update_state(preds_argmaxed, masks_argmaxed)
        except Exception as e:
            print(f"Error processing batch: {e}")
            continue
    return IOU_keras.result().numpy()

# -----------

# Evaluate on entire validation dataset
num_val_images = len(os.listdir(val_img_dir))
val_iou = evaluate_iou(model, val_img_gen, num_val_images, batch_size, n_classes)
print("Mean IoU (Validation, Entire Dataset) =", val_iou)

# -----------

# Visualize 5 random validation samples
try:
    for i in range(5):
        test_image_batch, test_mask_batch = val_img_gen.__next__()
        test_pred_batch = model.predict(test_image_batch, verbose=0)
        test_mask_argmaxed = np.argmax(test_mask_batch, axis=3)
        test_pred_argmaxed = np.argmax(test_pred_batch, axis=3)
        img_num = np.random.randint(0, test_image_batch.shape[0])
        plt.figure(figsize=(12, 8))
        plt.subplot(231)
        plt.title(f'Validation Image')
        plt.imshow(test_image_batch[img_num])
        plt.subplot(232)
        plt.title(f'Validation Label')
        plt.imshow(test_mask_argmaxed[img_num])
        plt.subplot(233)
        plt.title(f'Validation Prediction')
        plt.imshow(test_pred_argmaxed[img_num])
        plt.show()
except Exception as e:
    print(f"Error in validation visualization: {e}")

# -----------

# Evaluate on entire test dataset
num_test_images = len(os.listdir(test_img_dir))
test_iou = evaluate_iou(model, test_img_gen, num_test_images, batch_size, n_classes)
print("Mean IoU (Test, Entire Dataset) =", test_iou)

# -----------

# Visualize 5 random test samples
try:
    for i in range(5):
        new_test_image_batch, new_test_mask_batch = test_img_gen.__next__()
        new_test_pred_batch = model.predict(new_test_image_batch, verbose=0)
        new_test_mask_argmaxed = np.argmax(new_test_mask_batch, axis=3)
        new_test_pred_argmaxed = np.argmax(new_test_pred_batch, axis=3)
        img_num = np.random.randint(0, new_test_image_batch.shape[0])
        plt.figure(figsize=(12, 8))
        plt.subplot(231)
        plt.title(f'Test Image')
        plt.imshow(new_test_image_batch[img_num])
        plt.subplot(232)
        plt.title(f'Test Label')
        plt.imshow(new_test_mask_argmaxed[img_num])
        plt.subplot(233)
        plt.title(f'Test Prediction')
        plt.imshow(new_test_pred_argmaxed[img_num])
        plt.show()
except Exception as e:
    print(f"Error in test visualization: {e}")

# -----------

# Compare performance metrics
try:
    # Recompute loss on a single batch for consistency with previous code
    test_image_batch, test_mask_batch = val_img_gen.__next__()
    test_pred_batch = model.predict(test_image_batch, verbose=0)
    val_loss = np.mean(tf.keras.losses.categorical_crossentropy(test_mask_batch, test_pred_batch))

    new_test_image_batch, new_test_mask_batch = test_img_gen.__next__()
    new_test_pred_batch = model.predict(new_test_image_batch, verbose=0)
    test_loss = np.mean(tf.keras.losses.categorical_crossentropy(new_test_mask_batch, new_test_pred_batch))

    print("\n==================== PERFORMANCE METRICS COMPARISON ====================")
    print("Metric   |   Validation |      Test |       Diff")
    print("---------------------------------------------------------")
    print(f"Mean IoU |    {val_iou:.4f} |    {test_iou:.4f} |     {(test_iou - val_iou):.4f}")
    print(f"Loss     |    {val_loss:.4f} |    {test_loss:.4f} |     {(test_loss - val_loss):.4f}")

    metrics = ['Loss', 'Mean IoU']
    original_values = [val_loss, val_iou]
    new_values = [test_loss, test_iou]

    plt.figure(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35
    plt.bar(x - width/2, original_values, width, label='Validation', color='#B22222')
    plt.bar(x + width/2, new_values, width, label='Test', color='#D3D3D3')
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.title('Performance Comparison: Validation vs. Test Dataset')
    plt.xticks(x, metrics)
    plt.legend()
    plt.show()
except Exception as e:
    print(f"Error in performance comparison: {e}")

# -----------

