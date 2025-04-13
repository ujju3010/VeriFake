# train_deepfake.py
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# === DATASET DIRECTORY ===
DATASET_DIR = "uploads/frames"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

def load_data(dataset_path):
    images, labels = [], []

    for filename in os.listdir(dataset_path):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            label = 1 if "fake" in filename.lower() else 0
            img_path = os.path.join(dataset_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(label)

    X = np.array(images, dtype="float32") / 255.0
    y = np.array(labels)
    return X, y

print("[INFO] Loading dataset...")
X, y = load_data(DATASET_DIR)
print(f"[INFO] Loaded {len(X)} images.")

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
y_train_cat = to_categorical(y_train, 2)
y_val_cat = to_categorical(y_val, 2)

# Build Model
print("[INFO] Building model...")
base_model = Xception(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
print("[INFO] Training...")
history = model.fit(X_train, y_train_cat,
                    validation_data=(X_val, y_val_cat),
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS)

# Evaluate
loss, acc = model.evaluate(X_val, y_val_cat)
print(f"[INFO] Validation Accuracy: {acc*100:.2f}%")

# Save
model.save("models/deepfake_image_model.h5")
print("[INFO] Model saved to models/deepfake_image_model.h5")

# Plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Accuracy')

plt.tight_layout()
plt.savefig("models/training_plot.png")
plt.show()
