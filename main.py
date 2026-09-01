import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("cat_dog_pixel_values.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())


# ==========================================
# 2. SEPARATE X AND y
# ==========================================

X = df.drop("label", axis=1)

y = df["label"]


# ==========================================
# 3. LABEL ENCODING
# ==========================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

print("\nClasses:", label_encoder.classes_)
print("Encoded labels:", np.unique(y))
print("y dtype:", y.dtype)


# ==========================================
# 4. CONVERT PIXELS TO NUMERIC
# ==========================================

X = X.apply(pd.to_numeric, errors="coerce")

print("\nMissing values:", X.isnull().sum().sum())

X = X.fillna(0)

X = X.to_numpy(dtype=np.float32)

print("X dtype:", X.dtype)
print("X shape:", X.shape)


# ==========================================
# 5. NORMALIZE PIXELS
# ==========================================

X = X / 255.0


# ==========================================
# 6. RESHAPE
# ==========================================

X = X.reshape(-1, 32, 32, 1)

print("\nCNN Input Shape:", X.shape)


# ==========================================
# 7. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)


print("Training Labels:", y_train.shape)
print("Testing Labels:", y_test.shape)


# ==========================================
# 9. BUILD CNN MODEL
# ==========================================

model = Sequential([

    # --------------------------------------
    # First Convolution Layer
    # --------------------------------------

    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=(32, 32, 1)
    ),

    # --------------------------------------
    # First Pooling Layer
    # --------------------------------------

    MaxPooling2D(
        pool_size=(2, 2)
    ),

    # --------------------------------------
    # Second Convolution Layer
    # --------------------------------------

    Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu"
    ),

    # --------------------------------------
    # Second Pooling Layer
    # --------------------------------------

    MaxPooling2D(
        pool_size=(2, 2)
    ),

    # --------------------------------------
    # Flatten
    # --------------------------------------

    Flatten(),

    # --------------------------------------
    # Fully Connected Layer
    # --------------------------------------

    Dense(
        128,
        activation="relu"
    ),

    # --------------------------------------
    # Dropout
    # --------------------------------------

    Dropout(0.3),

    # --------------------------------------
    # Output Layer
    # --------------------------------------

    Dense(
        1,
        activation="sigmoid"
    )
])


# ==========================================
# 10. COMPILE MODEL
# ==========================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# 11. MODEL SUMMARY
# ==========================================

print("\nMODEL SUMMARY\n")

model.summary()


# ==========================================
# 12. TRAIN MODEL
# ==========================================

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_test,
        y_test
    ),

    epochs=10,

    batch_size=32
)


# ==========================================
# 13. EVALUATE MODEL
# ==========================================

loss, accuracy = model.evaluate(
    X_test,
    y_test
)


print("\nTest Loss:", loss)

print("Test Accuracy:", accuracy)

# ==========================================
# PREDICT NEW IMAGE
# ==========================================

from tensorflow.keras.preprocessing import image

image_path = "dog.jpeg"

# Load image exactly like training images
img = image.load_img(
    image_path,
    target_size=(32, 32),
    color_mode="grayscale"
)

# Convert image to NumPy
img_array = image.img_to_array(img)

# Normalize
img_array = img_array / 255.0

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

print("\nInput image shape:", img_array.shape)

# Prediction
prediction = model.predict(img_array, verbose=0)

probability = prediction[0][0]

print("Raw prediction:", probability)

# Get class names from LabelEncoder
if probability >= 0.5:
    predicted_index = 1
else:
    predicted_index = 0

predicted_class = label_encoder.inverse_transform(
    [predicted_index]
)[0]

print("Predicted class:", predicted_class)