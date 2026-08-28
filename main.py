
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --------------------------------------------
# 1. Load Dataset
# --------------------------------------------

df = pd.read_csv("Churn_Modelling.csv")

print(df.head())
print(df.columns)


# --------------------------------------------
# 2. Remove unnecessary columns
# --------------------------------------------

df = df.drop(
    ["RowNumber", "CustomerId", "Surname"],
    axis=1
)


# --------------------------------------------
# 3. Convert categorical columns
# --------------------------------------------

df = pd.get_dummies(
    df,
    columns=["Geography", "Gender"],
    drop_first=True
)

print(df.head())
print(df.columns)


# --------------------------------------------
# 4. Separate X and y
# --------------------------------------------

X = df.drop("Exited", axis=1)
y = df["Exited"]


# --------------------------------------------
# 5. Train-Test Split
# --------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------
# 6. Feature Scaling
# --------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# --------------------------------------------
# 7. Convert data to NumPy arrays
# --------------------------------------------

X_train = np.asarray(X_train).astype("float32")
X_test = np.asarray(X_test).astype("float32")

y_train = np.asarray(y_train).astype("float32")
y_test = np.asarray(y_test).astype("float32")


print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)


# --------------------------------------------
# 8. Create ANN Model
# --------------------------------------------

model = tf.keras.Sequential([

    # Input layer + Hidden Layer 1
    tf.keras.layers.Dense(
        16,
        activation="relu",
        input_shape=(X_train.shape[1],)
    ),

    # Hidden Layer 2
    tf.keras.layers.Dense(
        8,
        activation="relu"
    ),

    # Output Layer
    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])


# --------------------------------------------
# 9. Display Model Architecture
# --------------------------------------------

model.summary()


# --------------------------------------------
# 10. Compile Model
# --------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# --------------------------------------------
# 11. Train the ANN
# --------------------------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    verbose=1
)


# --------------------------------------------
# 12. Prediction
# --------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------
# 13. Convert probabilities to 0 or 1
# --------------------------------------------

y_pred_class = (y_pred >= 0.5).astype(int)


# --------------------------------------------
# 14. Accuracy
# --------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred_class
)

print("\nAccuracy:", accuracy)


# --------------------------------------------
# 15. Classification Report
# --------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred_class
    )
)


# --------------------------------------------
# 16. Confusion Matrix
# --------------------------------------------

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred_class
    )
)