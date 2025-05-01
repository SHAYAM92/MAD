import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Image dimensions
img_height, img_width = 150, 150
batch_size = 32

# Data Preprocessing and Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    r'C:\Users\DIGEESH\OneDrive\Desktop\project_v1\ds lab\Animals',  # Replace with your folder
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # Change here
    subset='training')

val_generator = train_datagen.flow_from_directory(
    r'C:\Users\DIGEESH\OneDrive\Desktop\project_v1\ds lab\Animals',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # Change here
    subset='validation',
    shuffle=False)

# Get number of classes
num_classes = len(train_generator.class_indices)

# Model Building (CNN)
model = models.Sequential([
    Input(shape=(img_height, img_width, 3)),  # Add Input layer
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # Changed to softmax
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Changed to categorical
              metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, validation_data=val_generator, epochs=10)

# Plot Accuracy and Loss
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()

# Evaluate model
val_preds = model.predict(val_generator)
val_preds_classes = np.argmax(val_preds, axis=1)  # Get predicted class index

# True labels
true_labels = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

# Classification report
print("\nClassification Report:")
print(classification_report(true_labels, val_preds_classes, target_names=class_labels))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, val_preds_classes))