
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Set image size and batch size
image_size = (128, 128)
batch_size = 32

# Data preparation
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    r'train',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_data = datagen.flow_from_directory(
    r'test',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Build CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(train_data.num_classes, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(train_data, epochs=10, validation_data=test_data)

# Evaluate model
loss, accuracy = model.evaluate(test_data)
print(f'\nTest Accuracy: {accuracy:.4f}')
print(f'Test Loss: {loss:.4f}')

# Predictions
y_pred = model.predict(test_data)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_data.classes

# Confusion Matrix and Classification Report
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_true, y_pred_classes))

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred_classes, target_names=test_data.class_indices.keys()))

# Predict a new image
def predict_image(img_path):
    img = load_img(img_path, target_size=image_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    class_labels = list(train_data.class_indices.keys())
    predicted_label = class_labels[predicted_class]
    
    print(f'Predicted Class for the input image: {predicted_label}')
    return predicted_label

# Example: Predict on a sample image
result = predict_image(r"image path")
