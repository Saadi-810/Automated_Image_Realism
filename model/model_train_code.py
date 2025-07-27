import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directory for saving plots
os.makedirs('evaluation_plots', exist_ok=True)

# Define the paths to your dataset folders
train_data_dir = "/home/noir/Image Realism/archive/train"  # Adjust as needed
image_size = (32, 32)
batch_size = 32

# Create data generators for training and validation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),  # Added dropout for regularization
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),   # Added dropout for regularization
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Train the model
history = model.fit(
    train_generator,
    epochs=20,  # Increased epochs
    validation_data=val_generator,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]
)

# Save the final model
model.save('final_model.h5')

# Evaluate model on validation data
print("\nEvaluating model on validation data...")
val_loss, val_accuracy, val_precision, val_recall = model.evaluate(val_generator)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Precision: {val_precision:.4f}")
print(f"Validation Recall: {val_recall:.4f}")

# Calculate predictions for confusion matrix
print("\nGenerating predictions for confusion matrix...")
val_generator.reset()
y_pred = (model.predict(val_generator) > 0.5).astype(int).flatten()
y_true = val_generator.classes

# Compute and plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('evaluation_plots/confusion_matrix.png', bbox_inches='tight', dpi=300)
plt.close()

# Plot training history
plt.figure(figsize=(15, 5))

# Accuracy plot
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Precision-Recall plot
plt.subplot(1, 3, 3)
plt.plot(history.history['precision'], label='Training Precision')
plt.plot(history.history['val_precision'], label='Validation Precision')
plt.plot(history.history['recall'], label='Training Recall')
plt.plot(history.history['val_recall'], label='Validation Recall')
plt.title('Precision and Recall')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()

plt.tight_layout()
plt.savefig('evaluation_plots/training_history.png', bbox_inches='tight', dpi=300)
plt.close()

# Save classification report
class_report = classification_report(y_true, y_pred)
print("\nClassification Report:")
print(class_report)

# Save evaluation results to file
with open('evaluation_plots/evaluation_results.txt', 'w') as f:
    f.write("Model Evaluation Results\n")
    f.write("=======================\n\n")
    f.write(f"Validation Loss: {val_loss:.4f}\n")
    f.write(f"Validation Accuracy: {val_accuracy:.4f}\n")
    f.write(f"Validation Precision: {val_precision:.4f}\n")
    f.write(f"Validation Recall: {val_recall:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write("====================\n")
    f.write(class_report)

# Save model summary
with open('evaluation_plots/model_architecture.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

print("\nEvaluation completed! Results have been saved to the 'evaluation_plots' directory.")