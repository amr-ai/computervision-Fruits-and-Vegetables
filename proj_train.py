import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Load the training and validation datasets
training_set = image_dataset_from_directory(
    "E:/codes/Python/data/train",
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

validation_set = image_dataset_from_directory(
    'E:/codes/Python/data/validation',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

# ==========================================================================(Deep Neural Network)============================================================
dnn = tf.keras.models.Sequential()

# Building Convolution Layer
dnn.add(layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=[64, 64, 3]))
dnn.add(layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
dnn.add(layers.MaxPool2D(pool_size=2, strides=2))
dnn.add(layers.Dropout(0.25))
dnn.add(layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
dnn.add(layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
dnn.add(layers.MaxPool2D(pool_size=2, strides=2))
dnn.add(layers.Dropout(0.25))
dnn.add(layers.Flatten())
dnn.add(layers.Dense(units=512, activation='relu'))
dnn.add(layers.Dropout(0.5))  # To avoid overfitting
dnn.add(layers.Dense(units=256, activation='relu'))
dnn.add(layers.Dropout(0.5))  # To avoid overfitting
dnn.add(layers.Dense(units=128, activation='relu'))
dnn.add(layers.Dropout(0.5))  # To avoid overfitting

# Output Layer
dnn.add(layers.Dense(units=22, activation='softmax'))

dnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
dnn.summary()

# Calculate the number of batches
steps_per_epoch = len(training_set)

# Train the model
training_history_dnn = dnn.fit(training_set, validation_data=validation_set, epochs=50, steps_per_epoch=steps_per_epoch)

# Evaluating Model
# Training set Accuracy
train_loss, train_acc = dnn.evaluate(training_set)
print('Training accuracy:', train_acc)

# Validation set Accuracy
val_loss, val_acc = dnn.evaluate(validation_set)
print('Validation accuracy:', val_acc)

# Save the model
dnn.save('trained_model_dnn.h5')

# Display training history
print(training_history_dnn.history)
