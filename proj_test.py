import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Load the test set
test_set = tf.keras.utils.image_dataset_from_directory(
    'E:/codes/Python/test',
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

# Load the trained model
cnn = tf.keras.models.load_model('E:/codes/Python/project/trained_model.h5')

# Choose an image interactively using a file dialog
Tk().withdraw()  # Hide the main window
image_path = askopenfilename(title="Select an Image")

# Reading an image in default mode
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converting BGR to RGB

# Displaying the image
plt.imshow(img)
plt.xticks([])
plt.yticks([])

# Load and preprocess the image for prediction
image = tf.keras.preprocessing.image.load_img(image_path, target_size=(64, 64))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.

# Make predictions
predictions = cnn.predict(input_arr)
result_index = np.argmax(predictions)  # Return index of max element

# Display the predicted class name on the image
predicted_class = test_set.class_names[result_index]
plt.text(4, 4, f'name: {predicted_class}', color='white', fontsize=15, bbox=dict(facecolor='black', alpha=1))

# Show the image with the predicted class name
plt.show()

# Single image Prediction
print("It's a {}".format(predicted_class))
