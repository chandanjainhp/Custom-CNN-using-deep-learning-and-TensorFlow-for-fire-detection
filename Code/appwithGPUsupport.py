import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import subprocess

# Load and preprocess image data
image_directory = '/working/data'  # Update this path as needed

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Split data into training and validation sets
)

train_generator = datagen.flow_from_directory(
    image_directory,
    target_size=(180, 180),
    batch_size=32,
    class_mode='binary',
    subset='training'  # Set as training data
)

validation_generator = datagen.flow_from_directory(
    image_directory,
    target_size=(180, 180),
    batch_size=32,
    class_mode='binary',
    subset='validation'  # Set as validation data
)

# Custom CNN Architecture for image data
def create_custom_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(180, 180, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Use sigmoid for binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

custom_cnn_model = create_custom_cnn_model()
print("Custom CNN Model Architecture:")
custom_cnn_model.summary()

# Train the Custom CNN Model
history_cnn = custom_cnn_model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Evaluate the Custom CNN Model
def evaluate_image_model(model, dataset):
    loss, accuracy = model.evaluate(dataset)
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")

evaluate_image_model(custom_cnn_model, validation_generator)

# Predict an image with the CNN model
def predict_image(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(180, 180))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create batch dimension
    predictions = model.predict(img_array)
    class_idx = int(predictions[0][0] > 0.5)  # Use threshold for binary classification
    class_label = list(train_generator.class_indices.keys())[class_idx]
    print(f"Image Classification Prediction: {class_label}")
    
    if class_label == "fire":  # Adjust this condition based on your class names
        send_alert_email()

def send_alert_email():
    # Email configuration
    fire_detection_department_email = "chandanjaincj93@gmail.com"
    subject = "Fire Detection Alert"
    body = "An alert has been triggered by the fire detection system. Please take necessary action."
    
    # Construct the email command
    command = f"echo '{body}' | mail -s '{subject}' {fire_detection_department_email}"
    
    # Send the email
    try:
        subprocess.run(command, shell=True, check=True)
        print("Email sent successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error sending email: {e}")

# Example usage
image_path = '/working/images.jpg'  # Update this path as needed
predict_image(image_path, custom_cnn_model)

# Visualizing Training History
plt.plot(history_cnn.history['accuracy'], label='accuracy')
plt.plot(history_cnn.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.grid(True)
plt.show()
