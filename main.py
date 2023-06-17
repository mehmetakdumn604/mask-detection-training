import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define directories
data_dir = 'data'

#  Create a data generator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    directory=data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training')

val_generator = datagen.flow_from_directory(
    directory=data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation')

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # output layer size is 3 as we have 3 classes
])

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=5, validation_data=val_generator)

# Convert the model to the TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model to disk
with open("model_unquant.tflite", "wb") as f:
    f.write(tflite_model)

# Save labels to file
indices_keys = ["incorrect_mask","with_mask", "without_mask"]
index = 0
with open("labels.txt", "w") as f:
    f.write('\n'.join(indices_keys[index:]))
    index += 1
    
