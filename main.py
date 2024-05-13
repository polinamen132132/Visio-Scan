from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import cv2
from io import BytesIO

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define class names for CIFAR-10
class_names = [
    "Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"
]

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Async function to start the bot and display a welcome message
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Welcome to the Image Classification Bot. Send /train to train the model or just send an image to classify."
    )

# Async function to train the model
async def train(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Training the model, please wait...")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    model.save('cifar_classifier.h5')  # Save the model in H5 format
    await update.message.reply_text("Model trained and saved!")
    
    category_message = "You can now send me photos from the following categories for recognition:\n- Plane\n- Car\n- Bird\n- Cat\n- Deer\n- Dog\n- Frog\n- Horse\n- Ship\n- Truck"
    await update.message.reply_text(category_message)

# Async function to handle received photos
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    photo_file = await update.message.photo[-1].get_file()
    photo_bytes = await photo_file.download_as_bytearray()
    image_stream = BytesIO(photo_bytes)
    image = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)

    # Load model for inference
    loaded_model = tf.keras.models.load_model('cifar_classifier.h5')
    prediction = loaded_model.predict(np.array([image / 255.0]))
    predicted_class = class_names[np.argmax(prediction)]
    await update.message.reply_text(f"In this image, I see a {predicted_class}.")

# Main function to run the bot
def main() -> None:
    application = Application.builder().token("5657338884:AAGcpZXSZ7uS1HfpYEdS7E5-MW0Q720UzIs").build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("train", train))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    application.run_polling()

if __name__ == "__main__":
    main()
