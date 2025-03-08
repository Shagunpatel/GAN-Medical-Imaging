import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import os
# Define dataset path (Modify this according to your dataset location)
DATASET_PATH = "C:\\Users\\Admin\\Python\\CVProject\\dataset\\COVID-19_Radiography_Dataset"
print(tf.__version__)
IMG_SIZE = 128  # Resize images to 128x128
BATCH_SIZE = 32  # Training batch size
print("Loading images...")
def load_images(dataset_path, img_size=IMG_SIZE):
    images = []
    categories = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

    for category in categories:
        folder_path = os.path.join(dataset_path, category)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = load_img(img_path, target_size=(img_size, img_size), color_mode="grayscale")
            img = img_to_array(img) / 255.0  # Normalize pixel values
            images.append(img)

    return np.array(images)

# Load dataset
real_images = load_images(DATASET_PATH)
print("real images done")
# Reshape data for TensorFlow
real_images = np.expand_dims(real_images, axis=-1)  # Add channel dimension

print("Dataset Loaded: ", real_images.shape)  # Expected output: (num_samples, 128, 128, 1)
def build_generator():
    print("Building Generator........")
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(8*8*256, activation="relu", input_shape=(100,)),  # Latent space input
        tf.keras.layers.Reshape((8, 8, 256)),  # Reshape to a small feature map
        tf.keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", activation="relu"),
        tf.keras.layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", activation="relu"),
        tf.keras.layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding="same", activation="relu"),
        tf.keras.layers.Conv2DTranspose(1, kernel_size=4, strides=2, padding="same", activation="tanh")  # Grayscale output
    ])
    return model

generator = build_generator()
print("Generator",generator)
generator.summary()
print(generator.summary())
def build_discriminator():
    print("Discriminator building.......")

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, padding="same", input_shape=(128, 128, 1)),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(256, kernel_size=4, strides=2, padding="same"),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation="sigmoid")  # Output: Probability of real or fake
    ])
    return model
discriminator = build_discriminator()
discriminator.trainable = True
print("dis",discriminator)
discriminator.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=["accuracy"])
discriminator.summary()
print("summary",discriminator.summary())
def build_gan(generator, discriminator):
    print('Building GAN......')
    discriminator.trainable = False  # Freeze discriminator during GAN training
    model = tf.keras.Sequential([generator, discriminator])
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))
    return model

gan = build_gan(generator, discriminator)
print("GAN",gan)
gan.summary()
print("Gan sum",gan.summary())
EPOCHS = 5000
BATCH_SIZE = 32
NOISE_DIM = 100  # Random noise vector size
SAVE_INTERVAL = 250  # Save images every 500 epochs

def train_gan(real_images, epochs=EPOCHS, batch_size=BATCH_SIZE):
    for epoch in range(epochs):
        # Train discriminator
        idx = np.random.randint(0, real_images.shape[0], batch_size)
        real_batch = real_images[idx]
        real_batch = real_batch.reshape((real_batch.shape[0], 128, 128, 1))
        noise = np.random.normal(0, 1, (batch_size, NOISE_DIM))
        fake_batch = generator.predict(noise)
        real_labels = np.ones((batch_size, 1)) * 0.9  # Real labels
        fake_labels = np.zeros((batch_size, 1))  # Fake labels
        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch(real_batch, real_labels)
        # print("d_loss_real",d_loss_real)
        # print(f"Real batch shape: {real_batch.shape}")  # Should be (32, 128, 128, 1)
        # print(f"Fake batch shape before training: {fake_batch.shape}")  # Should be (32, 128, 128, 1)
        # print(f"Discriminator input shape: {discriminator.input_shape}")  # Should be (None, 128, 128, 1)
        d_loss_fake = discriminator.train_on_batch(fake_batch, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        discriminator.trainable = False
        # Train generator
        noise = np.random.normal(0, 1, (batch_size, NOISE_DIM))
        valid_labels = np.ones((batch_size, 1))  # Trick the discriminator
        g_loss = gan.train_on_batch(noise, valid_labels)

        # Print progress
        print(f"Epoch {epoch}/{epochs}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

        # Save generated images
        if epoch % SAVE_INTERVAL == 0:
            generate_images(epoch)

def generate_images(epoch, noise_dim=NOISE_DIM):
    os.makedirs("generated_images", exist_ok=True)
    noise = np.random.normal(0, 1, (16, noise_dim))
    generated_images = generator.predict(noise)

    plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(generated_images[i].reshape(128, 128), cmap="gray")
        plt.axis("off")
    plt.savefig(f"generated_images/epoch_{epoch}.png")
    plt.close()
discriminator.trainable = False
train_gan(real_images)
