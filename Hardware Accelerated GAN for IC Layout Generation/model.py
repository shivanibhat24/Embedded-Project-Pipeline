import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os

# Define constants
LATENT_DIM = 128
IMAGE_SIZE = 64  # Representing a simplified chip layout as 64x64
CHANNELS = 1     # Grayscale for simplicity
BATCH_SIZE = 32
EPOCHS = 5000

# Generator model
def build_generator():
    model = keras.Sequential([
        # Foundation for 16x16 feature maps
        layers.Dense(16 * 16 * 256, input_shape=(LATENT_DIM,)),
        layers.Reshape((16, 16, 256)),
        
        # Upsampling layers
        layers.Conv2DTranspose(128, 4, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        
        layers.Conv2DTranspose(64, 4, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        
        # Output layer
        layers.Conv2D(CHANNELS, 4, padding='same', activation='tanh')
    ])
    return model

# Discriminator model
def build_discriminator():
    model = keras.Sequential([
        # Input layer
        layers.Conv2D(64, 4, strides=2, padding='same', input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)),
        layers.LeakyReLU(alpha=0.2),
        
        layers.Conv2D(128, 4, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        
        layers.Conv2D(256, 4, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        
        # Classification
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Custom GAN class
class ChipGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        
    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")
        
    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]
    
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        
        # Train discriminator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        generated_images = self.generator(random_latent_vectors)
        
        real_labels = tf.ones((batch_size, 1))
        fake_labels = tf.zeros((batch_size, 1))
        
        # Add noise to labels for smoothing
        real_labels += 0.05 * tf.random.uniform(tf.shape(real_labels))
        fake_labels += 0.05 * tf.random.uniform(tf.shape(fake_labels))
        
        with tf.GradientTape() as tape:
            # Get predictions
            real_predictions = self.discriminator(real_images)
            fake_predictions = self.discriminator(generated_images)
            
            # Calculate loss
            d_loss_real = self.loss_fn(real_labels, real_predictions)
            d_loss_fake = self.loss_fn(fake_labels, fake_predictions)
            d_loss = (d_loss_real + d_loss_fake) / 2
            
        # Update discriminator weights
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        
        # Train generator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        misleading_labels = tf.ones((batch_size, 1))
        
        with tf.GradientTape() as tape:
            # Get predictions
            fake_images = self.generator(random_latent_vectors)
            fake_predictions = self.discriminator(fake_images)
            
            # Calculate loss
            g_loss = self.loss_fn(misleading_labels, fake_predictions)
            
        # Update generator weights
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        
        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result()
        }
    
    def save_models(self, save_dir):
        """Save both generator and discriminator models"""
        os.makedirs(save_dir, exist_ok=True)
        self.generator.save(os.path.join(save_dir, "generator"))
        self.discriminator.save(os.path.join(save_dir, "discriminator"))

# Add design constraints to the generator
def add_design_constraints(generated_layouts, min_width=2, min_spacing=2):
    """Apply basic design rule constraints to generated layouts"""
    # This is a simplified version - production code would implement actual DRC
    constrained = tf.identity(generated_layouts)
    
    # Ensure minimum width (simple dilation + erosion operation)
    kernel = tf.ones((min_width, min_width, 1, 1))
    dilated = tf.nn.conv2d(constrained, kernel, strides=[1, 1, 1, 1], padding='SAME')
    eroded = 1.0 - tf.nn.conv2d(1.0 - dilated, kernel, strides=[1, 1, 1, 1], padding='SAME')
    
    # Ensure minimum spacing
    spacing_kernel = tf.ones((min_spacing, min_spacing, 1, 1))
    inverted = 1.0 - eroded
    dilated_space = tf.nn.conv2d(inverted, spacing_kernel, strides=[1, 1, 1, 1], padding='SAME')
    
    # Final layout with constraints
    final_layout = tf.clip_by_value(eroded - (dilated_space - inverted), 0, 1)
    
    return final_layout

# Setup and training function
def train_gan(dataset_path, output_dir="./chip_gan_output"):
    # Load and preprocess IC layout data
    # In a real scenario, this would load actual IC layouts
    # Here we're simulating with random data
    print("Loading and preprocessing data...")
    
    # Simulated dataset - replace with actual layout loading code
    layout_dataset = np.random.uniform(-1, 1, (1000, IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
    dataset = tf.data.Dataset.from_tensor_slices(layout_dataset)
    dataset = dataset.shuffle(buffer_size=1000).batch(BATCH_SIZE)
    
    # Build and compile the GAN
    print("Building models...")
    discriminator = build_discriminator()
    generator = build_generator()
    
    gan = ChipGAN(discriminator, generator, LATENT_DIM)
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        loss_fn=keras.losses.BinaryCrossentropy()
    )
    
    # Create directories for saving results
    os.makedirs(output_dir, exist_ok=True)
    
    # Training loop with sample generation
    print("Starting training...")
    for epoch in range(EPOCHS):
        # Train for one epoch
        gan.fit(dataset, epochs=1, verbose=1)
        
        # Save sample layouts every 100 epochs
        if epoch % 100 == 0:
            # Generate and save sample layouts
            random_vectors = tf.random.normal(shape=(16, LATENT_DIM))
            generated_layouts = generator(random_vectors)
            
            # Apply design constraints
            constrained_layouts = add_design_constraints(generated_layouts)
            
            # Save sample images
            plt.figure(figsize=(4, 4))
            for i in range(16):
                plt.subplot(4, 4, i+1)
                plt.imshow(constrained_layouts[i, :, :, 0], cmap='gray')
                plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"samples_epoch_{epoch}.png"))
            plt.close()
            
        # Save models every 500 epochs
        if epoch % 500 == 0:
            gan.save_models(os.path.join(output_dir, f"models_epoch_{epoch}"))
    
    # Save final models
    gan.save_models(os.path.join(output_dir, "models_final"))
    print("Training complete!")

# Example usage
if __name__ == "__main__":
    # Replace with path to actual layout dataset
    train_gan("./layout_dataset")
