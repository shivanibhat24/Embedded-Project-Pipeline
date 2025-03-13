import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Function to create synthetic IoT sensor data
def generate_synthetic_iot_data(num_samples=1000, num_time_steps=100, num_sensors=5):
    """
    Generate synthetic IoT sensor data with degradation patterns leading to failure
    """
    data = []
    labels = []
    
    for i in range(num_samples):
        # Determine if this sample will lead to failure (70% normal, 30% failure)
        will_fail = np.random.choice([0, 1], p=[0.7, 0.3])
        
        # Create a time series for each sensor
        sample = np.zeros((num_time_steps, num_sensors))
        
        for sensor in range(num_sensors):
            # Base signal: sine wave with random frequency and phase
            base_freq = np.random.uniform(0.05, 0.1)
            phase = np.random.uniform(0, 2*np.pi)
            amplitude = np.random.uniform(0.5, 1.5)
            
            # Add baseline and normal noise
            time_points = np.arange(num_time_steps)
            signal = amplitude * np.sin(base_freq * time_points + phase)
            
            # Add sensor-specific baseline
            baseline = np.random.uniform(-2, 2)
            signal += baseline
            
            # Add random noise
            noise = np.random.normal(0, 0.1, num_time_steps)
            signal += noise
            
            if will_fail:
                # For failure cases, add degradation patterns
                # Pattern depends on which sensor it is
                if sensor == 0:
                    # Exponential increase in the last 30% of time steps
                    failure_start = int(0.7 * num_time_steps)
                    degradation = np.exp(np.linspace(0, 2, num_time_steps - failure_start))
                    signal[failure_start:] += degradation
                elif sensor == 1:
                    # Sudden spikes increasing in frequency
                    for t in range(num_time_steps):
                        if np.random.random() < (t / num_time_steps) * 0.3:
                            signal[t] += np.random.uniform(1, 3)
                elif sensor == 2:
                    # Gradual linear drift
                    drift = np.linspace(0, np.random.uniform(1.5, 3), num_time_steps)
                    signal += drift
                elif sensor == 3:
                    # Oscillation with increasing amplitude
                    oscillation = np.sin(np.linspace(0, 10*np.pi, num_time_steps))
                    envelope = np.linspace(0.1, np.random.uniform(1, 2), num_time_steps)
                    signal += oscillation * envelope
                else:
                    # Random intermittent failures
                    for t in range(num_time_steps):
                        if np.random.random() < (t / num_time_steps) * 0.2:
                            signal[t] = np.random.choice([signal[t], signal[t] * np.random.uniform(1.5, 3)])
            
            sample[:, sensor] = signal
        
        data.append(sample)
        labels.append(will_fail)
    
    return np.array(data), np.array(labels)

# Self-attention module for SAGAN
def self_attention_block(x, ch):
    """
    Self-attention block for SAGAN
    """
    batch_size, height, width, num_channels = x.shape
    
    f = layers.Conv2D(ch // 8, 1, padding='same')(x)
    g = layers.Conv2D(ch // 8, 1, padding='same')(x)
    h = layers.Conv2D(ch, 1, padding='same')(x)
    
    # Reshape for matrix multiplication
    f_flat = layers.Reshape((height * width, ch // 8))(f)
    g_flat = layers.Reshape((height * width, ch // 8))(g)
    h_flat = layers.Reshape((height * width, ch))(h)
    
    # Transpose g for matrix multiplication
    g_flat = layers.Permute((2, 1))(g_flat)
    
    # Calculate attention map
    s = layers.Dot(axes=(2, 1))([f_flat, g_flat])
    attention_map = layers.Activation('softmax')(s)
    
    # Apply attention to h
    context = layers.Dot(axes=(2, 1))([attention_map, h_flat])
    context = layers.Reshape((height, width, ch))(context)
    
    # Scale factor
    gamma = tf.Variable(0.0, trainable=True)
    
    # Add back to input
    o = layers.Add()([layers.Multiply()([gamma, context]), x])
    
    return o

# Build the SAGAN Generator
def build_generator(latent_dim, num_time_steps, num_sensors):
    """
    Build the generator model for SAGAN
    """
    noise = layers.Input(shape=(latent_dim,))
    
    # First dense layer
    x = layers.Dense(num_time_steps * num_sensors // 4)(noise)
    x = layers.LeakyReLU(0.2)(x)
    
    # Reshape to prepare for convolutional layers
    x = layers.Reshape((num_time_steps // 4, num_sensors // 1, 1))(x)
    
    # Upsampling layers
    x = layers.Conv2DTranspose(128, (4, 1), strides=(2, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Self-attention block after the first upsampling
    x = self_attention_block(x, 128)
    
    x = layers.Conv2DTranspose(64, (4, 1), strides=(2, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Output layer
    x = layers.Conv2D(1, (3, 3), padding='same', activation='tanh')(x)
    
    # Reshape to match the expected IoT data shape
    output = layers.Reshape((num_time_steps, num_sensors))(x)
    
    return models.Model(noise, output)

# Build the SAGAN Discriminator
def build_discriminator(input_shape):
    """
    Build the discriminator model for SAGAN
    """
    input_data = layers.Input(shape=input_shape)
    
    # Reshape to 2D for convolution
    x = layers.Reshape((input_shape[0], input_shape[1], 1))(input_data)
    
    # Convolutional layers
    x = layers.Conv2D(64, (4, 3), strides=(2, 1), padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv2D(128, (4, 3), strides=(2, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Self-attention block
    x = self_attention_block(x, 128)
    
    x = layers.Conv2D(256, (4, 3), strides=(2, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Flatten and output
    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)
    
    return models.Model(input_data, x)

# Build the SAGAN model
def build_sagan(generator, discriminator):
    """
    Build the combined SAGAN model for training the generator
    """
    discriminator.trainable = False
    
    z = layers.Input(shape=(generator.input_shape[1],))
    generated_data = generator(z)
    validity = discriminator(generated_data)
    
    return models.Model(z, validity)

# Training function for SAGAN
def train_sagan(generator, discriminator, combined, x_train, latent_dim, epochs=500, batch_size=32):
    """
    Train the SAGAN model
    """
    # Labels for real and fake data
    real_label = 0.9  # Smoothing for GANs
    fake_label = 0.0
    
    d_losses = []
    g_losses = []
    
    for epoch in range(epochs):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        # Select a random batch of real samples
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        real_samples = x_train[idx]
        
        # Generate a batch of fake samples
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_samples = generator.predict(noise)
        
        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(real_samples, np.ones((batch_size, 1)) * real_label)
        d_loss_fake = discriminator.train_on_batch(fake_samples, np.zeros((batch_size, 1)) * fake_label)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # ---------------------
        #  Train Generator
        # ---------------------
        
        # Generate new noise for generator training
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        
        # Train the generator
        g_loss = combined.train_on_batch(noise, np.ones((batch_size, 1)))
        
        # Store losses for plotting
        d_losses.append(d_loss)
        g_losses.append(g_loss)
        
        # Print progress
        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{epochs}, D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")
    
    return d_losses, g_losses

# Generate mixed realistic/synthetic dataset for predictive maintenance model training
def generate_mixed_dataset(real_data, generator, latent_dim, num_synthetic=500):
    """
    Generate a mixed dataset of real and synthetic data for predictive maintenance
    """
    # Generate synthetic samples
    noise = np.random.normal(0, 1, (num_synthetic, latent_dim))
    synthetic_samples = generator.predict(noise)
    
    # Generate synthetic failure labels (biased towards failures for better training)
    synthetic_labels = np.random.choice([0, 1], size=num_synthetic, p=[0.3, 0.7])
    
    # Combine real and synthetic data
    mixed_data = np.concatenate([real_data[0], synthetic_samples])
    mixed_labels = np.concatenate([real_data[1], synthetic_labels])
    
    return mixed_data, mixed_labels

# Build a simple predictive maintenance model
def build_predictive_model(input_shape):
    """
    Build a simple LSTM model for predictive maintenance
    """
    model = models.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Visualize the generated patterns
def visualize_patterns(real_samples, generated_samples, num_to_show=3):
    """
    Visualize and compare real and generated sensor patterns
    """
    fig, axes = plt.subplots(num_to_show, 2, figsize=(15, 5*num_to_show))
    
    for i in range(num_to_show):
        # Plot real sample
        for sensor in range(real_samples.shape[2]):
            axes[i, 0].plot(real_samples[i, :, sensor], label=f'Sensor {sensor+1}')
        axes[i, 0].set_title(f'Real Sample {i+1}')
        axes[i, 0].legend()
        axes[i, 0].grid(True)
        
        # Plot generated sample
        for sensor in range(generated_samples.shape[2]):
            axes[i, 1].plot(generated_samples[i, :, sensor], label=f'Sensor {sensor+1}')
        axes[i, 1].set_title(f'Generated Sample {i+1}')
        axes[i, 1].legend()
        axes[i, 1].grid(True)
    
    plt.tight_layout()
    return fig

# Main function to run the entire pipeline
def run_iot_failure_prediction_pipeline():
    # Parameters
    num_time_steps = 100
    num_sensors = 5
    latent_dim = 100
    
    # Generate synthetic training data
    print("Generating synthetic IoT sensor data...")
    real_data = generate_synthetic_iot_data(num_samples=2000, 
                                            num_time_steps=num_time_steps, 
                                            num_sensors=num_sensors)
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    x_train = real_data[0].reshape(-1, num_sensors)
    x_train = scaler.fit_transform(x_train).reshape(-1, num_time_steps, num_sensors)
    
    # Build and train SAGAN
    print("Building SAGAN models...")
    generator = build_generator(latent_dim, num_time_steps, num_sensors)
    discriminator = build_discriminator((num_time_steps, num_sensors))
    
    # Compile discriminator
    discriminator.compile(loss='binary_crossentropy', 
                          optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
    
    # Compile combined model
    combined = build_sagan(generator, discriminator)
    combined.compile(loss='binary_crossentropy', 
                     optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
    
    # Train SAGAN
    print("Training SAGAN...")
    d_losses, g_losses = train_sagan(generator, discriminator, combined, 
                                     x_train, latent_dim, epochs=500, batch_size=32)
    
    # Generate mixed dataset for predictive maintenance
    print("Generating mixed dataset for predictive maintenance...")
    mixed_data, mixed_labels = generate_mixed_dataset(real_data=(x_train, real_data[1]), 
                                                      generator=generator, 
                                                      latent_dim=latent_dim, 
                                                      num_synthetic=1000)
    
    # Build and train predictive maintenance model
    print("Building and training predictive maintenance model...")
    pred_model = build_predictive_model((num_time_steps, num_sensors))
    
    # Train-test split
    from sklearn.model_selection import train_test_split
    x_train_pred, x_test_pred, y_train_pred, y_test_pred = train_test_split(
        mixed_data, mixed_labels, test_size=0.2, random_state=42
    )
    
    # Train predictive model
    history = pred_model.fit(
        x_train_pred, y_train_pred,
        validation_data=(x_test_pred, y_test_pred),
        epochs=50, batch_size=32, verbose=2
    )
    
    # Evaluate model
    print("Evaluating predictive maintenance model...")
    test_loss, test_acc = pred_model.evaluate(x_test_pred, y_test_pred, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Generate examples of device failure patterns
    print("Generating examples of device failure patterns...")
    noise = np.random.normal(0, 1, (5, latent_dim))
    generated_samples = generator.predict(noise)
    generated_samples = scaler.inverse_transform(
        generated_samples.reshape(-1, num_sensors)
    ).reshape(-1, num_time_steps, num_sensors)
    
    # Visualize some real and generated samples
    random_real_indices = np.random.choice(x_train.shape[0], 5)
    real_samples_for_viz = scaler.inverse_transform(
        x_train[random_real_indices].reshape(-1, num_sensors)
    ).reshape(-1, num_time_steps, num_sensors)
    
    # Generate failure probability predictions for a sequence
    print("Generating failure probability over time for a sample...")
    sample_idx = np.random.choice(np.where(real_data[1] == 1)[0])
    sample = x_train[sample_idx:sample_idx+1]
    
    # Create time-windowed predictions
    window_size = 20
    probs = []
    
    for i in range(num_time_steps - window_size + 1):
        window = sample[:, i:i+window_size, :]
        prob = pred_model.predict(window, verbose=0)[0][0]
        probs.append(prob)
    
    # Plot failure probability over time
    time_points = np.arange(window_size - 1, num_time_steps)
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, probs)
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.title('Failure Probability Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Failure Probability')
    plt.grid(True)
    
    print("Pipeline completed successfully!")
    return {
        'generator': generator,
        'discriminator': discriminator, 
        'pred_model': pred_model,
        'scaler': scaler,
        'training_history': history,
        'test_accuracy': test_acc
    }

# Example usage
if __name__ == "__main__":
    models = run_iot_failure_prediction_pipeline()
    
    # Generate additional failure scenarios
    latent_dim = 100
    num_time_steps = 100
    num_sensors = 5
    
    # Generate 10 different failure scenarios
    noise = np.random.normal(0, 1, (10, latent_dim))
    generated_failures = models['generator'].predict(noise)
    
    # Inverse transform to get original scale
    generated_failures = models['scaler'].inverse_transform(
        generated_failures.reshape(-1, num_sensors)
    ).reshape(-1, num_time_steps, num_sensors)
    
    # Predict failure probability for each scenario
    failure_probs = models['pred_model'].predict(generated_failures)
    
    # Print failure probabilities
    for i, prob in enumerate(failure_probs):
        print(f"Scenario {i+1} failure probability: {prob[0]:.4f}")
