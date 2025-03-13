# IoT Device Failure Prediction

A machine learning application for predicting and simulating IoT device failures using Self-Attention Generative Adversarial Networks (SAGANs).

## Overview

This project provides a complete solution for:

1. Generating realistic IoT device failure scenarios
2. Training predictive maintenance models
3. Analyzing real device data for failure prediction
4. Visualizing failure patterns and probabilities

The application uses a SAGAN architecture to learn complex temporal patterns in sensor data, enabling it to generate diverse and realistic failure scenarios. These synthetic scenarios can then be used to train robust predictive maintenance models, especially in situations where real failure data is limited.

## Installation

### Prerequisites

- Python 3.8 or higher
- TensorFlow 2.x
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn
- Joblib

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/iot-failure-prediction.git
   cd iot-failure-prediction
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## File Structure

- `model.py` - Core implementation of SAGAN and predictive models
- `interface.py` - Application interface with CLI and API functionalities
- `output/` - Default directory for saved models and outputs
- `examples/` - Example datasets and usage demonstrations

## Usage

The application provides both a command-line interface and a Python API.

### Command-Line Interface

#### Training New Models

```bash
python interface.py train --time-steps 100 --sensors 5 --gan-epochs 500 --pred-epochs 50 --save
```

Options:
- `--time-steps`: Number of time steps in each data sample (default: 100)
- `--sensors`: Number of sensors in each data sample (default: 5)
- `--latent-dim`: Latent dimension for the generator (default: 100)
- `--gan-epochs`: Number of epochs to train the GAN (default: 500)
- `--pred-epochs`: Number of epochs to train the predictive model (default: 50)
- `--output-dir`: Directory to save outputs (default: 'output')
- `--save`: Flag to save trained models (optional)

#### Generating Failure Scenarios

```bash
python interface.py generate --model-dir output/models_20250313_123456 --num-scenarios 10 --visualize
```

Options:
- `--model-dir`: Directory containing trained models (required)
- `--num-scenarios`: Number of scenarios to generate (default: 5)
- `--visualize`: Flag to visualize generated scenarios (optional)
- `--output-dir`: Directory to save outputs (default: 'output')

#### Analyzing Real Data

```bash
python interface.py analyze --model-dir output/models_20250313_123456 --data-file sensor_data.csv --sensor-cols temp pressure vibration
```

Options:
- `--model-dir`: Directory containing trained models (required)
- `--data-file`: CSV or Excel file containing sensor data (required)
- `--time-col`: Column name containing time information (optional)
- `--sensor-cols`: Column names containing sensor data (optional)
- `--output-dir`: Directory to save outputs (default: 'output')

### Python API

You can also use the application programmatically:

```python
from interface import IoTFailurePredictionApp

# Create app instance
app = IoTFailurePredictionApp()

# Update configuration if needed
app.update_config(
    num_time_steps=100,
    num_sensors=5,
    gan_epochs=500,
    pred_epochs=50
)

# Train models
app.train_models()

# Save models
model_dir = app.save_models()

# Generate failure scenarios
scenarios, probs = app.generate_failure_scenarios(num_scenarios=5)

# Visualize scenarios
app.visualize_scenarios(scenarios, probs)

# Analyze real data
results = app.analyze_real_data("sensor_data.csv")
```

## Model Architecture

### Self-Attention GAN (SAGAN)

The model uses a Self-Attention GAN architecture with:

- A generator that produces realistic sensor time series data
- A discriminator that distinguishes between real and generated data
- Self-attention mechanisms that help capture complex temporal dependencies
- Adversarial training to improve the quality of generated samples

### Predictive Maintenance Model

The predictive model is based on an LSTM architecture to capture temporal patterns in sensor data:

- LSTM layers to process sequential sensor readings
- Dropout layers to prevent overfitting
- Dense output layer with sigmoid activation for binary classification (failure/no failure)

## Synthetic Data Generation

The application can generate various failure patterns:

1. Exponential increases (catastrophic failures)
2. Sudden spikes with increasing frequency
3. Gradual linear drift
4. Oscillations with increasing amplitude
5. Random intermittent failures

You can modify these patterns in the `generate_synthetic_iot_data` function in `model.py` to match your specific device characteristics.

## Customization

### Adding New Failure Patterns

To add new failure patterns, modify the `generate_synthetic_iot_data` function in `model.py`:

```python
if will_fail:
    # Add your custom failure pattern
    if sensor == 5:  # For a new sensor
        # Example: Stepped degradation pattern
        steps = np.random.randint(3, 6)
        for step in range(steps):
            start = int((step / steps) * num_time_steps)
            end = int(((step + 1) / steps) * num_time_steps)
            signal[start:end] += step * 0.5
```

### Adjusting Model Hyperparameters

You can adjust model hyperparameters through the configuration:

```python
app = IoTFailurePredictionApp()
app.update_config(
    gan_epochs=1000,            # More training epochs
    latent_dim=200,             # Larger latent space
    pred_batch_size=64          # Larger batch size
)
```

## Examples

### Example 1: Train Models and Analyze a Real Dataset

```bash
# Train models
python interface.py train --save

# Analyze real data
python interface.py analyze --model-dir output/models_20250313_123456 --data-file examples/pump_sensors.csv
```

### Example 2: Generate Multiple Failure Scenarios

```bash
# Generate 20 failure scenarios
python interface.py generate --model-dir output/models_20250313_123456 --num-scenarios 20 --visualize
```

## Troubleshooting

### Common Issues

1. **MemoryError during training**:
   - Reduce batch size
   - Reduce model complexity
   - Train on a machine with more memory

2. **Poor quality generated samples**:
   - Increase training epochs
   - Adjust learning rate
   - Add more training data

3. **Low prediction accuracy**:
   - Increase the number of synthetic samples
   - Adjust the predictive model architecture
   - Ensure sensor data is properly normalized

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
