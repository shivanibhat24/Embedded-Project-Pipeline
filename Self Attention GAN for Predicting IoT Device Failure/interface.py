import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import seaborn as sns
from datetime import datetime
import json

# Import functions from model.py
try:
    from model import (generate_synthetic_iot_data, build_generator, 
                      build_discriminator, build_sagan, train_sagan,
                      generate_mixed_dataset, build_predictive_model,
                      visualize_patterns, run_iot_failure_prediction_pipeline)
except ImportError:
    print("Error: model.py not found in current directory.")
    print("Please ensure model.py is in the same directory as interface.py")
    exit(1)

class IoTFailurePredictionApp:
    def __init__(self):
        self.models = None
        self.config = {
            "num_time_steps": 100,
            "num_sensors": 5,
            "latent_dim": 100,
            "training_samples": 2000,
            "synthetic_samples": 1000,
            "gan_epochs": 500,
            "gan_batch_size": 32,
            "pred_epochs": 50,
            "pred_batch_size": 32,
            "output_dir": "output"
        }
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.config["output_dir"]):
            os.makedirs(self.config["output_dir"])
    
    def train_models(self, show_progress=True):
        """Train both the SAGAN and predictive maintenance models"""
        if show_progress:
            print("Starting training pipeline...")
            
        self.models = run_iot_failure_prediction_pipeline()
        
        if show_progress:
            print(f"Models trained successfully. Test accuracy: {self.models['test_accuracy']:.4f}")
        
        return self.models
    
    def save_models(self, output_dir=None):
        """Save trained models and configuration"""
        if self.models is None:
            print("Error: No trained models found. Run train_models() first.")
            return False
        
        if output_dir is None:
            output_dir = self.config["output_dir"]
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(output_dir, f"models_{timestamp}")
        os.makedirs(model_dir)
        
        # Save models
        self.models["generator"].save(os.path.join(model_dir, "generator_model"))
        self.models["discriminator"].save(os.path.join(model_dir, "discriminator_model"))
        self.models["pred_model"].save(os.path.join(model_dir, "predictive_model"))
        
        # Save scaler
        joblib.dump(self.models["scaler"], os.path.join(model_dir, "scaler.pkl"))
        
        # Save configuration
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=4)
            
        print(f"Models and configuration saved to {model_dir}")
        return model_dir
    
    def load_models(self, model_dir):
        """Load trained models and configuration from directory"""
        if not os.path.exists(model_dir):
            print(f"Error: Model directory {model_dir} not found.")
            return False
            
        try:
            # Load models
            generator = load_model(os.path.join(model_dir, "generator_model"))
            discriminator = load_model(os.path.join(model_dir, "discriminator_model"))
            pred_model = load_model(os.path.join(model_dir, "predictive_model"))
            
            # Load scaler
            scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
            
            # Load configuration
            with open(os.path.join(model_dir, "config.json"), "r") as f:
                self.config = json.load(f)
                
            self.models = {
                "generator": generator,
                "discriminator": discriminator,
                "pred_model": pred_model,
                "scaler": scaler
            }
            
            print(f"Models and configuration successfully loaded from {model_dir}")
            return True
        
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False
    
    def generate_failure_scenarios(self, num_scenarios=5, save_to_csv=True):
        """Generate failure scenarios using the trained generator"""
        if self.models is None:
            print("Error: No trained models found. Run train_models() or load_models() first.")
            return None
            
        print(f"Generating {num_scenarios} failure scenarios...")
        
        # Generate random noise for the generator
        noise = np.random.normal(0, 1, (num_scenarios, self.config["latent_dim"]))
        
        # Generate failure scenarios
        generated_failures = self.models["generator"].predict(noise)
        
        # Inverse transform to get original scale
        generated_failures = self.models["scaler"].inverse_transform(
            generated_failures.reshape(-1, self.config["num_sensors"])
        ).reshape(-1, self.config["num_time_steps"], self.config["num_sensors"])
        
        # Predict failure probability for each scenario
        failure_probs = self.models["pred_model"].predict(generated_failures)
        
        # Save scenarios to CSV if requested
        if save_to_csv:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            scenario_dir = os.path.join(self.config["output_dir"], f"scenarios_{timestamp}")
            os.makedirs(scenario_dir)
            
            # Save individual scenarios
            for i in range(num_scenarios):
                scenario_df = pd.DataFrame(generated_failures[i])
                scenario_df.columns = [f"Sensor_{j+1}" for j in range(self.config["num_sensors"])]
                scenario_df["Time"] = range(self.config["num_time_steps"])
                
                # Save to CSV
                file_path = os.path.join(scenario_dir, f"scenario_{i+1}_prob_{failure_probs[i][0]:.4f}.csv")
                scenario_df.to_csv(file_path, index=False)
                
            print(f"Scenarios saved to {scenario_dir}")
        
        return generated_failures, failure_probs
    
    def visualize_scenarios(self, scenarios=None, failure_probs=None, num_to_show=3, save_fig=True):
        """Visualize generated failure scenarios"""
        if scenarios is None:
            if self.models is None:
                print("Error: No trained models found. Run train_models() or load_models() first.")
                return
                
            # Generate new scenarios if none provided
            scenarios, failure_probs = self.generate_failure_scenarios(num_to_show, save_to_csv=False)
        
        # Limit the number to show if more were provided
        if num_to_show > scenarios.shape[0]:
            num_to_show = scenarios.shape[0]
            
        # Create figure for visualization
        fig, axes = plt.subplots(num_to_show, 1, figsize=(12, 5*num_to_show))
        if num_to_show == 1:
            axes = [axes]
            
        # Plot each scenario
        for i in range(num_to_show):
            for sensor in range(self.config["num_sensors"]):
                axes[i].plot(scenarios[i, :, sensor], label=f'Sensor {sensor+1}')
                
            prob_text = f" (Failure Probability: {failure_probs[i][0]:.4f})" if failure_probs is not None else ""
            axes[i].set_title(f'Generated Scenario {i+1}{prob_text}')
            axes[i].legend()
            axes[i].grid(True)
            axes[i].set_xlabel('Time Step')
            axes[i].set_ylabel('Sensor Value')
            
        plt.tight_layout()
        
        # Save figure if requested
        if save_fig:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fig_path = os.path.join(self.config["output_dir"], f"scenarios_viz_{timestamp}.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {fig_path}")
            
        plt.show()
        return fig
    
    def analyze_real_data(self, data_file, time_col=None, sensor_cols=None, visualize=True):
        """Analyze real sensor data for failure prediction"""
        if self.models is None:
            print("Error: No trained models found. Run train_models() or load_models() first.")
            return None
        
        # Load data from file
        try:
            if data_file.endswith('.csv'):
                df = pd.read_csv(data_file)
            elif data_file.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(data_file)
            else:
                print(f"Unsupported file format: {data_file}")
                return None
        except Exception as e:
            print(f"Error loading data file: {str(e)}")
            return None
        
        # Determine sensor columns if not specified
        if sensor_cols is None:
            # Try to guess sensor columns (exclude time and any obvious non-numeric columns)
            sensor_cols = [col for col in df.columns if col != time_col and 
                          df[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
            
            if len(sensor_cols) == 0:
                print("Error: Could not determine sensor columns. Please specify sensor_cols.")
                return None
            elif len(sensor_cols) != self.config["num_sensors"]:
                print(f"Warning: Found {len(sensor_cols)} sensor columns, but model expects {self.config['num_sensors']}.")
                if len(sensor_cols) > self.config["num_sensors"]:
                    print(f"Using first {self.config['num_sensors']} columns: {sensor_cols[:self.config['num_sensors']]}")
                    sensor_cols = sensor_cols[:self.config["num_sensors"]]
                else:
                    print("Not enough sensor columns found. Cannot proceed with analysis.")
                    return None
        
        # Extract sensor data
        sensor_data = df[sensor_cols].values
        
        # Check if we have enough time steps
        if sensor_data.shape[0] < self.config["num_time_steps"]:
            print(f"Warning: Data has only {sensor_data.shape[0]} time steps, but model expects {self.config['num_time_steps']}.")
            print("Padding data with zeros.")
            padding = np.zeros((self.config["num_time_steps"] - sensor_data.shape[0], len(sensor_cols)))
            sensor_data = np.vstack([sensor_data, padding])
        elif sensor_data.shape[0] > self.config["num_time_steps"]:
            print(f"Warning: Data has {sensor_data.shape[0]} time steps, but model expects {self.config['num_time_steps']}.")
            print(f"Using first {self.config['num_time_steps']} time steps.")
            sensor_data = sensor_data[:self.config["num_time_steps"], :]
        
        # Scale the data
        sensor_data_scaled = self.models["scaler"].transform(sensor_data)
        
        # Reshape for prediction
        sensor_data_scaled = sensor_data_scaled.reshape(1, self.config["num_time_steps"], self.config["num_sensors"])
        
        # Make prediction
        failure_prob = self.models["pred_model"].predict(sensor_data_scaled)[0][0]
        
        print(f"Overall failure probability: {failure_prob:.4f}")
        
        # Create time-windowed predictions
        window_size = 20
        if sensor_data.shape[0] >= window_size:
            print("Generating time-window failure predictions...")
            probs = []
            
            for i in range(sensor_data.shape[0] - window_size + 1):
                window = sensor_data_scaled[:, i:i+window_size, :]
                prob = self.models["pred_model"].predict(window, verbose=0)[0][0]
                probs.append(prob)
            
            # Plot failure probability over time if requested
            if visualize:
                time_points = np.arange(window_size - 1, sensor_data.shape[0])
                plt.figure(figsize=(12, 6))
                plt.plot(time_points, probs)
                plt.axhline(y=0.5, color='r', linestyle='--', label='Failure Threshold')
                plt.title('Failure Probability Over Time')
                plt.xlabel('Time Step')
                plt.ylabel('Failure Probability')
                plt.grid(True)
                plt.legend()
                
                # Save visualization
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                fig_path = os.path.join(self.config["output_dir"], f"failure_prob_{timestamp}.png")
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                print(f"Visualization saved to {fig_path}")
                
                plt.show()
            
            return {
                'overall_probability': failure_prob,
                'time_probabilities': probs,
                'sensor_data': sensor_data
            }
        
        return {
            'overall_probability': failure_prob,
            'sensor_data': sensor_data
        }

    def update_config(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
            else:
                print(f"Warning: Unknown configuration parameter '{key}'")
                
        print("Configuration updated.")
        return self.config


def main():
    """Main function to run the interface from command line"""
    parser = argparse.ArgumentParser(description='IoT Device Failure Prediction Interface')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--time-steps', type=int, default=100,
                             help='Number of time steps in each sample')
    train_parser.add_argument('--sensors', type=int, default=5,
                             help='Number of sensors in each sample')
    train_parser.add_argument('--latent-dim', type=int, default=100,
                             help='Latent dimension for the generator')
    train_parser.add_argument('--gan-epochs', type=int, default=500,
                             help='Number of epochs to train the GAN')
    train_parser.add_argument('--pred-epochs', type=int, default=50,
                             help='Number of epochs to train the predictive model')
    train_parser.add_argument('--output-dir', type=str, default='output',
                             help='Directory to save outputs')
    train_parser.add_argument('--save', action='store_true',
                             help='Save trained models')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate failure scenarios')
    gen_parser.add_argument('--model-dir', type=str, required=True,
                           help='Directory containing trained models')
    gen_parser.add_argument('--num-scenarios', type=int, default=5,
                           help='Number of scenarios to generate')
    gen_parser.add_argument('--visualize', action='store_true',
                           help='Visualize generated scenarios')
    gen_parser.add_argument('--output-dir', type=str, default='output',
                           help='Directory to save outputs')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze real data')
    analyze_parser.add_argument('--model-dir', type=str, required=True,
                              help='Directory containing trained models')
    analyze_parser.add_argument('--data-file', type=str, required=True,
                              help='CSV or Excel file containing sensor data')
    analyze_parser.add_argument('--time-col', type=str, default=None,
                              help='Column name containing time information')
    analyze_parser.add_argument('--sensor-cols', type=str, nargs='+', default=None,
                              help='Column names containing sensor data')
    analyze_parser.add_argument('--output-dir', type=str, default='output',
                              help='Directory to save outputs')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create application instance
    app = IoTFailurePredictionApp()
    
    # Process commands
    if args.command == 'train':
        # Update configuration
        app.update_config(
            num_time_steps=args.time_steps,
            num_sensors=args.sensors,
            latent_dim=args.latent_dim,
            gan_epochs=args.gan_epochs,
            pred_epochs=args.pred_epochs,
            output_dir=args.output_dir
        )
        
        # Train models
        app.train_models()
        
        # Save models if requested
        if args.save:
            app.save_models()
            
        # Generate and visualize a few examples
        app.visualize_scenarios(num_to_show=3)
        
    elif args.command == 'generate':
        # Update output directory
        app.update_config(output_dir=args.output_dir)
        
        # Load models
        if app.load_models(args.model_dir):
            # Generate scenarios
            scenarios, probs = app.generate_failure_scenarios(num_scenarios=args.num_scenarios)
            
            # Visualize if requested
            if args.visualize:
                app.visualize_scenarios(scenarios, probs)
                
    elif args.command == 'analyze':
        # Update output directory
        app.update_config(output_dir=args.output_dir)
        
        # Load models
        if app.load_models(args.model_dir):
            # Process sensor column names if provided
            sensor_cols = args.sensor_cols
            
            # Analyze data
            app.analyze_real_data(args.data_file, args.time_col, sensor_cols)
            
    else:
        parser.print_help()

# Simple usage example
if __name__ == "__main__":
    main()
