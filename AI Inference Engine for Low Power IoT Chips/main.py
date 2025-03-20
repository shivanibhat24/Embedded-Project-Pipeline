"""
Lightweight AI Inference Engine for Low-Power Chips
=================================================
A simple, efficient neural network inference engine optimized for
low-power chips and microcontrollers with minimal dependencies.
"""

import numpy as np
import json
import time
import os
from typing import Dict, List, Tuple, Union, Optional

class LightweightInferenceEngine:
    """
    A lightweight neural network inference engine optimized for low-power devices.
    Supports basic feed-forward neural networks with common activation functions.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Optional path to a pre-trained model file.
        """
        self.layers = []
        self.activations = []
        self.model_info = {}
        
        # Available activation functions
        self.activation_functions = {
            'relu': self._relu,
            'sigmoid': self._sigmoid,
            'tanh': self._tanh,
            'softmax': self._softmax,
            'linear': self._linear
        }
        
        if model_path:
            self.load_model(model_path)
    
    # Activation functions
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function: max(0, x)"""
        return np.maximum(0, x)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function: 1 / (1 + exp(-x))"""
        # Use clipping to prevent overflow
        x = np.clip(x, -15, 15)
        return 1 / (1 + np.exp(-x))
    
    def _tanh(self, x: np.ndarray) -> np.ndarray:
        """Hyperbolic tangent activation function"""
        return np.tanh(x)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation function"""
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _linear(self, x: np.ndarray) -> np.ndarray:
        """Linear activation (identity function)"""
        return x
    
    def load_model(self, model_path: str) -> None:
        """
        Load a model from a JSON file.
        
        Args:
            model_path: Path to the model JSON file.
        """
        try:
            with open(model_path, 'r') as f:
                model_data = json.load(f)
            
            self.model_info = model_data.get('info', {})
            
            # Load weights and biases
            self.layers = []
            self.activations = []
            
            for layer in model_data['layers']:
                # Convert lists to numpy arrays
                weights = np.array(layer['weights'], dtype=np.float32)
                bias = np.array(layer['bias'], dtype=np.float32)
                
                self.layers.append((weights, bias))
                self.activations.append(layer['activation'])
                
            print(f"Model loaded: {self.model_info.get('name', 'unnamed model')}")
            print(f"Input shape: {self.layers[0][0].shape[0]}")
            print(f"Output shape: {self.layers[-1][0].shape[0]}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def save_model(self, model_path: str, model_info: Optional[Dict] = None) -> None:
        """
        Save the current model to a JSON file.
        
        Args:
            model_path: Path where the model will be saved.
            model_info: Optional additional information about the model.
        """
        if model_info:
            self.model_info.update(model_info)
        
        model_data = {
            'info': self.model_info,
            'layers': []
        }
        
        for i, ((weights, bias), activation) in enumerate(zip(self.layers, self.activations)):
            layer_data = {
                'weights': weights.tolist(),
                'bias': bias.tolist(),
                'activation': activation
            }
            model_data['layers'].append(layer_data)
        
        try:
            with open(model_path, 'w') as f:
                json.dump(model_data, f)
            print(f"Model saved to {model_path}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            raise
    
    def predict(self, inputs: np.ndarray, quantized: bool = False) -> np.ndarray:
        """
        Run inference on the input data.
        
        Args:
            inputs: Input data as a numpy array of shape (batch_size, input_dim)
            quantized: If True, use 8-bit quantized inference for even lower power usage
            
        Returns:
            Model predictions as a numpy array
        """
        # Ensure inputs are 2D
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(1, -1)
        
        # Make sure dtype is float32 for standard inference
        if not quantized and inputs.dtype != np.float32:
            inputs = inputs.astype(np.float32)
        
        # Optional: Quantize for low-power inference
        if quantized:
            return self._quantized_inference(inputs)
        
        # Regular floating-point inference
        x = inputs
        for i, ((weights, bias), activation_name) in enumerate(zip(self.layers, self.activations)):
            # Linear transformation
            x = np.dot(x, weights) + bias
            
            # Apply activation function
            activation_fn = self.activation_functions.get(activation_name, self._linear)
            x = activation_fn(x)
        
        return x
    
    def _quantized_inference(self, inputs: np.ndarray) -> np.ndarray:
        """
        Perform inference using 8-bit quantization for ultra-low power usage.
        
        Args:
            inputs: Input data
            
        Returns:
            Model predictions
        """
        # Quantize inputs to int8
        x = inputs.astype(np.float32)
        x_scale = 127.0 / np.max(np.abs(x))
        x_q = np.round(x * x_scale).astype(np.int8)
        
        for i, ((weights, bias), activation_name) in enumerate(zip(self.layers, self.activations)):
            # Quantize weights to int8
            w_scale = 127.0 / np.max(np.abs(weights))
            w_q = np.round(weights * w_scale).astype(np.int8)
            
            # Dequantize for computation (in low-power hardware this would be done differently)
            x_float = x_q.astype(np.float32) / x_scale
            w_float = w_q.astype(np.float32) / w_scale
            
            # Linear transformation
            x = np.dot(x_float, w_float) + bias
            
            # Apply activation
            activation_fn = self.activation_functions.get(activation_name, self._linear)
            x = activation_fn(x)
            
            # Re-quantize for next layer
            if i < len(self.layers) - 1:  # Don't quantize output of final layer
                x_scale = 127.0 / np.max(np.abs(x))
                x_q = np.round(x * x_scale).astype(np.int8)
        
        return x

    def benchmark(self, input_shape: Tuple[int, int], iterations: int = 100) -> Dict:
        """
        Benchmark inference speed on random data.
        
        Args:
            input_shape: Tuple of (batch_size, input_dim)
            iterations: Number of iterations to run
            
        Returns:
            Dictionary with benchmark results
        """
        # Generate random test data
        test_inputs = np.random.random(input_shape).astype(np.float32)
        
        # Warm-up run
        _ = self.predict(test_inputs)
        
        # Benchmark standard inference
        start_time = time.time()
        for _ in range(iterations):
            _ = self.predict(test_inputs)
        std_time = time.time() - start_time
        
        # Benchmark quantized inference
        start_time = time.time()
        for _ in range(iterations):
            _ = self.predict(test_inputs, quantized=True)
        quant_time = time.time() - start_time
        
        return {
            'input_shape': input_shape,
            'iterations': iterations,
            'standard_inference_time': std_time,
            'standard_inference_fps': iterations / std_time,
            'quantized_inference_time': quant_time,
            'quantized_inference_fps': iterations / quant_time,
            'speedup_factor': std_time / quant_time if quant_time > 0 else 0
        }


class ModelConverter:
    """
    Utility class to convert models from popular frameworks to the lightweight format.
    This is a placeholder - in a real implementation, you would add support for
    converting from frameworks like TensorFlow Lite, ONNX, etc.
    """
    
    @staticmethod
    def from_simple_layers(layer_dims: List[int], activations: List[str]) -> LightweightInferenceEngine:
        """
        Create a simple model with random weights for testing.
        
        Args:
            layer_dims: List of layer dimensions [input_dim, hidden1, hidden2, ..., output_dim]
            activations: List of activation functions for each layer
            
        Returns:
            A LightweightInferenceEngine instance with the specified architecture
        """
        engine = LightweightInferenceEngine()
        
        # Initialize with random weights
        for i in range(len(layer_dims) - 1):
            # Xavier/Glorot initialization for better convergence
            limit = np.sqrt(6 / (layer_dims[i] + layer_dims[i+1]))
            weights = np.random.uniform(-limit, limit, (layer_dims[i], layer_dims[i+1])).astype(np.float32)
            bias = np.zeros(layer_dims[i+1], dtype=np.float32)
            
            engine.layers.append((weights, bias))
            engine.activations.append(activations[i])
        
        engine.model_info = {
            'name': 'Simple Random Model',
            'description': 'Randomly initialized model for testing',
            'layer_dims': layer_dims,
            'activations': activations
        }
        
        return engine


# Example usage
if __name__ == "__main__":
    # Create a simple test model with 2 inputs, 8 hidden units, and 1 output
    converter = ModelConverter()
    model = converter.from_simple_layers(
        layer_dims=[2, 8, 1],
        activations=['relu', 'sigmoid']
    )
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    model.save_model("models/test_model.json")
    
    # Load the model
    engine = LightweightInferenceEngine("models/test_model.json")
    
    # Generate some test data
    test_input = np.array([[0.5, 0.8]], dtype=np.float32)
    
    # Run inference
    result = engine.predict(test_input)
    print(f"Inference result: {result}")
    
    # Run quantized inference
    quantized_result = engine.predict(test_input, quantized=True)
    print(f"Quantized inference result: {quantized_result}")
    
    # Benchmark
    benchmark_results = engine.benchmark(input_shape=(64, 2), iterations=1000)
    print("Benchmark results:")
    for key, value in benchmark_results.items():
        print(f"  {key}: {value}")
