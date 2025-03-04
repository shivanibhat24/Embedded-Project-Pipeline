# Zero-Day Exploit Detection Using Anomaly Detection

![Cybersecurity Shield](https://img.shields.io/badge/Cybersecurity-Zero%20Day%20Detection-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12.0-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> *Developed by Team AI_Mavericks: Komal Sali, Parvati Pole, Shivani Bhat, Ibrahim*

## 🛡️ Project Overview

This project implements an advanced unsupervised deep learning pipeline for detecting zero-day exploits in network traffic. Using the UNSW-NB15 dataset, our solution combines **Variational Autoencoders (VAE)** with clustering techniques to identify anomalous network behavior that could indicate previously unknown vulnerabilities.

<p align="center">
  <img src="Visuals/tsne_visualization.png" alt="Visualization of Anomaly Detection" width="600"/>
</p>

### Why Zero-Day Detection Matters

Zero-day exploits represent one of the most critical threats in cybersecurity as they target previously unknown vulnerabilities that bypass traditional defense mechanisms. Our solution learns normal system behavior patterns and flags deviations that could indicate these emerging threats.

## 🔍 Key Components

### 1. Unsupervised Learning Architecture
- **Variational Autoencoders (VAE)** learn compact, latent representations of normal system behavior
- **Reconstruction Error Analysis** identifies anomalies where behavior deviates from the learned norm

### 2. Advanced Anomaly Detection
- **K-means and DBSCAN clustering** on latent space representations group similar patterns and highlight outliers
- **Dynamic threshold setting** based on statistical analysis of reconstruction errors

### 3. Robust Data Processing
- **Preprocessing pipeline** cleans and normalizes high-dimensional network data
- **Dimensionality reduction** through the VAE's encoder handles noisy, complex network patterns

## 📊 Dataset

The solution leverages the **UNSW-NB15 dataset**, a comprehensive collection designed for network intrusion detection research. It contains a diverse mixture of benign and malicious network traffic that simulates real-world cyber attack scenarios.

## 🚀 How It Works

### 1. Data Preprocessing
```
Raw Network Data → Feature Encoding → Normalization → Training Data
```

- Categorical features undergo one-hot encoding
- Numerical features are standardized
- The model trains exclusively on normal (non-attack) data

### 2. VAE Model Architecture
```
Input → Encoder → Latent Space → Decoder → Reconstruction
```

- Learns to compress and reconstruct normal network traffic patterns
- Creates a meaningful latent space representation of the data

### 3. Anomaly Detection Process
```
New Data → VAE → Calculate Reconstruction Error → Compare to Threshold → Flag Anomalies
```

- Sets a statistical threshold based on the distribution of reconstruction errors
- Flags samples with reconstruction errors above the threshold as potential zero-day exploits

### 4. Latent Space Analysis
```
Latent Representations → Clustering Algorithms → Anomaly Groups and Insights
```

- Applies K-means and DBSCAN clustering to reveal data structure
- Provides deeper insights into potential anomaly groups

## 📈 Performance Visualization

Our model generates several insightful visualizations:

<div style="display: flex; justify-content: space-between;">
  <div style="flex: 1; padding: 5px;">
    <p align="center"><strong>Reconstruction Error Distribution</strong></p>
    <p align="center"><img src="Visuals/reconstruction_error.png" alt="Reconstruction Error" width="300"/></p>
  </div>
  <div style="flex: 1; padding: 5px;">
    <p align="center"><strong>Anomaly Scores</strong></p>
    <p align="center"><img src="Visuals/anomaly_scores.png" alt="Anomaly Scores" width="300"/></p>
  </div>
  <div style="flex: 1; padding: 5px;">
    <p align="center"><strong>ROC Curve</strong></p>
    <p align="center"><img src="Visuals/roc_curve.png" alt="ROC Curve" width="300"/></p>
  </div>
</div>

## 🛠️ Installation and Setup

### Prerequisites

- Python 3.8+
- Virtual environment tool (recommended)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/AI-Mavericks/zero-day-detection.git
   cd zero-day-detection
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   
   # On Windows
   .venv\scripts\activate
   
   # On Unix/MacOS
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the main detection script**
   ```bash
   python zero_day_detection.py
   ```

## 🔄 Usage Guide

### Training a New Model

Train the model on your network data:

```bash
python zero_day_detection.py --train --data ./datasets/UNSW_NB15_training-set.csv
```

### Running Inference

Detect anomalies in new network traffic:

```bash
python inference.py --input ./testing/UNSW_NB15_testing-set.csv
```

### Optimizing the Model

Find optimal hyperparameters:

```bash
python hyperparameter_tuning.py
```

## 📁 Project Structure

```
zero-day-detection/
├── zero_day_detection.py     # Main implementation
├── inference.py              # Inference on new data
├── hyperparameter_tuning.py  # Hyperparameter optimization
├── analyze_dataset.py        # Dataset exploration tools
├── requirements.txt          # Package dependencies
├── datasets/                 # Training and testing data
├── models/                   # Saved model components
│   ├── encoder_model.keras
│   ├── decoder_model.keras
│   └── threshold.npy
└── Visuals/                  # Visualization outputs
```

## 🔬 Technical Deep Dive

### Variational Autoencoder Architecture

Our VAE implementation consists of:

```python
# Encoder Network
x = Input(shape=(input_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

# Sampling Layer
z = Lambda(sampling)([z_mean, z_log_var])

# Decoder Network
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_output = Dense(input_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded = decoder_output(h_decoded)
```

The VAE is trained to minimize:
1. **Reconstruction Loss**: How well the model reconstructs the input
2. **KL Divergence**: How close the learned latent distribution is to a standard normal distribution

### Clustering Implementation

We employ two complementary clustering algorithms:

```python
# K-means for centroid-based clustering
kmeans = KMeans(n_clusters=n_clusters)
kmeans_labels = kmeans.fit_predict(latent_representations)

# DBSCAN for density-based clustering and outlier detection
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan_labels = dbscan.fit_predict(latent_representations)
```

## 🔮 Future Improvements

- **Advanced Architectures**: Implement convolutional VAEs for better feature extraction
- **Real-time Detection**: Develop streaming data processing capabilities
- **Ensemble Methods**: Combine multiple anomaly detection techniques for greater accuracy
- **Explainability**: Add tools to interpret why specific traffic is flagged as anomalous
- **Adversarial Training**: Enhance robustness against evasion attempts

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- The UNSW-NB15 dataset creators for providing valuable cybersecurity research data
- The open-source community for TensorFlow, scikit-learn, and other tools

## 📞 Contact

Team AI_Mavericks - [ai.mavericks@example.com](mailto:ai.mavericks@example.com)

---

<p align="center">
  <strong>Protecting networks from the unknown, one anomaly at a time.</strong>
</p>
