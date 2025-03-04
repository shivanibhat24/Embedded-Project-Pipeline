# Zero-Day Exploit Detection Using Anomaly Detection

![Cybersecurity Shield](https://img.shields.io/badge/Cybersecurity-Zero%20Day%20Detection-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12.0-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

<p align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNjA3MzAyMGlwNHZ0ZDNlcjF1dWw1eGhha2swdjEzMTBpejAyMWc1YiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3oKIPEqDGUULpEU0aQ/giphy.gif" alt="Cybersecurity Animation" width="600"/>
</p>

> *Developed by Team AI_Mavericks: Komal Sali, Parvati Pole, Shivani Bhat, Ibrahim*

## 🛡️ Project Overview

This project implements an advanced unsupervised deep learning pipeline for detecting zero-day exploits in network traffic. Using the UNSW-NB15 dataset, our solution combines **Variational Autoencoders (VAE)** with clustering techniques to identify anomalous network behavior that could indicate previously unknown vulnerabilities.

<p align="center">
  <img src="Visuals/tsne_visualization.png" alt="Visualization of Anomaly Detection" width="600"/>
</p>

### Why Zero-Day Detection Matters

Zero-day exploits represent one of the most critical threats in cybersecurity as they target previously unknown vulnerabilities that bypass traditional defense mechanisms. Our solution learns normal system behavior patterns and flags deviations that could indicate these emerging threats.

<p align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExbzFma2szbWxxb24xdGRhcHl1bGI3MnV2a2dmbHpvdG0yYjcwajZnciZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/077i6AULCXc0FKTj9s/giphy.gif" alt="Network Security Visualization" width="400"/>
</p>

## 🔍 Key Components

### 1. Unsupervised Learning Architecture
- **Variational Autoencoders (VAE)** learn compact, latent representations of normal system behavior
- **Reconstruction Error Analysis** identifies anomalies where behavior deviates from the learned norm

### 2. Advanced Anomaly Detection
- **K-means and DBSCAN clustering** on latent space representations group similar patterns and highlight outliers
- **Dynamic threshold setting** based on statistical analysis of reconstruction errors

<div align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExbWhoNm8ycG9iemY2Mm5xNTBidmY0cW1oMXZraXR2dmUxZW5icnV0ZCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/l46Cy1rHbQ92uuLXa/giphy.gif" alt="Clustering Animation" width="350"/>
</div>

### 3. Robust Data Processing
- **Preprocessing pipeline** cleans and normalizes high-dimensional network data
- **Dimensionality reduction** through the VAE's encoder handles noisy, complex network patterns

## 📊 Dataset

The solution leverages the **UNSW-NB15 dataset**, a comprehensive collection designed for network intrusion detection research. It contains a diverse mixture of benign and malicious network traffic that simulates real-world cyber attack scenarios.

<p align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExaXRtNmljbnNsaTVkYXNsNzdxYzVpMnZtbXZrN3cwcWQzbGt4OXJlYiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3oKIPEqDGUULpEU0aQ/giphy.gif" alt="Data Processing" width="450"/>
</p>

## 🚀 How It Works

### 1. Data Preprocessing
```
Raw Network Data → Feature Encoding → Normalization → Training Data
```

<p align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExb2lwb3dhMHd6M2VwNnVsbGJjaGJpNzVsYzR5djJ5YnNoMndiOXhldCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/26tn33aiTi1jkl6H6/giphy.gif" alt="Data Processing Animation" width="400"/>
</p>

- Categorical features undergo one-hot encoding
- Numerical features are standardized
- The model trains exclusively on normal (non-attack) data

### 2. VAE Model Architecture
```
Input → Encoder → Latent Space → Decoder → Reconstruction
```

<p align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNGZ5ZmJsZTIzMHF5cmVxdXRxYzR5M3FkcmZlcDBod3EyZ2R3bXRxdyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/xT9IgzoKnwFNmISR8I/giphy.gif" alt="VAE Architecture Animation" width="500"/>
</p>

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

## 🛠️ Installation and Setup

### Prerequisites

- Python 3.8+
- Virtual environment tool (recommended)

<p align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExN2drMGduN2tmMXN6Y3gxem55bWlkbmh0NGNyM2JnY242cWRwMXo5dCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/1afuwyOsr5E8X9CuRV/giphy.gif" alt="Installation Animation" width="400"/>
</p>

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

<p align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExMG10MGJ1cm1wam43amR4cjg2aXNyOTlydXlkb2M0OTQyOGwzZTRwYiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/l46CyJmS9KUbID7tK/giphy.gif" alt="Usage Animation" width="400"/>
</p>

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

<p align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExeTAxdWl5ejMyZHY2c2lzdHcwb2RvaGFpeXoxOTY5aGJqc2FqY2dodSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/pOEbLRT4SwD35IELiQ/giphy.gif" alt="Project Structure" width="400"/>
</p>

## 🔬 Technical Deep Dive

### Variational Autoencoder Architecture

<p align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdjJuM3dsbjQwb2U2M2lqMmx6bTg0OGY0M25rZWJ3Z3Q1OWwxeGNrZiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/LmNwrBhejkK9EFP504/giphy.gif" alt="VAE Explained" width="450"/>
</p>

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

<p align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExeHZ5OTJ2eWRnM3Q3aGNxZWJ5Z2ttNzgyZTFncWNsYWIxYTF2cmRnaSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o6ZtaO9BZHcOjmErm/giphy.gif" alt="Clustering Explained" width="400"/>
</p>

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

<p align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZTAycXhhMnVrZ2hxMno2dXQxbWdrcTF3aHMwZXI1a3JxMHhrNnk4NyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/KK6stQSoFWGfgdffGu/giphy.gif" alt="Future Vision" width="450"/>
</p>

- **Advanced Architectures**: Implement convolutional VAEs for better feature extraction
- **Real-time Detection**: Develop streaming data processing capabilities
- **Ensemble Methods**: Combine multiple anomaly detection techniques for greater accuracy
- **Explainability**: Add tools to interpret why specific traffic is flagged as anomalous
- **Adversarial Training**: Enhance robustness against evasion attempts

<p align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExODBpOW8zcjF3NWx3aXZkanBrNjJibTR6Nml6MHlndHgwM3N5MHV0MCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o6Zt8qDiPE2d3kayI/giphy.gif" alt="Security Animation" width="150"/>
  <br>
  <strong>Protecting networks from the unknown, one anomaly at a time.</strong>
</p>
