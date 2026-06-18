# CENG 463 - Deep Embedded Clustering for Customer Segmentation

This repository contains the source code for the CENG 463 (Machine Learning) Term Project. The project focuses on applying **Deep Embedded Clustering (DEC)** to perform advanced customer segmentation on the UCI Online Retail dataset.

## Project Overview

Traditional clustering algorithms (such as K-Means, DBSCAN) often struggle with high-dimensional, skewed, and non-linear behavioral data. This project implements a **Deep Embedded Clustering (DEC)** framework, utilizing a stacked autoencoder to map Recency, Frequency, and Monetary (RFM) transaction metrics into a lower-dimensional latent bottleneck space. 

By jointly optimizing the latent representation and cluster centroids via Kullback-Leibler (KL) divergence minimization, the DEC model unwraps topologies to discover highly separable customer segments. 

**Key Results:**
- The optimal DEC architecture ($z=5$, $k=3$) achieved a **Silhouette Score of 0.9587** and a **Davies-Bouldin Index of 0.0479**.
- Successfully isolated ultra-high-value behavioral profiles ("Whales").
- Vastly outperformed standard baselines (K-Means silhouette: 0.3030).

## Features
- **Automated Pipeline:** Automatically downloads the "Online Retail" dataset from the UCI repository if not found locally.
- **RFM Feature Engineering:** Extracts metrics and applies strict logarithmic transformation for variance stabilization, followed by scaling.
- **Deep Embedded Clustering:** Custom PyTorch implementation of the DEC algorithm.
- **Baseline Models:** Evaluates K-Means, DBSCAN, and Gaussian Mixture Models (GMM) as benchmarks.
- **Ablation & Error Profiling:** Sweeps across latent dimensions ($z$) to determine mathematical sweet spots and profiles output boundary uncertanties.
- **Visualization:** Generates high-quality PCA and t-SNE 2D scatter plots modeling the deep latent geometries.

## Project Structure
- `main.py`: The orchestrator script. Runs the entire pipeline from data ingestion to model evaluation and visualization.
- `data_preprocessing.py`: Handles data downloading, cleaning, RFM generation, log-transformation, and standard scaling.
- `models.py`: PyTorch neural network definitions for the stacked Autoencoder.
- `dec_pipeline.py`: Houses the core mathematical logic for Phase 1 (pre-training) and Phase 2 (KL optimization).
- `baseline_models.py`: Contains implementations and evaluation logic for K-Means, DBSCAN, and GMM.
- `ablation_error_analysis.py`: Logic managing the ablation studies and boundary instance profiling.
- `visualization.py`: Reduces dimensionality and saves PCA and t-SNE scatter plots as `.png` files.
- `*.pth`: Serialized model weights for optimal pre-trained Autoencoder states (e.g., `autoencoder_optimal_z10.pth`).
- `*.tex`: Source code for LaTeX project reports.
- `requirements.txt`: Lists all required Python dependencies.

## Installation & Usage

This project is fully reproducible. You do not need to download the dataset manually.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/burito1505/CENG463-ML-TermProject.git
   cd CENG463-ML-TermProject
   ```

2. **Create a virtual environment (Optional but Recommended):**
   ```bash
   python -m venv .venv
   # Activate on Windows:
   .venv\Scripts\activate
   # Activate on Mac/Linux:
   source .venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the pipeline:**
   ```bash
   python main.py
   ```

Upon execution, the script will:
1. Download the dataset (if missing) and preprocess the data.
2. Train, evaluate, and save statistics for the baseline models.
3. Pre-train the Autoencoder, or load available `.pth` weights.
4. Execute the DEC joint clustering optimization phase.
5. Generate performance statistics, error profiles, and save `DEC_k3_PCA.png` and `DEC_k3_TSNE.png` visualizations.

