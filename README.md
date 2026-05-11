# CENG 463 - Deep Embedded Clustering for Customer Segmentation

This repository contains the source code for the CENG 463 (Machine Learning) Term Project. The project focuses on applying **Deep Embedded Clustering (DEC)** to perform advanced customer segmentation on the UCI Online Retail dataset.

Currently, the project is at the **Midphase (Technical Checkpoint)**, establishing a robust data preparation pipeline and evaluating traditional machine learning algorithms as baselines.

## Features
- **Automated Data Retrieval:** The pipeline automatically downloads the "Online Retail" dataset from the UCI repository if it is not found locally.
- **RFM Feature Engineering:** Extracts Recency, Frequency, and Monetary metrics from raw transactional logs.
- **Advanced Preprocessing:** Applies logarithmic transformation to handle extreme skewness in financial data, followed by standard scaling.
- **Baseline Models:** Evaluates K-Means, DBSCAN, and Gaussian Mixture Models (GMM) using internal clustering metrics (Silhouette Score & Davies-Bouldin Index).
- **Visualization:** Generates high-quality PCA and t-SNE 2D scatter plots to analyze cluster separability visually.

## Project Structure
- `main.py`: The orchestrator script. Runs the entire pipeline from data ingestion to model evaluation and visualization.
- `data_preprocessing.py`: Handles data downloading, cleaning, RFM creation, log-transformation, and scaling.
- `baseline_models.py`: Contains the implementations and evaluation logic for K-Means, DBSCAN, and GMM.
- `visualization.py`: Reduces dimensionality and saves PCA and t-SNE scatter plots as `.png` files.
- `requirements.txt`: Lists all Python dependencies required for reproducibility.
- `progress_report.tex`: The LaTeX source code for the midphase progress report.

## Installation & Usage

This project is fully reproducible. You do not need to download the dataset manually.

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd 463TermProject
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
1. Download the dataset (if missing) and preprocess it.
2. Train and evaluate the baseline models.
3. Print a formatted results table to the console and save it as `baseline_results.csv`.
4. Generate and save `K-Means_k3_PCA.png` and `K-Means_k3_TSNE.png` visualizations in the root directory.

## Future Work (Final Phase)
The current traditional baselines highlight the difficulty of clustering highly non-linear behavioral data (Silhouette scores ~0.30). In the final phase, an **Autoencoder-based Deep Embedded Clustering (DEC)** architecture will be implemented. A rigorous ablation study will be conducted to measure the impact of latent space dimensions and layer limits on cluster purity.
