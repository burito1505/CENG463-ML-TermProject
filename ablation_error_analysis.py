import torch
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score

from models import DeepEmbeddedClustering
from dec_pipeline import pretrain_autoencoder, train_dec

def run_ablation_study(X, n_clusters=3):
    """
    Runs an ablation study across different bottleneck dimensions (z).
    For each z, it pre-trains the Autoencoder, performs DEC joint training,
    calculates metrics, and saves the results.
    """
    z_dims = [2, 5, 10, 16]
    results = []

    print("\n======================================================")
    print("STARTING ABLATION STUDY FOR DEC")
    print("======================================================")

    for z in z_dims:
        print(f"\n--- Evaluating Bottleneck Dimension z = {z} ---")
        
        # Pretrain Autoencoder
        pretrained_path = f"autoencoder_pretrained_z{z}.pth"
        
        # We use a slightly reduced epoch/max_iter count here for the ablation 
        # study to keep the runtime manageable, while still showing convergence.
        pretrain_autoencoder(
            X, input_dim=X.shape[1], z_dim=z, 
            epochs=150, batch_size=256, lr=1e-3, patience=10, 
            save_path=pretrained_path
        )
        
        # Initialize DEC Model
        dec_model = DeepEmbeddedClustering(n_clusters=n_clusters, input_dim=X.shape[1], z_dim=z)
        
        # Joint Training
        trained_dec, assignments, latent_features = train_dec(
            X, dec_model, batch_size=256, lr=1e-4, max_iters=1000, 
            update_interval=140, pretrained_path=pretrained_path
        )
        
        # Metrics Calculation
        # We calculate metrics on the latent features as DEC optimizes the latent space.
        unique_labels = set(assignments)
        if len(unique_labels) > 1:
            sil_score = silhouette_score(latent_features, assignments)
            db_score = davies_bouldin_score(latent_features, assignments)
        else:
            sil_score = None
            db_score = None
            
        print(f"Results for z={z} -> Silhouette Score: {sil_score:.4f}, DB Index: {db_score:.4f}")
        
        results.append({
            'Latent Dimension (z)': z,
            'Silhouette Score': sil_score,
            'Davies-Bouldin Index': db_score
        })

    # Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv("dec_ablation_results.csv", index=False)
    print("\n======================================================")
    print(f"Ablation study complete. Results saved to 'dec_ablation_results.csv'.")
    print("======================================================\n")
    
    return results_df


def perform_error_analysis(X_raw, X_scaled, final_assignments, dec_model):
    """
    Profiles difficult or extreme segments in the dataset based on DEC's output:
    1. 'Whales': High-variance outliers in Monetary or Frequency.
    2. 'Boundary Instances': Customers where DEC is highly uncertain (low max soft assignment).
    """
    print("\n======================================================")
    print("ERROR ANALYSIS & SEGMENT PROFILING")
    print("======================================================")
    
    # Merge assignments back into the raw dataset for interpretable analysis
    analysis_df = X_raw.copy()
    analysis_df['Cluster'] = final_assignments
    
    # ---------------------------------------------------------
    # 1. Isolate Whale Customers
    # ---------------------------------------------------------
    # Whales are defined as customers in the top 2% of Monetary OR Frequency
    monetary_98 = analysis_df['Monetary'].quantile(0.98)
    frequency_98 = analysis_df['Frequency'].quantile(0.98)
    
    whales = analysis_df[(analysis_df['Monetary'] > monetary_98) | (analysis_df['Frequency'] > frequency_98)]
    
    print(f"\n--- 1. Whale Customers Analysis ---")
    print(f"Criteria: Monetary > {monetary_98:.2f} OR Frequency > {frequency_98:.2f}")
    print(f"Total Whales Found: {len(whales)}")
    
    if len(whales) > 0:
        print("\nCluster Distribution among Whales:")
        print(whales['Cluster'].value_counts().to_string())
        
        print("\nMean Profile of Whales per Cluster:")
        print(whales.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(2).to_string())
    
    # ---------------------------------------------------------
    # 2. Isolate Boundary Instances
    # ---------------------------------------------------------
    # We pass the scaled data back through the DEC model to get the soft assignments (q)
    dec_model.eval()
    if isinstance(X_scaled, pd.DataFrame):
        X_tensor = torch.tensor(X_scaled.values, dtype=torch.float32)
    else:
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
    with torch.no_grad():
        q, _, _ = dec_model(X_tensor)
        
    # Maximum probability represents the confidence of the model for that instance
    max_probs, _ = torch.max(q, dim=1)
    analysis_df['Assignment_Confidence'] = max_probs.numpy()
    
    # A max probability < 0.45 in a 3-cluster scenario means the model is almost guessing
    boundary_threshold = 0.45
    boundary_instances = analysis_df[analysis_df['Assignment_Confidence'] < boundary_threshold]
    
    print(f"\n--- 2. Boundary Instances Analysis ---")
    print(f"Criteria: DEC Assignment Confidence < {boundary_threshold}")
    print(f"Total Boundary Instances Found: {len(boundary_instances)}")
    
    if len(boundary_instances) > 0:
        print("\nCluster Distribution of Boundary Instances:")
        print(boundary_instances['Cluster'].value_counts().to_string())
        
        print("\nMean Profile of Boundary Instances per Cluster:")
        print(boundary_instances.groupby('Cluster')[['Recency', 'Frequency', 'Monetary', 'Assignment_Confidence']].mean().round(2).to_string())
        
        print("\nSample Boundary Instances (5 most uncertain):")
        print(boundary_instances.sort_values('Assignment_Confidence').head(5)[
            ['Recency', 'Frequency', 'Monetary', 'Cluster', 'Assignment_Confidence']
        ].to_string())
    else:
        # If the model is extremely confident, relax the threshold slightly to show the hardest examples
        relaxed_threshold = 0.55
        boundary_instances = analysis_df[analysis_df['Assignment_Confidence'] < relaxed_threshold]
        print(f"No instances found < {boundary_threshold}. Relaxing threshold to < {relaxed_threshold}.")
        print(f"Total Relaxed Boundary Instances: {len(boundary_instances)}")
        if len(boundary_instances) > 0:
             print("\nSample Borderline Instances (5 most uncertain):")
             print(boundary_instances.sort_values('Assignment_Confidence').head(5)[
                ['Recency', 'Frequency', 'Monetary', 'Cluster', 'Assignment_Confidence']
             ].to_string())
