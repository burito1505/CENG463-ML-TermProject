import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score
from data_preprocessing import download_online_retail_data, load_and_clean_data, extract_rfm_features, preprocess_data
from baseline_models import run_all_baselines, run_kmeans
from visualization import plot_clusters

# Import DEC modules
from models import DeepEmbeddedClustering
from dec_pipeline import pretrain_autoencoder, train_dec
from ablation_error_analysis import run_ablation_study, perform_error_analysis

def main():
    print("======================================================")
    print("CENG 463 - Machine Learning Term Project")
    print("Phase 2: Deep Embedded Clustering (DEC) Execution")
    print("======================================================\n")

    # 1. Data Preparation pipeline
    print("Step 1: Data Preparation")
    print("-------------------------")
    try:
        file_path = download_online_retail_data()
        raw_df = load_and_clean_data(file_path)
        X_raw = extract_rfm_features(raw_df)
        X_scaled = preprocess_data(X_raw)
        print(f"Data preparation complete. Shape of features: {X_scaled.shape}\n")
    except Exception as e:
        print(f"An error occurred during data preparation: {e}")
        return

    # 2. Run Baseline Models and Evaluate
    print("Step 2: Training Baseline Models")
    print("---------------------------------")
    try:
        results_df = run_all_baselines(X_scaled)
        
        print("\n======================================================")
        print("BASELINE RESULTS SUMMARY")
        print("======================================================")
        # Setting pandas options to display exactly as it is for the report
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(results_df.to_string(index=False))
        print("======================================================")
        
        # Optionally save the DataFrame to a CSV
        results_df.to_csv("baseline_results.csv", index=False)
        print("\nResults successfully saved to 'baseline_results.csv'.")
    except Exception as e:
        print(f"An error occurred during model training/evaluation: {e}")
        return

    # 3. Visualization
    print("\nStep 3: Visualizing the Best Baseline (K-Means)")
    print("-----------------------------------------------")
    try:
        # Re-run K-Means just to get the labels directly for plotting
        kmeans_labels = run_kmeans(X_scaled, n_clusters=3)
        plot_clusters(X_scaled, kmeans_labels, model_name="K-Means_k3", method="PCA")
        plot_clusters(X_scaled, kmeans_labels, model_name="K-Means_k3", method="TSNE")
        print("Baseline Visualizations generated.")
    except Exception as e:
        print(f"An error occurred during baseline visualization: {e}")

    # ======================================================
    # NEW PHASE 2 DEC WORKFLOW
    # ======================================================
    print("\n======================================================")
    print("PHASE 2: DEEP EMBEDDED CLUSTERING (DEC) WORKFLOW")
    print("======================================================\n")

    optimal_z = 10
    optimal_k = 3

    print(f"Step 4: Training Optimal DEC Model (z={optimal_z}, k={optimal_k})")
    print("-----------------------------------------------------------------")
    try:
        # 4a. Pre-train Autoencoder
        pretrained_path = f"autoencoder_optimal_z{optimal_z}.pth"
        pretrain_autoencoder(X_scaled, input_dim=X_scaled.shape[1], z_dim=optimal_z, epochs=150, batch_size=256, lr=1e-3, patience=10, save_path=pretrained_path)

        # 4b. Joint Training
        dec_model = DeepEmbeddedClustering(n_clusters=optimal_k, input_dim=X_scaled.shape[1], z_dim=optimal_z)
        dec_model, final_assignments, latent_features = train_dec(
            X_scaled, dec_model, batch_size=256, lr=1e-4, max_iters=1000, 
            update_interval=140, pretrained_path=pretrained_path
        )

        # Calculate DEC metrics
        dec_sil = silhouette_score(latent_features, final_assignments)
        dec_db = davies_bouldin_score(latent_features, final_assignments)

        # Visualizing DEC
        print("\nStep 5: Visualizing the Final DEC Model")
        print("---------------------------------------")
        plot_clusters(pd.DataFrame(latent_features), final_assignments, model_name="DEC_k3", method="PCA")
        plot_clusters(pd.DataFrame(latent_features), final_assignments, model_name="DEC_k3", method="TSNE")

        # Step 6: Ablation Study
        print("\nStep 6: Running Ablation Study")
        print("------------------------------")
        run_ablation_study(X_scaled, n_clusters=optimal_k)

        # Step 7: Error Analysis
        print("\nStep 7: Performing Error Analysis")
        print("---------------------------------")
        perform_error_analysis(X_raw, X_scaled, final_assignments, dec_model)

        # Step 8: Final Summary Table
        print("\n======================================================")
        print("FINAL COMPARISON SUMMARY (Best Baseline vs DEC)")
        print("======================================================")
        
        # Get KMeans results from the baseline df
        kmeans_row = results_df[results_df['Model'] == 'K-Means (k=3)'].iloc[0]
        
        summary_df = pd.DataFrame([
            {'Model': 'K-Means (k=3)', 'Silhouette Score': kmeans_row['Silhouette Score'], 'Davies-Bouldin Index': kmeans_row['Davies-Bouldin Index']},
            {'Model': f'DEC (z={optimal_z}, k={optimal_k})', 'Silhouette Score': dec_sil, 'Davies-Bouldin Index': dec_db}
        ])
        
        print(summary_df.to_string(index=False))
        print("======================================================\n")

    except Exception as e:
        print(f"An error occurred during DEC pipeline execution: {e}")

if __name__ == "__main__":
    main()
