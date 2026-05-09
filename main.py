import pandas as pd
from data_preprocessing import get_prepared_data
from baseline_models import run_all_baselines

def main():
    print("======================================================")
    print("CENG 463 - Deep Embedded Clustering (DEC) Midphase")
    print("Technical Checkpoint: Data Prep and Baseline Models")
    print("======================================================\n")

    # 1. Data Preparation pipeline
    print("Step 1: Data Preparation")
    print("-------------------------")
    try:
        X_scaled = get_prepared_data()
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

if __name__ == "__main__":
    main()
