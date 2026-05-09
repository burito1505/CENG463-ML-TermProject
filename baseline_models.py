import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score

def run_kmeans(X, n_clusters=3, random_state=42):
    """
    Runs K-Means clustering algorithm.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    labels = kmeans.fit_predict(X)
    return labels

def run_dbscan(X, eps=0.5, min_samples=5):
    """
    Runs DBSCAN clustering algorithm.
    Note: DBSCAN inherently finds the number of clusters. 
    It might label points as noise (-1).
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    return labels

def run_gmm(X, n_components=3, random_state=42):
    """
    Runs Gaussian Mixture Models clustering algorithm.
    """
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    labels = gmm.fit_predict(X)
    return labels

def evaluate_model(X, labels, model_name):
    """
    Evaluates a clustering model using Silhouette Score and Davies-Bouldin Index.
    Excludes noise (-1) from Silhouette calculation if using DBSCAN, 
    but for completeness we calculate it generally. If only 1 cluster is found, metrics are invalid.
    """
    # If the algorithm found less than 2 distinct clusters (e.g. DBSCAN finding all noise or 1 cluster)
    unique_labels = set(labels)
    if len(unique_labels) < 2 or (len(unique_labels) == 2 and -1 in unique_labels):
        return {
            'Model': model_name,
            'Silhouette Score': None,
            'Davies-Bouldin Index': None,
            'Num Clusters': len(unique_labels) - (1 if -1 in unique_labels else 0)
        }
    
    # Calculate metrics
    sil_score = silhouette_score(X, labels)
    db_index = davies_bouldin_score(X, labels)
    
    return {
        'Model': model_name,
        'Silhouette Score': sil_score,
        'Davies-Bouldin Index': db_index,
        'Num Clusters': len(unique_labels) - (1 if -1 in unique_labels else 0) # Discount noise cluster for count
    }

def run_all_baselines(X) -> pd.DataFrame:
    """
    Runs K-Means, DBSCAN, and GMM, evaluates them, and returns a DataFrame.
    """
    print("Running baseline models...")
    results = []
    
    # 1. K-Means
    print(" - Run K-Means")
    kmeans_labels = run_kmeans(X, n_clusters=3)
    results.append(evaluate_model(X, kmeans_labels, 'K-Means (k=3)'))
    
    # 2. DBSCAN
    # We applied both Log Transformation and StandardScaler.
    # Therefore, the data is compacted (mean=0, std=1). Since standard deviation is 1, 
    # an eps of 1.5 is massive and swallows the whole dataset into one cluster.
    # Let's reduce eps to a more reasonable scale for standardized data.
    print(" - Run DBSCAN")
    dbscan_labels = run_dbscan(X, eps=0.3, min_samples=10)
    results.append(evaluate_model(X, dbscan_labels, 'DBSCAN (eps=0.3, min=10)'))
    
    # 3. GMM
    print(" - Run GMM")
    gmm_labels = run_gmm(X, n_components=3)
    results.append(evaluate_model(X, gmm_labels, 'Gaussian Mixture Model (k=3)'))
    
    results_df = pd.DataFrame(results)
    return results_df
