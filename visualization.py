import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plot_clusters(X_scaled: pd.DataFrame, labels: list, model_name: str, method: str = 'PCA'):
    """
    Reduces dimensionality of the features and plots the clusters.
    """
    print(f"Generating {method} plot for {model_name}...")
    
    if method.upper() == 'PCA':
        reducer = PCA(n_components=2, random_state=42)
    elif method.upper() == 'TSNE':
        reducer = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    else:
        raise ValueError("Method must be 'PCA' or 'TSNE'")
        
    reduced_features = reducer.fit_transform(X_scaled)
    
    df_plot = pd.DataFrame({
        'Dim 1': reduced_features[:, 0],
        'Dim 2': reduced_features[:, 1],
        'Cluster': labels
    })
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='Dim 1', y='Dim 2',
        hue='Cluster',
        palette=sns.color_palette("hls", as_cmap=True) if len(set(labels)) > 1 else "deep",
        data=df_plot,
        legend="full",
        alpha=0.6,
        edgecolor=None
    )
    plt.title(f'{model_name} Clusters visualized with {method.upper()}', fontsize=16)
    plt.xlabel(f'{method.upper()} Dimension 1', fontsize=12)
    plt.ylabel(f'{method.upper()} Dimension 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    file_name = f"{model_name.replace(' ', '_').replace('=', '').replace('(', '').replace(')', '')}_{method.upper()}.png"
    plt.savefig(file_name, bbox_inches='tight', dpi=300)
    print(f"Plot saved successfully as '{file_name}'.")
    plt.close()
