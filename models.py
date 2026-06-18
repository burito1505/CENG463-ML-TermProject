import torch
import torch.nn as nn
from torch.nn import Parameter

class StackedAutoencoder(nn.Module):
    """
    Stacked Autoencoder network for initial dimensionality reduction.
    Encoder architecture: input_dim -> 500 -> 500 -> 2000 -> z_dim
    Decoder architecture: z_dim -> 2000 -> 500 -> 500 -> input_dim
    """
    def __init__(self, input_dim=3, z_dim=10):
        super(StackedAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, z_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        """
        Forward pass. Returns reconstructed input and the latent space embedding.
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

class ClusteringLayer(nn.Module):
    """
    Clustering layer that converts input samples (latent features) to soft labels.
    The probability (soft assignment q_ij) is calculated using Student's t-distribution
    between the embedding vectors and cluster centroids.
    """
    def __init__(self, n_clusters=3, z_dim=10, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        
        # Trainable cluster centroids parameter
        self.centroids = Parameter(torch.Tensor(n_clusters, z_dim))
        
        # Initialize centroids to prevent NaN issues if used before K-Means initialization
        nn.init.xavier_normal_(self.centroids.data)

    def forward(self, x):
        """
        Compute soft assignments (q_ij) between latent points x and cluster centroids.
        x shape: (batch_size, z_dim)
        Returns: Soft assignments of shape (batch_size, n_clusters)
        """
        # Distance squared: ||x - c||^2
        # x shape: (N, d), centroids shape: (K, d)
        # We expand dimensions to get (N, K, d) to compute the distance easily.
        dist = torch.sum((x.unsqueeze(1) - self.centroids) ** 2, dim=2)
        
        # Student's t-distribution soft assignment
        q = 1.0 / (1.0 + dist / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        
        # Normalize over clusters to get probability distribution
        q = (q.t() / torch.sum(q, dim=1)).t()
        
        return q

class DeepEmbeddedClustering(nn.Module):
    """
    Deep Embedded Clustering (DEC) Model.
    Unifies the StackedAutoencoder and the ClusteringLayer for end-to-end training.
    """
    def __init__(self, n_clusters=3, input_dim=3, z_dim=10, alpha=1.0):
        super(DeepEmbeddedClustering, self).__init__()
        self.autoencoder = StackedAutoencoder(input_dim=input_dim, z_dim=z_dim)
        self.clustering_layer = ClusteringLayer(n_clusters=n_clusters, z_dim=z_dim, alpha=alpha)

    def forward(self, x):
        """
        Forward pass for the DEC model.
        Returns:
            q: Soft cluster assignments
            x_recon: Reconstructed input (useful if joint loss includes MSE)
            z: Latent space representation
        """
        x_recon, z = self.autoencoder(x)
        q = self.clustering_layer(z)
        return q, x_recon, z
