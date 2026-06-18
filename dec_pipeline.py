import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import copy

from models import StackedAutoencoder, DeepEmbeddedClustering

class CustomDataset(Dataset):
    """
    A custom PyTorch Dataset that returns the index of the sample along with the data.
    This is necessary for DEC to fetch the correct target distribution 'p' for each mini-batch.
    """
    def __init__(self, X):
        # Handle both pandas DataFrames and numpy arrays
        if isinstance(X, pd.DataFrame):
            self.X = torch.tensor(X.values, dtype=torch.float32)
        else:
            self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], idx

def pretrain_autoencoder(X, input_dim=3, z_dim=10, epochs=200, batch_size=256, lr=1e-3, patience=10, save_path="autoencoder_pretrained.pth"):
    """
    Phase 1: Pre-trains the Stacked Autoencoder using MSE loss.
    Implements 80/20 train/validation split and Early Stopping.
    """
    print("\n--- Phase 1: Autoencoder Pre-training ---")
    
    # Ensure data is numpy array
    if isinstance(X, pd.DataFrame):
        X_data = X.values
    else:
        X_data = X
        
    # 80-20 Train-Validation Split
    X_train, X_val = train_test_split(X_data, test_size=0.2, random_state=42)
    
    # Create DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize Model, Loss, Optimizer
    autoencoder = StackedAutoencoder(input_dim=input_dim, z_dim=z_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    
    # Early Stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_weights = copy.deepcopy(autoencoder.state_dict())
    
    for epoch in range(epochs):
        # Training
        autoencoder.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs = batch[0]
            optimizer.zero_grad()
            reconstructed, _ = autoencoder(inputs)
            loss = criterion(reconstructed, inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        autoencoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0]
                reconstructed, _ = autoencoder(inputs)
                loss = criterion(reconstructed, inputs)
                val_loss += loss.item() * inputs.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_weights = copy.deepcopy(autoencoder.state_dict())
        else:
            patience_counter += 1
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Best Val Loss: {best_val_loss:.6f}")
            break
            
    # Save and return best model
    torch.save(best_model_weights, save_path)
    print(f"Pre-trained weights saved to '{save_path}'.")
    autoencoder.load_state_dict(best_model_weights)
    return autoencoder

def target_distribution(q):
    """
    Computes the target distribution p_ij given the soft assignments q_ij.
    Formula: p_ij = (q_ij^2 / f_j) / sum_j' (q_ij'^2 / f_j')
    where f_j = sum_i q_ij (soft cluster frequencies)
    """
    weight = (q ** 2) / torch.sum(q, dim=0)
    p = (weight.t() / torch.sum(weight, dim=1)).t()
    return p

def train_dec(X, dec_model, batch_size=256, lr=1e-4, max_iters=2000, update_interval=140, pretrained_path="autoencoder_pretrained.pth"):
    """
    Phase 2: Joint Training of Deep Embedded Clustering.
    Optimizes the clustering layer and the encoder via KL-Divergence.
    """
    print("\n--- Phase 2: DEC Joint Training ---")
    
    # Load pre-trained autoencoder weights into the DEC model
    try:
        dec_model.autoencoder.load_state_dict(torch.load(pretrained_path))
        print("Successfully loaded pre-trained autoencoder weights.")
    except Exception as e:
        print(f"Warning: Could not load pre-trained weights from {pretrained_path}. Error: {e}")
        
    dataset = CustomDataset(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Extract latent features for KMeans initialization
    print("Extracting latent representations for KMeans initialization...")
    dec_model.eval()
    with torch.no_grad():
        _, z = dec_model.autoencoder(dataset.X)
        z_numpy = z.cpu().numpy()
        
    print(f"Running KMeans (k={dec_model.clustering_layer.n_clusters}) on latent space...")
    kmeans = KMeans(n_clusters=dec_model.clustering_layer.n_clusters, n_init=20, random_state=42)
    kmeans.fit(z_numpy)
    
    # Initialize clustering layer centroids with KMeans cluster centers
    dec_model.clustering_layer.centroids.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    print("Cluster centroids initialized.")
    
    # Setup Optimizer and Loss function
    # Only train the encoder and the clustering layer (freeze decoder for pure DEC, or train all. Standard DEC trains all).
    optimizer = optim.Adam(dec_model.parameters(), lr=lr)
    
    # PyTorch KLDivLoss expects log(predictions) and true probabilities
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    # Initialize target distribution P
    P = torch.zeros((len(dataset), dec_model.clustering_layer.n_clusters), dtype=torch.float32)
    
    # Training Loop
    dec_model.train()
    ite = 0
    
    print(f"Starting DEC optimization for {max_iters} iterations (Update interval: {update_interval})...")
    
    while ite < max_iters:
        for batch_x, batch_idx in dataloader:
            
            # Periodically calculate the Target Distribution P
            if ite % update_interval == 0:
                dec_model.eval()
                with torch.no_grad():
                    q_all, _, _ = dec_model(dataset.X)
                    P = target_distribution(q_all)
                dec_model.train()
                
            # Current mini-batch target P
            p_batch = P[batch_idx]
            
            # Forward pass
            optimizer.zero_grad()
            q_batch, _, _ = dec_model(batch_x)
            
            # KL Divergence Loss
            # q_batch + 1e-8 prevents log(0)
            loss = criterion(torch.log(q_batch + 1e-8), p_batch)
            
            # Backward and Optimize
            loss.backward()
            optimizer.step()
            
            if ite % 200 == 0 or ite == max_iters - 1:
                print(f"Iteration {ite:4d}/{max_iters} | KL Divergence Loss: {loss.item():.6f}")
                
            ite += 1
            if ite >= max_iters:
                break
                
    print("DEC Joint Training complete.")
    
    # Final Evaluation/Assignment
    dec_model.eval()
    with torch.no_grad():
        final_q, _, final_z = dec_model(dataset.X)
        final_assignments = torch.argmax(final_q, dim=1).numpy()
        latent_features = final_z.numpy()
        
    return dec_model, final_assignments, latent_features
