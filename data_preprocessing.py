import os
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.preprocessing import StandardScaler
import requests
from io import BytesIO

def download_online_retail_data(file_path: str = "Online Retail.xlsx"):
    """
    Downloads the Online Retail dataset from UCI if it doesn't exist locally.
    """
    if os.path.exists(file_path):
        print(f"{file_path} already exists. Skipping download.")
        return file_path
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    print(f"Downloading dataset from {url}...")
    response = requests.get(url)
    response.raise_for_status()
    
    with open(file_path, 'wb') as f:
        f.write(response.content)
    print("Download completed.")
    return file_path

def load_and_clean_data(file_path: str = "Online Retail.xlsx") -> pd.DataFrame:
    """
    Loads the dataset and performs initial cleaning (handling missing values, anomalies).
    """
    print("Loading dataset into pandas...")
    df = pd.read_excel(file_path)
    
    print("Cleaning data...")
    # Drop rows without CustomerID
    df.dropna(subset=['CustomerID'], inplace=True)
    
    # Keep only positive quantities and unit prices
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    
    # Calculate Total Price for each transaction
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    
    return df

def extract_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts Recency, Frequency, and Monetary (RFM) features per customer.
    """
    print("Extracting RFM features...")
    # Set snapshot date as 1 day after the latest invoice date for recency calculation
    snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    
    # Aggregate data at the CustomerID level
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days, # Recency
        'InvoiceNo': 'count',                                    # Frequency
        'TotalPrice': 'sum'                                      # Monetary
    }).reset_index()
    
    # Rename columns for clarity
    rfm.rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalPrice': 'Monetary'
    }, inplace=True)
    
    return rfm

def preprocess_data(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies log transformation and scales the RFM features.
    """
    print("Applying log transformation and scaling the features...")
    features = ['Recency', 'Frequency', 'Monetary']
    
    # RFM data is usually highly skewed. Log transformation helps distance-based 
    # algorithms like KMeans and DBSCAN form better boundaries.
    rfm_log = rfm_df.copy()
    for col in features:
        # We ensure all values are strictly positive before log transformation
        rfm_log[col] = np.log1p(rfm_log[col] - rfm_log[col].min() + 1)
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(rfm_log[features])
    
    rfm_scaled = pd.DataFrame(scaled_features, columns=features, index=rfm_log['CustomerID'])
    return rfm_scaled

def get_prepared_data() -> pd.DataFrame:
    """
    Orchestrates the data downloading, cleaning, and preprocessing pipeline.
    """
    file_path = download_online_retail_data()
    raw_df = load_and_clean_data(file_path)
    rfm_df = extract_rfm_features(raw_df)
    rfm_scaled_df = preprocess_data(rfm_df)
    
    return rfm_scaled_df
