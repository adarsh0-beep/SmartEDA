"""
Data Loading Utilities
Handles CSV file loading and data validation
"""

import pandas as pd
import streamlit as st

def load_data(file):
    """
    Load CSV file into pandas DataFrame
    
    Parameters:
    file: Uploaded file object from Streamlit
    
    Returns:
    pandas.DataFrame or None if error
    """
    try:
        # Try reading with default settings first
        df = pd.read_csv(file)
        
        # Basic validation
        if df.empty:
            st.error("The uploaded file is empty!")
            return None
        
        # Clean column names (remove spaces, make lowercase)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        return df
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def get_data_info(df):
    """
    Extract comprehensive information about the dataset
    
    Parameters:
    df: pandas DataFrame
    
    Returns:
    dict: Dictionary containing dataset information
    """
    if df is None:
        return {}
    
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percent': (df.isnull().sum() / len(df) * 100).to_dict(),
        'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # in MB
    }
    
    return info