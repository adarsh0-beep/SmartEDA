"""
AI Insights Generator
Automatically generates insights from data
"""

import pandas as pd
import numpy as np

def generate_insights(df):
    """
    Generate automated insights from the dataset
    
    Parameters:
    df: pandas DataFrame
    
    Returns:
    list: List of insight strings
    """
    insights = []
    
    # Dataset overview insights
    insights.append(f"Dataset contains {df.shape[0]:,} rows and {df.shape[1]} columns")
    
    # Missing values insights
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        insights.append(f"Found missing values in {len(missing_cols)} columns: {', '.join(missing_cols[:3])}")
    else:
        insights.append("Dataset is complete with no missing values")
    
    # Numerical column insights
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        # Find highest and lowest values
        max_col = df[numeric_cols].max().idxmax()
        max_val = df[numeric_cols].max().max()
        insights.append(f"Highest value found in '{max_col}': {max_val:,.2f}")
        
        min_col = df[numeric_cols].min().idxmin()
        min_val = df[numeric_cols].min().min()
        insights.append(f"Lowest value found in '{min_col}': {min_val:,.2f}")
        
        # Correlation insights
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            # Find highest correlation (excluding self-correlation)
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        insights.append(f"Strong correlation ({corr_val:.2f}) between '{corr_matrix.columns[i]}' and '{corr_matrix.columns[j]}'")
    
    # Categorical column insights
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols[:3]:  # Limit to first 3 categorical columns
        unique_count = df[col].nunique()
        if unique_count <= 10:
            most_frequent = df[col].mode().iloc[0]
            freq_count = df[col].value_counts().iloc[0]
            insights.append(f"In '{col}', '{most_frequent}' appears most frequently ({freq_count} times)")
        else:
            insights.append(f"Column '{col}' has {unique_count} unique categories")
    
    # Trend insights for time series (if date-like column exists)
    date_columns = df.select_dtypes(include=['datetime64']).columns
    if len(date_columns) > 0 and len(numeric_cols) > 0:
        insights.append("Date column detected - consider time series analysis")
        
        # Check for trends
        for col in numeric_cols[:2]:
            if len(df) > 1:
                trend = "increasing" if df[col].iloc[-1] > df[col].iloc[0] else "decreasing"
                change_pct = ((df[col].iloc[-1] - df[col].iloc[0]) / df[col].iloc[0]) * 100
                insights.append(f" {col} shows {trend} trend ({change_pct:+.1f}% change)")
    
    return insights

def detect_anomalies(df, threshold=2):
    """
    Detect anomalies using Z-score method
    
    Parameters:
    df: pandas DataFrame
    threshold: Z-score threshold for anomaly detection
    
    Returns:
    list: Columns with anomalies detected
    """
    anomalies = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        # Calculate Z-scores
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        # Check for outliers
        if (z_scores > threshold).any():
            anomalies.append(col)
    
    return anomalies

def suggest_visualizations(df):
    """
    Suggest appropriate visualizations based on data types
    
    Parameters:
    df: pandas DataFrame
    
    Returns:
    list: Suggested visualization types
    """
    suggestions = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    
    if len(numeric_cols) >= 2:
        suggestions.append("Correlation Heatmap - Shows relationships between variables")
    
    if len(numeric_cols) >= 1 and len(cat_cols) >= 1:
        suggestions.append("Bar Charts - Compare categories across metrics")
    
    if len(numeric_cols) >= 1:
        suggestions.append("Line Charts - Visualize trends over time (if date column exists)")
        suggestions.append("Histograms - Understand data distribution")
    
    if len(numeric_cols) >= 2:
        suggestions.append("Scatter Plots - Explore relationships between two variables")
    
    return suggestions