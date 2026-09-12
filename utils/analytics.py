"""
Analytics Functions
Contains statistical and visualization helper functions
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def generate_summary_stats(df):
    """
    Generate comprehensive summary statistics
    
    Parameters:
    df: pandas DataFrame
    
    Returns:
    pandas.DataFrame: Summary statistics
    """
    summary = pd.DataFrame()
    
    # Numerical columns statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        numeric_stats = df[numeric_cols].describe().T
        numeric_stats['variance'] = df[numeric_cols].var()
        numeric_stats['skewness'] = df[numeric_cols].skew()
        summary = pd.concat([summary, numeric_stats])
    
    # Categorical columns statistics
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        cat_stats = pd.Series({
            'count': df[col].count(),
            'unique': df[col].nunique(),
            'top': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
            'freq': df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else None
        }, name=col)
        summary = pd.concat([summary, cat_stats.to_frame().T])
    
    return summary

def create_correlation_plot(df):
    """
    Create an interactive correlation heatmap
    
    Parameters:
    df: pandas DataFrame with numeric columns
    
    Returns:
    plotly.graph_objects.Figure: Correlation heatmap
    """
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmin=-1, zmax=1,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Correlation Heatmap',
        width=800,
        height=600,
        xaxis_title='Features',
        yaxis_title='Features'
    )
    
    return fig

def calculate_growth_rate(series):
    """
    Calculate growth rate for a time series
    
    Parameters:
    series: pandas Series
    
    Returns:
    float: Growth rate percentage
    """
    if len(series) < 2:
        return 0
    
    growth = ((series.iloc[-1] - series.iloc[0]) / series.iloc[0]) * 100
    return growth