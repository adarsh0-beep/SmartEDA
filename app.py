# streamlit run "/media/sai-satyam/New Volume/Programming/python/Streamlet_Course/project/app.py"



# ---------------------------------------------------------------------------
# 1. importing the required libraries
# ---------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Importing customed utilities
from utils.data_loader import load_data, get_data_info
from utils.analytics import generate_summary_stats, create_correlation_plot
from utils.insights import generate_insights, detect_anomalies
from utils.analytics import *
from utils.insights import *
from utils.data_loader import *

# doing the configration
st.set_page_config(
    page_title="SmartEDA",
    layout="wide",
    initial_sidebar_state="expanded"
)

def init_session_state():
    """Initialize all session state variables"""
    if 'data' not in st.session_state:
        st.session_state.data = None  
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False  

init_session_state()

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------


st.sidebar.title("Navigation")
st.sidebar.divider()

page = st.sidebar.radio(
    "Go to",
    ["Home", "Upload Data", "Analytics", "AI Insights", "About"]
)




# ---------------------------------------------------------------------------
# PAGE 1: HOME
# ---------------------------------------------------------------------------

if page == "Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.title("SmartEDA")
        st.markdown(
            """
            ### Transform Your Data into Actionable Insights
            
            **Upload, analyze, and understand your data** with our intelligent analytics platform.
            
            #### Features:
            - **Easy CSV Upload** - Drag and drop or browese files 
            - **Interactive Visualizations** - Bar charts, line charts, and heatmaps
            - **AI-Powered Insights** - Automatic pattern detection
            - **Real-time Analytics** - Filter and explore dynamically
            - **Statistical Summary** - Complete data overview
            
            ### Quick Start:
            1. Go to **Upload Data** section
            2. Upload your CSV file
            3. Explore **Analytics** and **AI Insights**
            """
        )
    
    with col2:
        st.markdown("### Quick Stats")
        # Display metrics if data is loaded
        if st.session_state.data_loaded and st.session_state.data is not None:
            df = st.session_state.data
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                st.metric("Rows", df.shape[0])
                st.metric("Columns", df.shape[1])
            with col2_2:
                st.metric("Missing Values", df.isnull().sum().sum())
                st.metric("Duplicate items", df.duplicated().sum())

        # if data is not loaded 
        else:
            st.info("No data loaded yet!\n\nUpload a CSV file to see metrics.")
            st.metric("Status", "Ready for upload", delta="Waiting")
            st.divider()
            st.markdown("### What You Can Analyze")
            st.markdown("This dashboard works great with any structured CSV data, for example:")
            st.markdown("**Sales & Business** — Monthly revenue, product performance, customer trends")
            st.markdown("If it's a CSV, this dashboard can handle it")
           
    
    # Key Benefits Section
    st.divider()
    st.subheader("Why Choose Smart Analytics Dashboard?")
    st.markdown("**Fast Processing**  -  Handle large datasets efficiently with optimized pandas operations" )
    st.markdown("**Beautiful Visuals**  - Interactive charts that help you spot patterns instantly")
    st.markdown("**Smart AI**  -  Automatic insight generation saves hours of manual analysis")


# ---------------------------------------------------------------------------
# PAGE 2: UPLOAD DATA
# ---------------------------------------------------------------------------


elif page == "Upload Data":

    st.title("Upload Your Dataset")
    st.markdown("Support CSV files (Comma Separated Values)")
    
    # File uploader widget
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv']
    )
    
    if uploaded_file is not None:
        # Load the data using our utility function
        data = load_data(uploaded_file)
        
        if data is not None:
            st.session_state.data = data
            st.session_state.data_loaded = True
            
            # Success message
            st.success(f"Successfully loaded {data.shape[0]} rows and {data.shape[1]} columns")
            
            # Display dataset information in expandable sections
            with st.expander("Dataset Preview", expanded=True):
                st.dataframe(data.head(11), use_container_width=True)
            
            # Two columns for dataset stats
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Dataset Shape")
                st.metric("Rows", data.shape[0])
                st.metric("Columns", data.shape[1])
                
                st.subheader("Data Types")
                st.write(data.dtypes.to_frame("Data Type"))
            
            with col2:
                st.subheader("Missing Values Analysis")
                missing_count = data.isnull().sum()
                missing_percent = (missing_count / len(data)) * 100
                
                missing_df = pd.DataFrame({
                    'Missing Count': missing_count,
                    'Missing Percent': missing_percent
                })
                missing_df = missing_df[missing_df['Missing Count'] > 0]
                
                if len(missing_df) > 0:
                    st.warning(f"Found {missing_df.shape[0]} columns with missing values")
                    st.dataframe(missing_df)
                else:
                    st.success("No missing values found in the dataset!")
            
            # Basic statistics
            with st.expander("Quick Statistics"):
                st.write("### Numerical Columns Summary")
                st.dataframe(data.describe())
                
                if data.select_dtypes(include=['object']).columns.tolist():
                    st.write("### Categorical Columns")
                    for col in data.select_dtypes(include=['object']).columns:
                        st.write(f"**{col}** - Unique values: {data[col].nunique()}")
                        st.write(data[col].value_counts().head())
        else:
            st.error("Error loading file. Please check the format.")
    else:
        st.info("Please upload a CSV file to begin") 

        # Show sample data example
        with st.expander("Sample CSV Format Example"):
            sample_data = pd.DataFrame({
                'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
                'Sales': [1000, 1500, 1200],
                'Customers': [50, 75, 60],
                'Region': ['North', 'South', 'East']
            })

            st.dataframe(sample_data)

# ---------------------------------------------------------------------------
# PAGE 3: ANALYTICS
# ---------------------------------------------------------------------------

elif page == "Analytics":
    st.title("SmartEDA")
    
    # Check if data is loaded
    if not st.session_state.data_loaded or st.session_state.data is None:
        st.warning("No data loaded! Please upload a dataset first.")
        st.stop()
    
    df = st.session_state.data
    
    # Create tabs for different analytics features
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Visualizations", "Summary Stats", "Correlation", "Filters", "Insights"]
    )
    
    # TAB 1: VISUALIZATIONS
    with tab1:
        st.subheader("Interactive Visualizations")
        
        # Get column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if len(numeric_cols) == 0:
            st.warning("No numeric columns available for visualization")
        else:
            # Chart type selection
            chart_type = st.selectbox(
                "Select Chart Type",
                ["Bar Chart", "Line Chart", "Scatter Plot", "Area Chart"]
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                x_axis = st.selectbox("X-axis", df.columns)
            
            with col2:
                if chart_type == "Scatter Plot":
                    y_axis = st.selectbox("Y-axis", numeric_cols)
                else:
                    y_axis = st.multiselect("Y-axis (Values)", numeric_cols, default=numeric_cols[:1] if numeric_cols else [])
            
            # Create plots based on selection
            if chart_type == "Bar Chart" and y_axis:
                fig = px.bar(df, x=x_axis, y=y_axis, title=f"Bar Chart: {x_axis} vs {', '.join(y_axis)}")
                st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "Line Chart" and y_axis:
                fig = px.line(df, x=x_axis, y=y_axis, title=f"Line Chart: {x_axis} vs {', '.join(y_axis)}")
                st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "Scatter Plot":
                color_by = st.selectbox("Color by", [None] + categorical_cols)
                fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by, 
                               title=f"Scatter Plot: {x_axis} vs {y_axis}")
                st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "Area Chart" and y_axis:
                fig = px.area(df, x=x_axis, y=y_axis, title=f"Area Chart: {x_axis} vs {', '.join(y_axis)}")
                st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: SUMMARY STATISTICS
    with tab2:
        st.subheader("Summary Statistics")
        
        stats_type = st.radio("Select Statistics Type", ["Full Summary", "Numerical Only", "Categorical Only"])
        
        if stats_type == "Full Summary":
            st.write(generate_summary_stats(df))
            st.dataframe(df.describe(include='all'))
        elif stats_type == "Numerical Only":
            st.dataframe(df.describe())
        else:
            cat_cols = df.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                st.dataframe(df[cat_cols].describe())
            else:
                st.info("No categorical columns found")
    
    # TAB 3: CORRELATION
    with tab3:
        st.subheader("Correlation Analysis")
        
        if len(numeric_cols) >= 2:
            # Create correlation heatmap
            fig = create_correlation_plot(df[numeric_cols])
            st.plotly_chart(fig, use_container_width=True)
            
            # Show correlation matrix
            with st.expander("View Correlation Matrix"):
                corr_matrix = df[numeric_cols].corr()
                st.dataframe(corr_matrix)
        else:
            st.warning("Need at least 2 numeric columns for correlation analysis")
    
    # TAB 4: FILTERS
    with tab4:
        st.subheader("Data Filtering")
        
        # Dynamic filters based on column types
        filtered_df = df.copy()
        
        # Numeric column filters
        # Numeric column filters
    for col in numeric_cols[:3]:

        numeric_data = df[col].dropna()

        if len(numeric_data) == 0:
            continue

        min_val = float(numeric_data.min())
        max_val = float(numeric_data.max())

    # Skip invalid ranges
        if min_val >= max_val:
            continue

        filter_range = st.slider(
            f"Filter {col}",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val)
    )

        filtered_df = filtered_df[
            (filtered_df[col] >= filter_range[0]) &
            (filtered_df[col] <= filter_range[1])
    ]
        
        # Categorical column filters
        # Categorical column filters

        # Categorical column filters

    for i, col in enumerate(categorical_cols[:2]):

        unique_vals = df[col].unique().tolist()

        selected_vals = st.multiselect(
        f"Select {col} values",
        unique_vals,
        default=unique_vals[:3] if len(unique_vals) > 3 else unique_vals,
        key=f"categorical_filter_{i}"
    )

        if selected_vals:
            filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]

    st.success(f"Filtered dataset: {len(filtered_df)} rows out of {len(df)}")

    st.dataframe(filtered_df)
    
    # TAB 5: QUICK INSIGHTS
    with tab5:
        st.subheader("Quick Analytics Insights")
        
        # Show key metrics
        if numeric_cols:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                highest_col = df[numeric_cols].max().idxmax()
                highest_val = df[numeric_cols].max().max()
                st.metric("Highest Value", f"{highest_val:,.2f}", highest_col)
            
            with col2:
                lowest_col = df[numeric_cols].min().idxmin()
                lowest_val = df[numeric_cols].min().min()
                st.metric("Lowest Value", f"{lowest_val:,.2f}", lowest_col)
            
            with col3:
                avg_val = df[numeric_cols].mean().mean()
                st.metric("Global Average", f"{avg_val:,.2f}")

# ============================================================================
# PAGE 4: AI INSIGHTS
# ============================================================================

elif page == "AI Insights":
    st.title("AI-Powered Insights")
    st.markdown("**Intelligent pattern detection and data analysis**")
    
    if not st.session_state.data_loaded or st.session_state.data is None:
        st.warning("Please upload a dataset first to generate insights!")
        st.stop()
    
    df = st.session_state.data
    
    # Generate AI insights
    with st.spinner("Analyzing your data..."):
        insights = generate_insights(df)
    
    # Display insights in cards
    st.subheader("Key Findings")
    
    for i, insight in enumerate(insights):
        with st.container():
            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                st.markdown(f"### {i+1}")
            with col2:
                st.info(insight)
            st.markdown("---")
    
    # Trend Analysis
    st.subheader("Trend Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        # Detect trends in numeric columns
        for col in numeric_cols[:3]:
            trend = "increasing" if df[col].iloc[-1] > df[col].iloc[0] else "decreasing"
            
            col1, col2 = st.columns([3, 1])
            with col1:
                fig = px.line(df, x=df.index, y=col, title=f"{col} - {trend} trend")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                change_pct = ((df[col].iloc[-1] - df[col].iloc[0]) / df[col].iloc[0]) * 100
                st.metric(
                    f"{col} Change",
                    f"{change_pct:+.1f}%",
                    delta=f"From {df[col].iloc[0]:.2f} to {df[col].iloc[-1]:.2f}"
                )
    else:
        st.info("No numeric columns found for trend analysis")
    
    # Anomaly Detection
    st.subheader("Anomaly Detection")
    
    anomalies = detect_anomalies(df)
    if anomalies:
        st.warning(f"Found potential anomalies in: {', '.join(anomalies)}")
        for col in anomalies[:2]:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = df[z_scores > 2]
            if len(outliers) > 0:
                st.write(f"**{col}** - Found {len(outliers)} potential outliers")
                st.dataframe(outliers[[col]])
    else:
        st.success("No significant anomalies detected in the dataset!")
    
    # Recommendations
    st.subheader("Recommendations")
    st.markdown("""
    Based on the analysis, here are some recommendations:
    
    1. **Data Quality**: Consider handling missing values before analysis
    2. **Feature Engineering**: Create new features from existing ones
    3. **Outlier Analysis**: Investigate detected anomalies
    4. **Correlation**: Look for strong correlations between variables
    5. **Trend Monitoring**: Continue tracking identified trends
    """)

# ---------------------------------------------------------------------------
# PAGE 5: ABOUT
# ---------------------------------------------------------------------------

elif page == "About":
    st.header("About This Project")

    st.markdown(
        """   
    Project OverviewAn interactive web application that transforms raw CSV data into meaningful insights, no coding required from the end user, This project was Built to showcase Python, data visualization, and Streamlit development skills,The Smart SmartEDA is a professional web application that transforms raw CSV data into actionable insights through interactive visualizations and automated analysis.The application features a clean sidebar navigation with five core modules: Home for project overview, Upload Data for CSV ingestion with automatic data validation, Analytics for bar/line/scatter charts and correlation heatmaps with real-time filtering, AI Insights for automated pattern detection and trend analysis, and comprehensive statistical summaries.Built with Streamlit for the frontend, Pandas for data manipulation, and Plotly for interactive charts, this tool handles missing values, detects anomalies using z-score analysis, 
    generates intelligent text insights, and maintains data persistence through session state management. The modular code architecture separates concerns into data loading, analytics, and insights utilities, 
    making it maintainable and scalable for future enhancements like Excel support, machine learning predictions, or database integration.
    which demonstrates proficiency in:
    
    - **Python Programming** - Clean, modular, well-documented code
    - **Streamlit** - Interactive web application development
    - **Data Analysis** - Pandas, NumPy for data manipulation
    - **Data Visualization** - Plotly, Matplotlib, Seaborn
    - **UI/UX Design** - Professional and user-friendly interface

    """) 
    
    st.divider()

    st.markdown("# Core features")

    col1, col2  = st.columns(2)
    
    with col1:
        st.metric("", "Upload CSV file", delta = "Instant preiview")

    with col2:
        st.metric("", "Interactive charts", delta = "Bar, line, and correlation heatmap charts")

    col3, col4 = st.columns(2)
    
    with col3:
        st.metric("", "AI insight engine", delta = "detects the patterns")
    
    with col4:
        st.metric("", "Data Analytics", delta = "Comparing the data")

    