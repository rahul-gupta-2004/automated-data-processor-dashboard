import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Set standard template for consistency
TEMPLATE = "plotly_white"

# Generate charts specifically for Classification tasks
def get_classification_charts(df, target):
    charts = [] # List to store generated figures
    
    # 1. Class Balance (Bar Chart)
    counts = df[target].value_counts().reset_index()
    counts.columns = [target, 'count']
    fig1 = px.bar(
        counts, x=target, y='count', 
        title="Class Balance (Bar Chart)", 
        color=target,
        template=TEMPLATE
    )
    charts.append(fig1)
    
    # 2. Feature Correlation (Heatmap)
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    if target in num_cols:
        num_cols.remove(target) # Remove target for correlation matrix
        
    if len(num_cols) > 1:
        corr = df[num_cols].corr() # Calculate correlation matrix
        fig2 = px.imshow(
            corr, 
            text_auto=False,
            title="Correlation Heatmap",
            color_continuous_scale='RdBu_r', 
            template=TEMPLATE,
            height=600
        )
        charts.append(fig2)

    # 3. Class Proportion (Pie Chart)
    fig3 = px.pie(
        counts, values='count', names=target,
        title="Class Distribution (Percentage)",
        template=TEMPLATE,
        hole=0.3 # Create a donut style chart
    )
    charts.append(fig3)

    # 4. Feature Distribution (Box Plot)
    if len(num_cols) > 0:
        # Find feature with most unique values for plotting
        main_feat = sorted(num_cols, key=lambda c: df[c].nunique(), reverse=True)[0]
        fig4 = px.box(
            df, x=target, y=main_feat, color=target,
            title=f"Distribution of {main_feat} by Class",
            template=TEMPLATE
        )
        charts.append(fig4)
        
    return charts

# Generate charts specifically for Regression tasks
def get_regression_charts(df, target):
    charts = [] # List to store generated figures

    # 1. Target Distribution (Histogram)
    fig1 = px.histogram(
        df, x=target, 
        title=f"Target Distribution: {target}",
        template=TEMPLATE
    )
    charts.append(fig1)

    # 2. Relationship with Trend (Scatter)
    num_features = [c for c in df.select_dtypes(include=['number']).columns if c != target]
    if len(num_features) > 0:
        # Select the most varied feature for the X-axis
        main_feat = sorted(num_features, key=lambda c: df[c].nunique(), reverse=True)[0]
        fig2 = px.scatter(
            df, x=main_feat, y=target, 
            trendline="ols", # Add ordinary least squares trendline
            title=f"Relationship: {main_feat} vs {target}",
            template=TEMPLATE
        )
        charts.append(fig2)

    # 3. Data Sequence (Line Chart)
    # Shows how values fluctuate across the dataset rows
    fig3 = px.line(
        df.head(200), y=target, # Display first 200 rows for clarity
        title=f"Data Sequence: {target} (First 200 rows)",
        template=TEMPLATE
    )
    charts.append(fig3)

    # 4. Target Spread (Box Plot)
    fig4 = px.box(
        df, y=target,
        title=f"Box Plot of {target}",
        template=TEMPLATE
    )
    charts.append(fig4)
            
    return charts