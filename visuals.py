import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Set standard template for consistency
TEMPLATE = "plotly_white"

def get_classification_charts(df, target):
    charts = []
    
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
        num_cols.remove(target)
        
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        fig2 = px.imshow(
            corr, 
            text_auto=False,
            title="Correlation Heatmap",
            color_continuous_scale='RdBu_r', 
            template=TEMPLATE,
            height=600
        )
        charts.append(fig2)

    # 3. Class Proportion (Pie Chart) - Simplified and Basic
    fig3 = px.pie(
        counts, values='count', names=target,
        title="Class Distribution (Percentage)",
        template=TEMPLATE,
        hole=0.3 # Donut style for a modern but simple look
    )
    charts.append(fig3)

    # 4. Feature Distribution (Box Plot)
    if len(num_cols) > 0:
        main_feat = sorted(num_cols, key=lambda c: df[c].nunique(), reverse=True)[0]
        fig4 = px.box(
            df, x=target, y=main_feat, color=target,
            title=f"Distribution of {main_feat} by Class",
            template=TEMPLATE
        )
        charts.append(fig4)
        
    return charts

def get_regression_charts(df, target):
    charts = []

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
        main_feat = sorted(num_features, key=lambda c: df[c].nunique(), reverse=True)[0]
        fig2 = px.scatter(
            df, x=main_feat, y=target, 
            trendline="ols",
            title=f"Relationship: {main_feat} vs {target}",
            template=TEMPLATE
        )
        charts.append(fig2)

    # 3. Data Sequence (Line Chart) - Simple and Basic
    # Shows the fluctuation of values across the dataset
    fig3 = px.line(
        df.head(200), y=target, # Showing first 200 for clarity
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