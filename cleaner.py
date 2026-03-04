import pandas as pd
import numpy as np

def clean_data(df, config):
    new_df = df.copy()
    changes = []
    
    # 1. Type Casting
    if 'type_changes' in config:
        for col, new_type in config['type_changes'].items():
            try:
                if new_type == "Numeric":
                    new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
                elif new_type == "Categorical":
                    new_df[col] = new_df[col].astype(str)
                elif new_type == "Temporal":
                    new_df[col] = pd.to_datetime(new_df[col], errors='coerce')
                changes.append(f"Casted {col} to {new_type}.")
            except Exception as e:
                changes.append(f"Failed to cast {col} to {new_type}: {e}")

    # 2. Duplicate Handling
    if config.get('drop_duplicates'):
        initial_count = len(new_df)
        new_df = new_df.drop_duplicates()
        if initial_count - len(new_df) > 0:
            changes.append(f"Removed {initial_count - len(new_df)} duplicates.")

    # 3. Missing Value Handling
    for col, action in config.get('missing_actions', {}).items():
        if action == "Drop Rows":
            new_df = new_df.dropna(subset=[col])
        elif action == "Mean":
            new_df[col] = new_df[col].fillna(new_df[col].mean())
        elif action == "Median":
            new_df[col] = new_df[col].fillna(new_df[col].median())
        elif action == "Zero-fill":
            new_df[col] = new_df[col].fillna(0)
        elif action == "Mode":
            mode_val = new_df[col].mode()
            if not mode_val.empty:
                new_df[col] = new_df[col].fillna(mode_val[0])
        elif action == "Unknown":
            new_df[col] = new_df[col].fillna("Unknown")

    # 4. Outlier Removal (3 Sigma)
    if config.get('remove_outliers'):
        num_cols = new_df.select_dtypes(include=[np.number]).columns
        # Excluding the target column from outlier removal if it's classification? 
        # Usually we remove based on features.
        initial_len = len(new_df)
        for col in num_cols:
            if new_df[col].std() > 0:
                z_score = (new_df[col] - new_df[col].mean()) / new_df[col].std()
                new_df = new_df[np.abs(z_score) < 3]
        if initial_len - len(new_df) > 0:
            changes.append(f"Removed {initial_len - len(new_df)} outliers (3-sigma rule).")
            
    return new_df, changes