import pandas as pd
import numpy as np

# Main function to clean data based on user configuration
def clean_data(df, config):
    new_df = df.copy() # Make a copy to avoid original data modification
    changes = [] # List to track changes made
    
    # 1. Type Casting
    if 'type_changes' in config:
        for col, new_type in config['type_changes'].items():
            try:
                # Convert column to Numeric
                if new_type == "Numeric":
                    new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
                # Convert column to Categorical
                elif new_type == "Categorical":
                    new_df[col] = new_df[col].astype(str)
                # Convert column to Temporal (Date/Time)
                elif new_type == "Temporal":
                    new_df[col] = pd.to_datetime(new_df[col], errors='coerce')
                changes.append(f"Casted {col} to {new_type}.")
            except Exception as e:
                changes.append(f"Failed to cast {col} to {new_type}: {e}")

    # 2. Duplicate Handling
    if config.get('drop_duplicates'):
        initial_count = len(new_df)
        new_df = new_df.drop_duplicates() # Remove duplicate rows
        if initial_count - len(new_df) > 0:
            changes.append(f"Removed {initial_count - len(new_df)} duplicates.")

    # 3. Missing Value Handling
    for col, action in config.get('missing_actions', {}).items():
        if action == "Drop Rows":
            new_df = new_df.dropna(subset=[col]) # Drop rows with null values
        elif action == "Mean":
            new_df[col] = new_df[col].fillna(new_df[col].mean()) # Fill with column mean
        elif action == "Median":
            new_df[col] = new_df[col].fillna(new_df[col].median()) # Fill with column median
        elif action == "Zero-fill":
            new_df[col] = new_df[col].fillna(0) # Fill with zero
        elif action == "Mode":
            mode_val = new_df[col].mode()
            if not mode_val.empty:
                new_df[col] = new_df[col].fillna(mode_val[0]) # Fill with most frequent value
        elif action == "Unknown":
            new_df[col] = new_df[col].fillna("Unknown") # Fill with 'Unknown' string

    # 4. Outlier Removal (3 Sigma)
    if config.get('remove_outliers'):
        num_cols = new_df.select_dtypes(include=[np.number]).columns
        initial_len = len(new_df)
        for col in num_cols:
            if new_df[col].std() > 0:
                # Keep rows within 3 standard deviations
                z_score = (new_df[col] - new_df[col].mean()) / new_df[col].std()
                new_df = new_df[np.abs(z_score) < 3]
        if initial_len - len(new_df) > 0:
            changes.append(f"Removed {initial_len - len(new_df)} outliers (3-sigma rule).")
            
    return new_df, changes