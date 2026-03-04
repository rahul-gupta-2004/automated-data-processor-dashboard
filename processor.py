import pandas as pd
from sklearn import datasets
import io

# Load sample datasets from Scikit-Learn library
def load_sklearn_data(name):
    loaders = {
        "iris": datasets.load_iris,
        "wine": datasets.load_wine,
        "breast_cancer": datasets.load_breast_cancer,
        "diabetes": datasets.load_diabetes
    }
    data = loaders[name]() # Load the chosen dataset
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target # Add the target column
    return df

# Load data from a CSV file with encoding fallback
def load_csv(uploaded_file):
    try:
        # Try reading with standard UTF-8 encoding
        return pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        try:
            # Fallback to Latin-1 encoding if UTF-8 fails
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding='latin-1')
        except Exception as e:
            raise Exception(f"Error loading CSV (Latin-1 fallback also failed): {e}")
    except Exception as e:
        raise Exception(f"Error loading CSV: {e}")

# Identify if the task is Classification or Regression based on target column
def identify_task(df, target_col):
    # Check if data type is object or boolean
    if df[target_col].dtype == 'object' or df[target_col].dtype == 'bool':
        return "Classification"
    
    # Check ratio of unique values to determine task type
    unique_ratio = df[target_col].nunique() / len(df)
    if unique_ratio < 0.05 or df[target_col].nunique() < 20:
        return "Classification"
    return "Regression"

# Find the target column using common keywords in column names
def find_target_heuristic(columns):
    high_conf = ['target', 'label', 'class', 'y_true', 'output', 'survived'] # Common target names
    domain_spec = ['price', 'revenue', 'score', 'prediction', 'charges'] # Domain specific targets
    cols_lower = [c.lower() for c in columns]
    
    # Search for keywords in lowercased column names
    for kw in high_conf:
        if kw in cols_lower:
            return columns[cols_lower.index(kw)]
    for kw in domain_spec:
        if kw in cols_lower:
            return columns[cols_lower.index(kw)]
    return columns[-1] # Default to the last column