import pandas as pd
from sklearn import datasets
import io

def load_sklearn_data(name):
    loaders = {
        "iris": datasets.load_iris,
        "wine": datasets.load_wine,
        "breast_cancer": datasets.load_breast_cancer,
        "diabetes": datasets.load_diabetes
    }
    data = loaders[name]()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

def load_csv(uploaded_file):
    try:
        # Try reading with default utf-8 first
        return pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        try:
            # Fallback to latin-1
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding='latin-1')
        except Exception as e:
            raise Exception(f"Error loading CSV (Latin-1 fallback also failed): {e}")
    except Exception as e:
        raise Exception(f"Error loading CSV: {e}")

def identify_task(df, target_col):
    if df[target_col].dtype == 'object' or df[target_col].dtype == 'bool':
        return "Classification"
    
    unique_ratio = df[target_col].nunique() / len(df)
    if unique_ratio < 0.05 or df[target_col].nunique() < 20:
        return "Classification"
    return "Regression"

def find_target_heuristic(columns):
    high_conf = ['target', 'label', 'class', 'y_true', 'output', 'survived']
    domain_spec = ['price', 'revenue', 'score', 'prediction', 'charges']
    cols_lower = [c.lower() for c in columns]
    
    for kw in high_conf:
        if kw in cols_lower:
            return columns[cols_lower.index(kw)]
    for kw in domain_spec:
        if kw in cols_lower:
            return columns[cols_lower.index(kw)]
    return columns[-1]