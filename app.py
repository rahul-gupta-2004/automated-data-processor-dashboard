import streamlit as st
import pandas as pd
import numpy as np
from processor import load_sklearn_data, load_csv, identify_task, find_target_heuristic
from cleaner import clean_data
from visuals import get_classification_charts, get_regression_charts

st.set_page_config(page_title="Data Dashboard", layout="wide")

# --- SESSION STATE ---
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'clean_data' not in st.session_state:
    st.session_state.clean_data = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'task_type' not in st.session_state:
    st.session_state.task_type = None

def reset_app():
    st.session_state.raw_data = None
    st.session_state.clean_data = None

st.title("Automated Data Processor and Dashboard")

# --- SIDEBAR: DATA IMPORT ---
st.sidebar.header("Data Source")
source = st.sidebar.radio("Choose Source", ["Scikit-Learn", "Upload CSV"], on_change=reset_app)

if source == "Scikit-Learn":
    ds_choice = st.sidebar.selectbox("Select Dataset", ["iris", "wine", "breast_cancer", "diabetes"])
    if st.sidebar.button("Load Dataset"):
        st.session_state.raw_data = load_sklearn_data(ds_choice)
        st.session_state.clean_data = None
else:
    file = st.sidebar.file_uploader("Upload CSV", type="csv", on_change=reset_app)
    if file:
        try:
            st.session_state.raw_data = load_csv(file)
        except Exception as e:
            st.error(f"Error: {e}")

# --- MAIN LOGIC ---
if st.session_state.raw_data is not None:
    df = st.session_state.raw_data
    
    # Grid for metrics
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Rows", df.shape[0])
    col_m2.metric("Features", df.shape[1])
    col_m3.metric("Numeric Cols", len(df.select_dtypes(include=[np.number]).columns))
    col_m4.metric("Null Values", df.isnull().sum().sum())

    # Tabs for navigation
    tabs = st.tabs(["Summary", "Cleaning", "Visualizations", "Data View"])

    # TAB 1: SUMMARY
    with tabs[0]:
        st.subheader("Data Summary")
        
        with st.expander("Data Preview", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)
        
        with st.expander("Data Info"):
            info_df = pd.DataFrame({
                "Column": df.columns,
                "Non-Null Count": df.notnull().sum().values,
                "Data Type": [str(t) for t in df.dtypes.values]
            })
            st.table(info_df)
            
        with st.expander("Descriptive Statistics"):
            st.dataframe(df.describe(), use_container_width=True)

    # TAB 2: CLEANING
    with tabs[1]:
        st.subheader("Cleaning Pipeline")
        
        suggested = find_target_heuristic(df.columns.tolist())
        target_col = st.selectbox("Select Target Column", df.columns, index=df.columns.get_loc(suggested))
        task_type = identify_task(df, target_col)
        st.info(f"Task Identified: {task_type}")

        config = {'missing_actions': {}, 'type_changes': {}, 'drop_duplicates': False, 'remove_outliers': False}
        
        st.write("Step 1: Type Conversion")
        for col in df.columns:
            config['type_changes'][col] = st.selectbox(
                f"Change {col} from {str(df[col].dtype)} to:", 
                ["No Change", "Numeric", "Categorical", "Temporal"], 
                key=f"type_{col}"
            )

        st.write("Step 2: Missing Values")
        null_cols = df.columns[df.isnull().any()]
        if len(null_cols) == 0:
            st.success("No missing values found.")
        for col in null_cols:
            is_num = df[col].dtype in ['int64', 'float64']
            options = ["None", "Mean", "Median", "Zero-fill", "Drop Rows"] if is_num else ["None", "Mode", "Unknown", "Drop Rows"]
            config['missing_actions'][col] = st.selectbox(f"Action for {col}", options, key=f"miss_{col}")

        config['drop_duplicates'] = st.checkbox("Remove Duplicates", value=True)
        if task_type == "Regression":
            config['remove_outliers'] = st.checkbox("Remove Outliers (3 Standard Deviations)")

        if st.button("Apply Cleaning"):
            cleaned_df, changes = clean_data(df, config)
            st.session_state.clean_data = cleaned_df
            st.session_state.target_col = target_col
            st.session_state.task_type = task_type
            
            for ch in changes:
                st.write(f"- {ch}")
            st.success("Cleaning applied successfully.")

    # TAB 3: VISUALIZATIONS
    with tabs[2]:
        if st.session_state.clean_data is None:
            st.warning("Please apply cleaning first.")
        else:
            clean_df = st.session_state.clean_data
            
            # Sidebar Filter Slicing
            st.sidebar.write("Data Filters")
            filtered_df = clean_df.copy()
            numeric_cols = clean_df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                min_v, max_v = float(clean_df[col].min()), float(clean_df[col].max())
                if min_v != max_v:
                    range_vals = st.sidebar.slider(f"Filter {col}", min_v, max_v, (min_v, max_v))
                    filtered_df = filtered_df[(filtered_df[col] >= range_vals[0]) & (filtered_df[col] <= range_vals[1])]

            st.subheader(f"{st.session_state.task_type} Analysis")
            if st.session_state.task_type == "Classification":
                charts = get_classification_charts(filtered_df, st.session_state.target_col)
            else:
                charts = get_regression_charts(filtered_df, st.session_state.target_col)
            
            # Grid display for 4 charts
            for i in range(0, len(charts), 2):
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(charts[i], use_container_width=True)
                if i + 1 < len(charts):
                    with col2:
                        st.plotly_chart(charts[i+1], use_container_width=True)

    # TAB 4: DATA VIEW
    with tabs[3]:
        st.subheader("Processed CSV File")
        if st.session_state.clean_data is not None:
            st.dataframe(st.session_state.clean_data, use_container_width=True)
            
            st.divider()
            st.subheader("Download Section")
            csv_file = st.session_state.clean_data.to_csv(index=False).encode('utf-8')
            st.download_button("Download Cleaned CSV", csv_file, "cleaned_data.csv", "text/csv")
        else:
            st.info("Cleaned data will appear here after you click 'Apply Cleaning' in the Cleaning tab.")
            st.write("Current Raw Data Preview:")
            st.dataframe(df, use_container_width=True)

else:
    st.info("Please load a dataset to start.")