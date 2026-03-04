# Automated Data Processor and Dashboard

Live Demo: [https://automated-data-dashboard.streamlit.app/](https://automated-data-dashboard.streamlit.app/)

A comprehensive Streamlit-based application designed for data exploration, cleaning, and visualization. This tool simplifies the data preprocessing pipeline and provides interactive insights through automated task identification.

## Project Description

The Data Processor and Dashboard is an interactive tool that allows users to load datasets, perform essential data cleaning operations, and visualize trends based on the nature of the data (Classification or Regression). It integrates seamless data handling with dynamic Plotly visualizations.

## Key Features

- **Data Loading**: Load standard Scikit-Learn datasets (Iris, Wine, Breast Cancer, Diabetes) or upload custom CSV files.
- **Exploratory Data Analysis**: View data summaries, descriptive statistics, and detailed column information.
- **Cleaning Pipeline**:
    - Interactive type conversion for columns.
    - Multiple strategies for handling missing values (Mean, Median, Mode, Zero-fill, etc.).
    - Automatic removal of duplicates and outliers.
- **Automated Task Identification**: Heuristic-based detection of whether a dataset represents a Classification or Regression task.
- **Interactive Visualizations**: Dynamic Plotly charts tailored to the identified task type, including distribution plots, correlation heatmaps, and scatter plots.
- **Data Filtering**: Sidebar sliders to filter numeric data in real-time.
- **Export Data**: Download the cleaned and processed dataset as a CSV file.

## Technologies Used

- Python - Core programming language
- Streamlit - Web application framework
- Pandas - Data manipulation and analysis
- NumPy - Numerical operations
- Scikit-learn - Dataset sourcing and machine learning utilities
- Plotly - Interactive data visualizations
- Statsmodels - Statistical modeling and analysis

## Installation and Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd cia_3
   ```

2. **Create a virtual environment**:
   ```bash
   python -x venv app_cia3_venv
   source app_cia3_venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Launch the app and use the sidebar to select a data source (Scikit-Learn or CSV Upload).
2. Explore the **Summary** tab to understand your data.
3. Use the **Cleaning** tab to define how to handle types and missing values, then click "Apply Cleaning".
4. Navigate to the **Visualizations** tab to see automated insights.
5. Download your processed data from the **Data View** tab.

## Sample Datasets

The repository includes a `datasets/` directory with several CSV files that can be used to test the application features:

- **Titanic-Dataset.csv**: Classic classification dataset for predicting survival.
- **mushrooms.csv**: Classification dataset for identifying poisonous vs edible mushrooms.
- **insurance.csv**: Regression dataset for predicting medical insurance costs.
- **houses.csv**: Regression dataset for predicting housing prices.
- **student's dropout dataset.csv**: Complex classification dataset for academic performance.

Users can upload these files using the "Upload CSV" option in the sidebar to explore different visualization types (Classification vs Regression).

## File Structure

- `app.py`: The main Streamlit application script.
- `cleaner.py`: Logic for data cleaning and outlier removal.
- `processor.py`: Utilities for data loading and task identification.
- `visuals.py`: Plotly chart generation for different task types.
- `datasets/`: Directory containing sample CSV files.
- `requirements.txt`: List of Python dependencies.
