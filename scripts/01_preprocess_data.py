import pandas as pd
import config # Import our configuration file

def preprocess_rul_data():
    """
    Processes the raw NASA Turbofan dataset (FD001) to create the final
    dataframe for RUL prediction, including feature engineering.
    """
    print("--- Starting RUL Data Preprocessing ---")

    # Define column names based on the dataset documentation
    column_names = [
        'unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3',
        'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5', 'sensor_6',
        'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10', 'sensor_11', 'sensor_12',
        'sensor_13', 'sensor_14', 'sensor_15', 'sensor_16', 'sensor_17', 'sensor_18',
        'sensor_19', 'sensor_20', 'sensor_21'
    ]

    # Load the training data for FD001 using the path from config
    df = pd.read_csv(
        config.NASA_DATA_DIR / 'train_FD001.txt',
        sep=r'\s+',
        header=None,
        names=column_names
    )
    print(f"Loaded raw RUL data with shape: {df.shape}")

    # Calculate RUL
    max_cycles = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    max_cycles.columns = ['unit_number', 'max_cycles']
    df = pd.merge(df, max_cycles, on='unit_number', how='left')
    df['RUL'] = df['max_cycles'] - df['time_in_cycles']
    df.drop(columns=['max_cycles'], inplace=True)

    # Drop useless columns identified in EDA
    df.drop(columns=config.RUL_COLS_TO_DROP, inplace=True)
    print(f"Dropped useless columns. Shape is now: {df.shape}")

    # Create rolling window features
    sensor_cols = [col for col in df.columns if col.startswith('sensor')]
    for col in sensor_cols:
        df[f'{col}_rolling_mean'] = df.groupby('unit_number')[col].rolling(window=config.RUL_WINDOW_SIZE, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f'{col}_rolling_std'] = df.groupby('unit_number')[col].rolling(window=config.RUL_WINDOW_SIZE, min_periods=1).std().reset_index(level=0, drop=True)
    
    df.fillna(0, inplace=True)
    print(f"Added rolling features. Final shape: {df.shape}")

    # Cap RUL
    df['RUL'] = df['RUL'].clip(upper=config.RUL_CAP)

    # Save the processed data
    output_path = config.PROCESSED_DATA_DIR / "rul_processed_data.csv"
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True) # Ensure the directory exists
    df.to_csv(output_path, index=False)
    print(f"Successfully saved processed RUL data to {output_path}")

def preprocess_classification_data():
    """
    Processes the raw Kaggle Predictive Maintenance dataset for fault classification.
    """
    print("\n--- Starting Classification Data Preprocessing ---")

    # Load data using the path from config
    df = pd.read_csv(config.KAGGLE_DATA_FILE)
    print(f"Loaded raw classification data with shape: {df.shape}")

    # Clean column names
    original_columns = df.columns
    new_columns = [col.replace('[K]', '').replace('[rpm]', '').replace('[Nm]', '').replace('[min]', '').strip().replace(' ', '_').lower() for col in original_columns]
    df.columns = new_columns

    # Drop useless ID columns from config
    df = df.drop(columns=config.CLASSIFICATION_COLS_TO_DROP)
    print(f"Dropped ID columns. Final shape: {df.shape}")
    
    # Save the processed data
    output_path = config.PROCESSED_DATA_DIR / "classification_processed_data.csv"
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True) # Ensure the directory exists
    df.to_csv(output_path, index=False)
    print(f"Successfully saved processed classification data to {output_path}")

if __name__ == '__main__':
    # This block runs only when the script is executed directly
    preprocess_rul_data()
    preprocess_classification_data()
    print("\nPreprocessing complete!")