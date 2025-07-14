import pandas as pd
import numpy as np
import config

def generate_data(n_samples, concept, random_state=42):
    """Generates synthetic data for a given concept."""
    np.random.seed(random_state)
    # Feature 1: A clear indicator
    X1 = np.random.rand(n_samples) * 10
    # Feature 2: Noise
    X2 = np.random.randn(n_samples) * 5
    
    # The relationship between X1 and y changes based on the concept
    if concept == 'A':
        # Concept A: y increases with X1
        y = (2 * X1 + X2/2 + np.random.randn(n_samples)) > 10
    else: # Concept B
        # Concept B: y decreases with X1
        y = (2 * (10 - X1) + X2/2 + np.random.randn(n_samples)) > 10
        
    df = pd.DataFrame({'feature1': X1, 'feature2': X2, 'target': y.astype(int)})
    return df

def generate_drift_dataset():
    """Generates a dataset with an abrupt concept drift."""
    print("--- Generating Synthetic Drift Dataset ---")
    
    # Concept A for the first half
    df_concept_a = generate_data(500, 'A', random_state=42)
    
    # Concept B for the second half
    df_concept_b = generate_data(500, 'B', random_state=43) # Different seed for variety
    
    # Combine them to create a drift
    drift_df = pd.concat([df_concept_a, df_concept_b], ignore_index=True)
    
    # Save the dataset
    output_path = config.PROCESSED_DATA_DIR / "drift_simulation_data.csv"
    drift_df.to_csv(output_path, index=False)
    
    print(f"Successfully generated and saved drift dataset with {len(drift_df)} samples.")
    print(f"Dataset saved to: {output_path}")

if __name__ == '__main__':
    generate_drift_dataset()