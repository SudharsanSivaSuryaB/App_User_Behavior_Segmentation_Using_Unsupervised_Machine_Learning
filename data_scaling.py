import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the featured data
df = pd.read_csv("featured_data/featured_data.csv")

print(f"Original data shape: {df.shape}")
print(f"Original data statistics:\n{df.describe()}\n")

# Initialize the StandardScaler
scaler = StandardScaler()

# Apply scaling to all features
scaled_data = scaler.fit_transform(df)

# Convert the scaled data back to a DataFrame to maintain column names
df_scaled = pd.DataFrame(scaled_data, columns=df.columns)

print(f"Scaled data shape: {df_scaled.shape}")
print(f"Scaled data statistics:\n{df_scaled.describe()}\n")

# Verify that scaling was applied correctly (mean should be ~0, std should be ~1)
print("Verification of scaling:")
print(f"Mean of scaled features:\n{df_scaled.mean()}")
print(f"\nStandard deviation of scaled features:\n{df_scaled.std()}\n")

# Save the scaled data
df_scaled.to_csv("featured_data/featured_data_scaled.csv", index=False)

print("Data Scaling Done: scaled features saved to featured_data/featured_data_scaled.csv")
