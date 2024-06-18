import pandas as pd
import numpy as np
# Normalize dataset stereomatch10.csv

# Load the dataset
df = pd.read_csv('stereomatch10.csv')

# Normalize the dataset exept time and cost columns that remain the same

# Normalize the dataset
for col in df.columns:
    if col not in ['exec_time_s', 'cost']:
        df[col] = (df[col] - df[col].mean()) / df[col].std()

# Save the normalized dataset
df.to_csv('scaledstereomatch10.csv', index=False)

