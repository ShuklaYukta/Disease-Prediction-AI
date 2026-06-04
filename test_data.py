import pandas as pd
df = pd.read_csv('dataset/training.csv')
print("✅ SUCCESS!")
print(f"Shape: {df.shape}")           # (4920, 133)
print(f"Diseases: {df['prognosis'].nunique()}")  # 42
print(df['prognosis'].value_counts().head())
