import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split

#1. Load data
df = pd.read_csv("kelulusan_mahasiswa.csv")

print("Dataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

#2. Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

#3. Remove duplicates
df = df.drop_duplicates()
print(f"\nRows after removing duplicates: {len(df)}")

#4. EDA - Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['IPK'])
plt.title('IPK Distribution - Boxplot')
plt.savefig("boxplot_ipk.png", bbox_inches='tight')
plt.close()

#5. Descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe())

#6. Histogram
plt.figure(figsize=(8, 6))
sns.histplot(df['IPK'], bins=10, kde=True)
plt.title('IPK Distribution - Histogram')
plt.savefig("histogram_ipk.png", bbox_inches='tight')
plt.close()

#7. Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='IPK', y='Waktu_Belajar_Jam', data=df, hue='Lulus')
plt.title('IPK vs Study Time by Graduation Status')
plt.legend(title='Lulus', labels=['Tidak Lulus', 'Lulus'])
plt.savefig("scatter_ipk_vs_study.png", bbox_inches='tight')
plt.close()

#8. Correlation heatmap
plt.figure(figsize=(10, 8))
numeric_cols = df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", fmt='.2f')
plt.title('Correlation Heatmap')
plt.savefig("heatmap_correlation.png", bbox_inches='tight')
plt.close()

#9. Feature engineering
df['Rasio_Absensi'] = df['Jumlah_Absensi'] / 14  # Assuming 14 total sessions
df['IPK_x_Study'] = df['IPK'] * df['Waktu_Belajar_Jam']
df.to_csv("processed_kelulusan.csv", index=False)
print("\nProcessed data saved to 'processed_kelulusan.csv'")

#10. Train-test split
X = df.drop('Lulus', axis=1)
y = df['Lulus']

print(f"\nDataset size: {len(df)} samples")
print(f"Class distribution: {y.value_counts().to_dict()}")

if len(df) < 30:
    print("\n WARNING: Dataset is very small (< 30 samples).")
    print("Consider collecting more data for reliable model training.")

try:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
except ValueError as e:
    print(f"\n Stratification failed: {e}")
    print("Using non-stratified split instead.")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42)

print(f"\nSplit sizes:")
print(f"Training: {X_train.shape[0]} samples")
print(f"Validation: {X_val.shape[0]} samples")
print(f"Test: {X_test.shape[0]} samples")