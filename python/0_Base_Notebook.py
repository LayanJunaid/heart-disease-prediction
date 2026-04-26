# Generated from: 0_Base_Notebook.ipynb
# Converted at: 2026-04-10T10:43:56.474Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# <a href="https://colab.research.google.com/github/LayanJunaid/heart-disease-prediction/blob/main/0_Base_Notebook.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


!pip install -q seaborn --upgrade


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import joblib


sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)

print(" All libraries loaded successfully!")

from google.colab import files
uploaded = files.upload()

col_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
             'restecg', 'thalach', 'exang', 'oldpeak',
             'slope', 'ca', 'thal', 'condition']

df = pd.read_csv('processed.cleveland.data',
                 names=col_names,
                 na_values='?')

df['condition'] = df['condition'].apply(lambda x: 1 if x > 0 else 0)

print("Dataset loaded!")
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

print("=" * 55)
print("FIRST 5 ROWS:")
display(df.head())

print("\nLAST 5 ROWS:")
display(df.tail())

print("\nCOLUMN NAMES:")
print(df.columns.tolist())

print("=" * 55)
print("DATA TYPES & NON-NULL COUNTS:")
print()
df.info()

print("=" * 55)
print("STATISTICAL SUMMARY:")
print()
display(df.describe().round(2))

print("=" * 55)
print("MISSING VALUES ANALYSIS:")
print()

missing_count = df.isnull().sum()
missing_pct   = (missing_count / len(df) * 100).round(2)

missing_df = pd.DataFrame({
    'Missing Count': missing_count,
    'Missing %':     missing_pct
})

display(missing_df)

plt.figure(figsize=(12, 4))
sns.heatmap(df.isnull(), cbar=False,
            cmap='viridis', yticklabels=False)
plt.title('Missing Values Heatmap\n(Yellow = Missing)', fontsize=13)
plt.tight_layout()
plt.savefig('missing_values.png', dpi=150)
plt.show()

print("\n Missing values check complete!")

print("=" * 55)
print("TARGET VARIABLE DISTRIBUTION:")
print()
print(df['condition'].value_counts())
print()
print("Class Balance (%):")
print((df['condition'].value_counts(normalize=True) * 100).round(1))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.countplot(x='condition', data=df,
              palette='Set2', ax=axes[0],
              edgecolor='white')
axes[0].set_title('Heart Disease Count', fontsize=13)
axes[0].set_xlabel('0 = No Disease  |  1 = Disease')
axes[0].set_ylabel('Count')
for p in axes[0].patches:
    axes[0].annotate(f'{int(p.get_height())}',
                     (p.get_x() + p.get_width()/2, p.get_height()),
                     ha='center', va='bottom', fontsize=12)

df['condition'].value_counts().plot(
    kind='pie', ax=axes[1],
    labels=['No Disease', 'Disease'],
    autopct='%1.1f%%',
    colors=['#66b3ff', '#ff9999'],
    startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
axes[1].set_title('Class Distribution (%)', fontsize=13)
axes[1].set_ylabel('')

plt.suptitle('Target Variable Analysis', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('target_distribution.png', dpi=150)
plt.show()

feature_info = {
    'age':      ['Numerical',   'Age of patient in years'],
    'sex':      ['Categorical', '1 = Male, 0 = Female'],
    'cp':       ['Categorical', 'Chest pain type: 0=Typical, 1=Atypical, 2=Non-anginal, 3=Asymptomatic'],
    'trestbps': ['Numerical',   'Resting blood pressure (mmHg)'],
    'chol':     ['Numerical',   'Serum cholesterol (mg/dl)'],
    'fbs':      ['Categorical', 'Fasting blood sugar > 120mg/dl: 1=True, 0=False'],
    'restecg':  ['Categorical', 'Resting ECG results: 0=Normal, 1=ST-T abnormality, 2=LV hypertrophy'],
    'thalach':  ['Numerical',   'Maximum heart rate achieved'],
    'exang':    ['Categorical', 'Exercise induced angina: 1=Yes, 0=No'],
    'oldpeak':  ['Numerical',   'ST depression induced by exercise relative to rest'],
    'slope':    ['Categorical', 'Slope of peak exercise ST segment: 0=Up, 1=Flat, 2=Down'],
    'ca':       ['Categorical', 'Number of major vessels colored by fluoroscopy (0-3)'],
    'thal':     ['Categorical', 'Thalassemia: 0=Normal, 1=Fixed defect, 2=Reversible defect'],
    'condition':['TARGET',      '0 = No Heart Disease | 1 = Heart Disease Present']
}

desc_df = pd.DataFrame.from_dict(
    feature_info, orient='index',
    columns=['Type', 'Description']
)
desc_df.index.name = 'Feature'

print("FEATURE DESCRIPTIONS:")
display(desc_df)

print("=" * 55)
print("NUMERICAL FEATURES DISTRIBUTION")
print()

num_features = ['age','trestbps','chol','thalach','oldpeak']

df[num_features].hist(
    bins=20,
    figsize=(14,8),
    color='#2E86AB',
    edgecolor='black'
)

plt.suptitle("Distribution of Numerical Features", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig("numerical_distributions.png", dpi=150)
plt.show()

print("=" * 55)
print("OUTLIER DETECTION WITH BOXPLOTS")
print()

plt.figure(figsize=(14,8))

for i, col in enumerate(num_features):
    plt.subplot(2,3,i+1)
    sns.boxplot(x=df[col], color='#66B2FF')
    plt.title(col)

plt.suptitle("Outlier Detection in Numerical Features", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig("boxplots_outliers.png", dpi=150)
plt.show()

print("=" * 55)
print("CATEGORICAL FEATURES DISTRIBUTION")
print()

cat_features = ['sex','cp','fbs','restecg','exang','slope','ca','thal']

plt.figure(figsize=(14,10))

for i, col in enumerate(cat_features):
    plt.subplot(3,3,i+1)
    sns.countplot(data=df, x=col, palette="Set2")
    plt.title(col)

plt.suptitle("Categorical Feature Distributions", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig("categorical_distributions.png", dpi=150)
plt.show()

print("=" * 55)
print("NUMERICAL FEATURES VS TARGET")
print()

plt.figure(figsize=(14,8))

for i, col in enumerate(num_features):
    plt.subplot(2,3,i+1)
    sns.boxplot(x='condition', y=col, data=df, palette="coolwarm")
    plt.title(f"{col} vs Heart Disease")

plt.suptitle("Numerical Features vs Target Variable", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig("numerical_vs_target.png", dpi=150)
plt.show()

print("=" * 55)
print("CATEGORICAL FEATURES VS TARGET")
print()

cat_features = ['sex','cp','fbs','restecg','exang','slope','ca','thal']

plt.figure(figsize=(16,12))

for i, col in enumerate(cat_features):
    plt.subplot(3,3,i+1)
    sns.countplot(
        data=df,
        x=col,
        hue='condition',
        palette=['#66b3ff','#ff9999'],
        edgecolor='white'
    )
    plt.title(f"{col} vs Heart Disease")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.legend(title='Condition', labels=['No Disease','Disease'])

plt.suptitle("Categorical Features vs Target Variable", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("categorical_vs_target.png", dpi=150)
plt.show()

print("=" * 55)
print("CORRELATION HEATMAP")
print()

plt.figure(figsize=(12,8))

corr = df.corr(numeric_only=True)

sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    linewidths=0.5
)

plt.title("Feature Correlation Matrix", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.show()

print("=" * 55)
print("PAIRPLOT ANALYSIS")
print()

sns.pairplot(
    df[['age','chol','thalach','oldpeak','condition']],
    hue='condition',
    palette='Set1'
)

plt.suptitle("Pairwise Feature Relationships", y=1.02)
plt.savefig("pairplot.png", dpi=150)
plt.show()

# ============================================================
# SAVE RAW DATAFRAME FOR DOWNSTREAM NOTEBOOKS
# ============================================================
# Create the shared saved_data folder — all subsequent notebooks read from here
SAVE_DIR = 'saved_data'
os.makedirs(SAVE_DIR, exist_ok=True)

# Save the raw loaded DataFrame (with binary condition, NaNs still present)
# 03_preprocessing.py will load this instead of re-uploading the file
joblib.dump(df, f'{SAVE_DIR}/raw_df.pkl')

print(f'\n  Raw DataFrame saved to: {SAVE_DIR}/raw_df.pkl')
print(f'  Shape: {df.shape[0]} rows × {df.shape[1]} columns')
print(f'  Missing values: {df.isnull().sum().sum()} total NaN values')
print('  → 03_preprocessing.py will load from this file.')
print('=' * 55)
print('  0_Base_Notebook complete! Proceed to 03_preprocessing.py')
print('=' * 55)
