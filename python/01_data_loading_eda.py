# Generated from: 01_data_loading_eda.ipynb
# Converted at: 2026-04-10T10:44:31.240Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# <a href="https://colab.research.google.com/github/LayanJunaid/heart-disease-prediction/blob/main/01_data_loading_eda.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


!pip install -q seaborn --upgrade


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


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