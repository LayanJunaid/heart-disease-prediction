# Generated from: 02_EDA_Visualization.ipynb
# Converted at: 2026-04-10T10:45:11.319Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# <a href="https://colab.research.google.com/github/LayanJunaid/heart-disease-prediction/blob/main/02_EDA_Visualization.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

%matplotlib inline
sns.set_style("whitegrid")

from google.colab import files
uploaded = files.upload()

column_names = [
    'age','sex','cp','trestbps','chol','fbs',
    'restecg','thalach','exang','oldpeak',
    'slope','ca','thal','condition'
]

df = pd.read_csv(
    "processed.cleveland.data",
    names=column_names
)

print("Dataset Shape:", df.shape)
df.head()

df['condition'] = df['condition'].apply(lambda x: 0 if x == 0 else 1)

df['condition'].value_counts()

print("Dataset Info:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

numerical = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

categorical = [
    'sex', 'cp', 'fbs', 'restecg',
    'exang', 'slope', 'ca', 'thal'
]

print("Numerical features:", numerical)
print("Categorical features:", categorical)

fig, axes = plt.subplots(2, 3, figsize=(16,10))
axes = axes.flatten()

for i, col in enumerate(numerical):

    axes[i].hist(df[col], bins=25,
                 color='steelblue',
                 edgecolor='white',
                 alpha=0.8)

    axes[i].axvline(df[col].mean(),
                    color='red',
                    linestyle='--',
                    label=f'Mean: {df[col].mean():.1f}')

    axes[i].axvline(df[col].median(),
                    color='green',
                    linestyle='--',
                    label=f'Median: {df[col].median():.1f}')

    axes[i].set_title(f'Distribution of {col}')
    axes[i].legend(fontsize=8)

axes[-1].set_visible(False)

plt.suptitle('Numerical Features Distribution', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(16,10))
axes = axes.flatten()

for i, col in enumerate(numerical):

    sns.boxplot(x='condition',
                y=col,
                data=df,
                palette='Set2',
                ax=axes[i])

    axes[i].set_title(f'{col} vs Heart Disease')
    axes[i].set_xlabel('0 = No Disease | 1 = Disease')

axes[-1].set_visible(False)

plt.suptitle('Numerical Features vs Target')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2,4, figsize=(20,10))
axes = axes.flatten()

for i, col in enumerate(categorical):

    ct = pd.crosstab(df[col], df['condition'],
                     normalize='index') * 100

    ct.plot(kind='bar',
            ax=axes[i],
            color=['#66b3ff','#ff9999'],
            edgecolor='white',
            rot=0)

    axes[i].set_title(f'{col} vs Heart Disease (%)')
    axes[i].set_ylabel('Percentage')
    axes[i].legend(['No Disease','Disease'])

plt.suptitle('Categorical Features vs Target')

plt.tight_layout()
plt.show()

plt.figure(figsize=(12,9))

corr = df.corr(numeric_only=True)

mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='RdYlGn',
    center=0,
    square=True,
    linewidths=0.5
)

plt.title('Correlation Matrix — All Features')

plt.tight_layout()
plt.show()

target_corr = corr['condition'].drop('condition')\
                               .abs()\
                               .sort_values(ascending=False)

print("Top Correlations with Target:")
display(target_corr)

top_features = target_corr.head(5).index.tolist() + ['condition']

sns.pairplot(
    df[top_features],
    hue='condition',
    palette={0:'#66b3ff',1:'#ff6666'},
    plot_kws={'alpha':0.6}
)

plt.suptitle('Pairplot — Top 5 Correlated Features', y=1.02)

plt.show()

outlier_summary = {}

for col in numerical:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR

    outliers = df[(df[col] < lower) |
                  (df[col] > upper)]

    outlier_summary[col] = {
        'Count': len(outliers),
        'Percentage': f"{len(outliers)/len(df)*100:.1f}%"
    }

pd.DataFrame(outlier_summary).T