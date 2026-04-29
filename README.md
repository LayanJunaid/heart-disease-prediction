# heart-disease-prediction
🫀 Heart Disease Prediction using Machine Learning
📌 Overview

This project aims to predict the presence of heart disease using machine learning techniques. By analyzing patient health attributes such as age, blood pressure, cholesterol levels, and other clinical features, the model can classify whether a patient is at risk of heart disease.

Early prediction of heart disease plays a crucial role in reducing mortality and enabling timely medical intervention.

🎯 Objectives
Build a machine learning pipeline for heart disease prediction
Apply data preprocessing techniques
Compare multiple classification models
Evaluate model performance using different metrics
Identify the best-performing model
📊 Dataset

The dataset used in this project contains medical attributes such as:

Age
Sex
Chest pain type
Resting blood pressure
Cholesterol
Maximum heart rate
And other clinical features

The target variable indicates:

0 → No heart disease
1 → Presence of heart disease
⚙️ Methodology
1. Data Preprocessing
Case folding (lowercasing text if needed)
Tokenization (if text involved)
Handling missing values
Removing noise and irrelevant data
Feature scaling (e.g., normalization/standardization)
2. Exploratory Data Analysis (EDA)
Data visualization using graphs
Correlation analysis
Identifying important features
3. Feature Selection
Removing irrelevant features
Keeping only the most impactful variables
Reducing dimensionality
4. Model Building

Several machine learning models are implemented, such as:

Logistic Regression
Decision Tree
Random Forest
Support Vector Machine (SVM)
Naive Bayes
📈 Model Evaluation

Models are evaluated using:

Accuracy
Precision
Recall
F1-score

These metrics help ensure reliable predictions, especially in medical applications where recall is critical.

🏆 Results
Different models are compared
The best model is selected based on performance metrics
Feature selection impact is analyzed
🚀 How to Run the Project
Clone the repository:
git clone https://github.com/LayanJunaid/heart-disease-prediction.git
Install dependencies:
pip install -r requirements.txt
Run the notebook:
Open the .ipynb file using Jupyter Notebook or Google Colab
Run all cells sequentially
🧠 Technologies Used
Python
Pandas & NumPy
Scikit-learn
Matplotlib & Seaborn
Jupyter Notebook
⚠️ Disclaimer

This project is for educational purposes only and should not be used as a substitute for professional medical diagnosis.
