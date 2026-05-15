
 // Returns static educational / project resources.


exports.getResources = async (req, res) => {
  res.json({
    success: true,
    resources: {
      project: {
        githubRepository: process.env.GITHUB_REPO_URL || "https://github.com/your-team/heart-disease-prediction",
        dataset: "https://archive.ics.uci.edu/dataset/45/heart+disease",
        datasetDirect: "https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci",
      },
      technologiesUsed: [
        "AI / Machine Learning (scikit-learn, XGBoost, SMOTE)",
        "Node.js / Express.js",
        "MongoDB / Mongoose",
        "React (Frontend)",
        "Python (ML Bridge)",
      ],
      team: {
        members: [
          { name: "Sidra Ashram" },
          { name: "Layan Junaid" },
          { name: "Ruha Kabbani" },
        ],
        supervisor: "Shahaboddin Daneshvar",
      },
      features: [
        { key: "age", label: "Age", description: "Age of the patient in years", unit: "years" },
        { key: "sex", label: "Sex", description: "0 = Female, 1 = Male", unit: "" },
        { key: "cp", label: "Chest Pain Type", description: "0=Typical angina, 1=Atypical, 2=Non-anginal, 3=Asymptomatic", unit: "" },
        { key: "trestbps", label: "Resting Blood Pressure", description: "Resting blood pressure on admission", unit: "mm Hg" },
        { key: "chol", label: "Serum Cholesterol", description: "Serum cholesterol level", unit: "mg/dl" },
        { key: "fbs", label: "Fasting Blood Sugar", description: "Fasting blood sugar > 120 mg/dl: 1=yes, 0=no", unit: "" },
        { key: "restecg", label: "Resting ECG", description: "0=Normal, 1=ST-T abnormality, 2=LV hypertrophy", unit: "" },
        { key: "thalach", label: "Max Heart Rate", description: "Maximum heart rate achieved", unit: "bpm" },
        { key: "exang", label: "Exercise Induced Angina", description: "1=yes, 0=no", unit: "" },
        { key: "oldpeak", label: "ST Depression", description: "ST depression induced by exercise relative to rest", unit: "" },
        { key: "slope", label: "Slope of ST Segment", description: "0=Upsloping, 1=Flat, 2=Downsloping", unit: "" },
        { key: "ca", label: "Major Vessels", description: "Number of major vessels colored by fluoroscopy (0-3)", unit: "" },
        { key: "thal", label: "Thalassemia", description: "1=Normal, 2=Fixed defect, 3=Reversible defect", unit: "" },
      ],
    },
  });
};
