/**

 * Helpers for normalizing and describing UCI Cleveland features.
 */

const FEATURE_LABELS = {
  age: "Age",
  sex: "Sex",
  cp: "Chest Pain Type",
  trestbps: "Resting Blood Pressure",
  chol: "Serum Cholesterol",
  fbs: "Fasting Blood Sugar",
  restecg: "Resting ECG",
  thalach: "Maximum Heart Rate",
  exang: "Exercise Induced Angina",
  oldpeak: "ST Depression",
  slope: "Slope of ST Segment",
  ca: "Major Vessels (Fluoroscopy)",
  thal: "Thalassemia",
};


 // Converts raw feature values into a human-readable object.
 
exports.formatFeaturesForDisplay = (features) => {
  return Object.entries(features).map(([key, value]) => ({
    key,
    label: FEATURE_LABELS[key] || key,
    value,
  }));
};


 // casting all to numbers and stripping extra fields
exports.sanitizeFeatures = (body) => {
  const keys = Object.keys(FEATURE_LABELS);
  const result = {};
  for (const key of keys) {
    if (body[key] !== undefined) {
      result[key] = Number(body[key]);
    }
  }
  return result;
};
