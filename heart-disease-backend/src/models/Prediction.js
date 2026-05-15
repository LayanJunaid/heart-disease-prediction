/**
 * Prediction Model — models/Prediction.js
 *
 * Stores each prediction request along with its features and result.
 * Feature names map directly to the UCI Cleveland Heart Disease dataset.
 */

const mongoose = require("mongoose");

const featuresSchema = new mongoose.Schema(
  {
    age: {
      type: Number,
      required: true,
      min: [1, "Age must be positive"],
      max: [120, "Age seems unrealistic"],
    },
    sex: {
      type: Number,
      required: true,
      enum: { values: [0, 1], message: "Sex must be 0 (female) or 1 (male)" },
    },
    cp: {
      // Chest pain type: 0=typical angina, 1=atypical, 2=non-anginal, 3=asymptomatic
      type: Number,
      required: true,
      enum: { values: [0, 1, 2, 3], message: "cp must be 0-3" },
    },
    trestbps: {
      // Resting blood pressure (mm Hg)
      type: Number,
      required: true,
      min: [50, "trestbps seems too low"],
      max: [300, "trestbps seems too high"],
    },
    chol: {
      // Serum cholesterol (mg/dl)
      type: Number,
      required: true,
      min: [50, "chol seems too low"],
      max: [700, "chol seems too high"],
    },
    fbs: {
      // Fasting blood sugar > 120 mg/dl: 1=true, 0=false
      type: Number,
      required: true,
      enum: { values: [0, 1], message: "fbs must be 0 or 1" },
    },
    restecg: {
      // Resting ECG: 0=normal, 1=ST-T abnormality, 2=LV hypertrophy
      type: Number,
      required: true,
      enum: { values: [0, 1, 2], message: "restecg must be 0, 1, or 2" },
    },
    thalach: {
      // Maximum heart rate achieved
      type: Number,
      required: true,
      min: [50, "thalach seems too low"],
      max: [250, "thalach seems too high"],
    },
    exang: {
      // Exercise-induced angina: 1=yes, 0=no
      type: Number,
      required: true,
      enum: { values: [0, 1], message: "exang must be 0 or 1" },
    },
    oldpeak: {
      // ST depression induced by exercise relative to rest
      type: Number,
      required: true,
      min: [0, "oldpeak cannot be negative"],
      max: [10, "oldpeak seems too high"],
    },
    slope: {
      // Slope of peak exercise ST segment: 0=upsloping, 1=flat, 2=downsloping
      type: Number,
      required: true,
      enum: { values: [0, 1, 2], message: "slope must be 0, 1, or 2" },
    },
    ca: {
      // Number of major vessels colored by fluoroscopy (0-3)
      type: Number,
      required: true,
      enum: { values: [0, 1, 2, 3], message: "ca must be 0-3" },
    },
    thal: {
      // Thalassemia: 1=normal, 2=fixed defect, 3=reversible defect
      type: Number,
      required: true,
      enum: { values: [1, 2, 3], message: "thal must be 1, 2, or 3" },
    },
  },
  { _id: false }
);

const predictionSchema = new mongoose.Schema(
  {
    user: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      default: null, // null = anonymous prediction
    },
    features: {
      type: featuresSchema,
      required: true,
    },
    result: {
      prediction: {
        type: Number, // 0 = No disease, 1 = Disease
        required: true,
        enum: [0, 1],
      },
      probability: {
        type: Number, // 0.0 – 1.0
        required: true,
        min: 0,
        max: 1,
      },
      riskLevel: {
        type: String,
        enum: ["low", "moderate", "high"],
        required: true,
      },
      modelUsed: {
        type: String,
        default: "ensemble",
      },
      featureImportance: {
        type: Map,
        of: Number,
        default: {},
      },
    },
    ipAddress: {
      type: String,
      default: null,
    },
    processingTimeMs: {
      type: Number,
      default: null,
    },
  },
  {
    timestamps: true,
    toJSON: { virtuals: true },
  }
);


predictionSchema.virtual("probabilityPercent").get(function () {
  return (this.result.probability * 100).toFixed(2);
});


predictionSchema.index({ user: 1, createdAt: -1 });
predictionSchema.index({ createdAt: -1 });

module.exports = mongoose.model("Prediction", predictionSchema);
