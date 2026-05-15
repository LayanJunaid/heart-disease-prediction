/**
 * Akış:
 * 1. features (ham API verisi) → dataMapper → OHE feature vektörü
 * 2. OHE vektör → Python predict_bridge.py (stdin JSON)
 * 3. Python → SVM tahmin → stdout JSON
 * 4. Hata varsa → JS Logistic Regression fallback
 *
 * Python modelinin beklediği 14 feature (features.pkl):
 * age, ca_1, ca_2, ca_3, chol, cp_3.0, cp_4.0,
 * exang, oldpeak, sex, slope_2.0, thal_7, thalach, trestbps
 */

const { spawn } = require("child_process");
const path = require("path");
const fs   = require("fs");
const logger = require("../utils/logger");
const { transformToModelFeatures } = require("../utils/dataMapper");


const PROJECT_ROOT = process.cwd();
const ML_DIR        = path.join(PROJECT_ROOT, "ml");
const PYTHON_BRIDGE = path.join(ML_DIR, "predict_bridge.py");
const META_PATH     = path.join(ML_DIR, "meta.json");
const PYTHON_BIN = path.join(PROJECT_ROOT, "venv", "Scripts", "python.exe");


// meta.json'dan feature importance oku 
const loadFeatureImportance = () => {
  try {
    if (!fs.existsSync(META_PATH)) return null;
    const meta = JSON.parse(fs.readFileSync(META_PATH, "utf8"));
    return meta.feature_importance || null;
  } catch {
    return null;
  }
};

// Statik feature importance (lasso_coefficients + rf_importances'tan) 
// saved_data/lasso_coefficients.csv ve rf_importances.csv'den hesaplanan değerler.
// Gerçek feature isimlerini (OHE sonrası) kullanıyor.
const STATIC_FEATURE_IMPORTANCE = {
  thalach    : 0.178,
  oldpeak    : 0.142,
  "cp_4.0"   : 0.131,   
  thal_7     : 0.118,   
  ca_1       : 0.089,
  ca_2       : 0.071,
  ca_3       : 0.063,
  exang      : 0.058,
  "slope_2.0": 0.042,
  "cp_3.0"   : 0.038,
  age        : 0.031,
  sex        : 0.018,
  trestbps   : 0.012,
  chol       : 0.009,
};


const getRiskLevel = (probability) => {
  if (probability < 0.35) return "low";
  if (probability < 0.65) return "moderate";
  return "high";
};

//  JS Fallback: Lojistik Regresyon 
const LR_COEFFICIENTS = {
  intercept  : -1.2,
  age        :  0.025,
  sex        : -0.51,
  trestbps   :  0.004,
  chol       :  0.002,
  thalach    : -0.018,
  oldpeak    :  0.45,
  exang      :  0.75,
  ca_1       :  0.40,
  ca_2       :  0.65,
  ca_3       :  0.90,
  "cp_3.0"   :  0.55,
  "cp_4.0"   :  0.80,
  "slope_2.0":  0.30,
  thal_7     :  0.62,
};

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

const jsFallbackPredict = (oheFeatures) => {
  let logit = LR_COEFFICIENTS.intercept;
  for (const [key, coef] of Object.entries(LR_COEFFICIENTS)) {
    if (key === "intercept") continue;
    logit += coef * (Number(oheFeatures[key]) || 0);
  }
  const probability = sigmoid(logit);
  return { prediction: probability >= 0.5 ? 1 : 0, probability };
};

//  Python bridge çağrısı 
const callPythonBridge = (oheFeatures) => {
  return new Promise((resolve, reject) => {
    const proc = spawn(PYTHON_BIN, [PYTHON_BRIDGE], { timeout: 30000 });

    let stdout = "";
    let stderr = "";

    proc.stdin.write(JSON.stringify(oheFeatures) + "\n");
    proc.stdin.end();

    proc.stdout.on("data", (chunk) => (stdout += chunk.toString()));
    proc.stderr.on("data", (chunk) => (stderr += chunk.toString()));

    proc.on("close", (code) => {
      if (stderr.trim()) {
        logger.warn(`[Python Bridge] stderr: ${stderr.trim()}`);
      }

      // Exit 2 = model dosyaları yok → JS fallback
      if (code === 2) return reject(new Error("MODEL_NOT_FOUND"));

      if (code !== 0) {
        return reject(new Error(`Bridge exit=${code}: ${stderr.trim()}`));
      }

      const trimmed = stdout.trim();
      if (!trimmed) return reject(new Error("Bridge boş çıktı döndürdü"));

      try {
        const result = JSON.parse(trimmed);
        if (result.error) return reject(new Error(result.error));
        resolve(result);
      } catch {
        reject(new Error(`JSON parse hatası: ${trimmed.slice(0, 200)}`));
      }
    });

    proc.on("error", (err) =>
      reject(new Error(`Python spawn hatası: ${err.message}`))
    );
  });
};

//  Ana servis fonksiyonu 
exports.runPrediction = async (rawFeatures) => {
  // Ham veriyi → OHE feature vektörüne dönüştür
  const oheFeatures = transformToModelFeatures(rawFeatures);

  logger.info(
    `[Prediction] OHE features: ${JSON.stringify(oheFeatures)}`
  );

  let prediction, probability, modelUsed;

  try {
    // Python SVM modelini çağır
    const pyResult = await callPythonBridge(oheFeatures);

    prediction  = pyResult.prediction;
    probability = pyResult.probability;
    modelUsed   = pyResult.model_used || "SVC";

    logger.info(
      `[Prediction] SVM | pred=${prediction} | prob=${probability} | n_features=${pyResult.n_features}`
    );
  } catch (err) {
    // Fallback: JS Lojistik Regresyon
    const reason = err.message === "MODEL_NOT_FOUND"
      ? "model.pkl bulunamadı — train.py çalıştır"
      : err.message;

    logger.warn(`[Prediction] JS fallback. Neden: ${reason}`);

    const fallback = jsFallbackPredict(oheFeatures);
    prediction  = fallback.prediction;
    probability = fallback.probability;
    modelUsed   = "logistic_regression_fallback";
  }

  // Feature importance: meta.json varsa oradan, yoksa statik
  const rawFeatureImportance = loadFeatureImportance() || STATIC_FEATURE_IMPORTANCE;



  // MongoDB nesne anahtarlarında nokta (.) karakterini kabul etmiyor
  const safeFeatureImportance = {};
  if (rawFeatureImportance) {
    for (const [key, value] of Object.entries(rawFeatureImportance)) {
      const safeKey = key.replace(/\./g, '_'); 
      safeFeatureImportance[safeKey] = value;
    }
  }

  return {
    prediction,
    probability   : parseFloat(probability.toFixed(4)),
    riskLevel     : getRiskLevel(probability),
    modelUsed,
    featureImportance: safeFeatureImportance, // Temizlenmiş objeyi kullan
  };
};