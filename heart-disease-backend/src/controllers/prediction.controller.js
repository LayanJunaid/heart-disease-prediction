
 //Ham UCI verisi (13 özellik) gelir → service OHE'ye çevirir → Python SVM tahmin yapar.
 

const predictionService = require("../services/prediction.service");
const Prediction = require("../models/Prediction");
const { AppError } = require("../middleware/errorHandler");
const { formatForDisplay } = require("../utils/dataMapper");
const logger = require("../utils/logger");

//POST /api/v1/predict 
exports.predict = async (req, res, next) => {
  const startTime = Date.now();
  try {
    // Ham UCI değerleri (validate.js zaten doğruladı)
    const rawFeatures = {
      age      : req.body.age,
      sex      : req.body.sex,
      cp       : req.body.cp,
      trestbps : req.body.trestbps,
      chol     : req.body.chol,
      fbs      : req.body.fbs,
      restecg  : req.body.restecg,
      thalach  : req.body.thalach,
      exang    : req.body.exang,
      oldpeak  : req.body.oldpeak,
      slope    : req.body.slope,
      ca       : req.body.ca,
      thal     : req.body.thal,
    };

    const userId    = req.user ? req.user.id : null;
    const ipAddress = req.ip || req.headers["x-forwarded-for"];

    // Service: OHE dönüşümü + Python SVM tahmini
    const predictionResult = await predictionService.runPrediction(rawFeatures);

    // DB'ye kaydet (ham feature'larla)
    const record = await Prediction.create({
      user    : userId,
      features: rawFeatures,
      result  : predictionResult,
      ipAddress,
      processingTimeMs: Date.now() - startTime,
    });

    logger.info(
      `[Prediction] id=${record._id} | pred=${predictionResult.prediction} | ` +
      `risk=${predictionResult.riskLevel} | model=${predictionResult.modelUsed} | ` +
      `user=${userId || "anonymous"}`
    );

    res.status(200).json({
      success: true,
      predictionId: record._id,
      result: {
        prediction      : predictionResult.prediction,
        probability     : predictionResult.probability,
        probabilityPercent: (predictionResult.probability * 100).toFixed(2),
        riskLevel       : predictionResult.riskLevel,
        modelUsed       : predictionResult.modelUsed,
        featureImportance: predictionResult.featureImportance,
        inputSummary    : formatForDisplay(
          require("../utils/dataMapper").transformToModelFeatures(rawFeatures)
        ),
      },
      processingTimeMs: Date.now() - startTime,
    });
  } catch (err) {
    next(err);
  }
};

// GET /api/v1/predict/:id 
exports.getPredictionById = async (req, res, next) => {
  try {
    const prediction = await Prediction.findById(req.params.id).populate(
      "user", "name email"
    );

    if (!prediction) return next(new AppError("Tahmin kaydı bulunamadı.", 404));

    if (
      prediction.user &&
      req.user &&
      prediction.user._id.toString() !== req.user.id &&
      req.user.role !== "admin"
    ) {
      return next(new AppError("Bu kaydı görme yetkiniz yok.", 403));
    }

    res.json({ success: true, prediction });
  } catch (err) {
    next(err);
  }
};

// GET /api/v1/predict/stats 
exports.getStats = async (req, res, next) => {
  try {
    const total     = await Prediction.countDocuments();
    const positives = await Prediction.countDocuments({ "result.prediction": 1 });

    const avgProb = await Prediction.aggregate([
      { $group: { _id: null, avg: { $avg: "$result.probability" } } },
    ]);

    const riskBreakdown = await Prediction.aggregate([
      { $group: { _id: "$result.riskLevel", count: { $sum: 1 } } },
    ]);

    res.json({
      success: true,
      stats: {
        total,
        positives,
        negatives   : total - positives,
        positiveRate: total > 0 ? ((positives / total) * 100).toFixed(1) : 0,
        avgProbability: avgProb[0]?.avg?.toFixed(3) || 0,
        riskBreakdown : riskBreakdown.reduce((acc, r) => {
          acc[r._id] = r.count;
          return acc;
        }, {}),
      },
    });
  } catch (err) {
    next(err);
  }
};
