
 // Manages prediction history for authenticated users.

const Prediction = require("../models/Prediction");
const { AppError } = require("../middleware/errorHandler");

//  GET /api/v1/history 
exports.getMyHistory = async (req, res, next) => {
  try {
    const page = Math.max(1, parseInt(req.query.page) || 1);
    const limit = Math.min(50, Math.max(1, parseInt(req.query.limit) || 10));
    const skip = (page - 1) * limit;

    const filter = { user: req.user.id };

    // Optional filter by risk level
    if (req.query.riskLevel) {
      filter["result.riskLevel"] = req.query.riskLevel;
    }

    // Optional filter by prediction outcome
    if (req.query.prediction !== undefined) {
      filter["result.prediction"] = parseInt(req.query.prediction);
    }

    const [predictions, total] = await Promise.all([
      Prediction.find(filter)
        .sort({ createdAt: -1 })
        .skip(skip)
        .limit(limit)
        .select("-__v"),
      Prediction.countDocuments(filter),
    ]);

    res.set("X-Total-Count", total);
    res.json({
      success: true,
      data: predictions,
      pagination: {
        page,
        limit,
        total,
        pages: Math.ceil(total / limit),
      },
    });
  } catch (err) {
    next(err);
  }
};

// GET /api/v1/history/:id 
exports.getHistoryItem = async (req, res, next) => {
  try {
    const prediction = await Prediction.findOne({
      _id: req.params.id,
      user: req.user.id,
    });

    if (!prediction) return next(new AppError("Prediction record not found.", 404));

    res.json({ success: true, data: prediction });
  } catch (err) {
    next(err);
  }
};

// DELETE /api/v1/history/:id 
exports.deleteHistoryItem = async (req, res, next) => {
  try {
    const prediction = await Prediction.findOneAndDelete({
      _id: req.params.id,
      user: req.user.id,
    });

    if (!prediction) return next(new AppError("Prediction record not found.", 404));

    res.json({ success: true, message: "Prediction record deleted." });
  } catch (err) {
    next(err);
  }
};

// DELETE /api/v1/history
exports.clearHistory = async (req, res, next) => {
  try {
    const result = await Prediction.deleteMany({ user: req.user.id });
    res.json({
      success: true,
      message: `Deleted ${result.deletedCount} prediction record(s).`,
    });
  } catch (err) {
    next(err);
  }
};

//  GET /api/v1/history/summary
exports.getMySummary = async (req, res, next) => {
  try {
    const userId = req.user.id;

    const [total, positive, byRisk, avgProb] = await Promise.all([
      Prediction.countDocuments({ user: userId }),
      Prediction.countDocuments({ user: userId, "result.prediction": 1 }),
      Prediction.aggregate([
        { $match: { user: require("mongoose").Types.ObjectId.createFromHexString(userId) } },
        { $group: { _id: "$result.riskLevel", count: { $sum: 1 } } },
      ]),
      Prediction.aggregate([
        { $match: { user: require("mongoose").Types.ObjectId.createFromHexString(userId) } },
        { $group: { _id: null, avg: { $avg: "$result.probability" } } },
      ]),
    ]);

    res.json({
      success: true,
      summary: {
        total,
        positive,
        negative: total - positive,
        riskBreakdown: byRisk.reduce((acc, r) => ({ ...acc, [r._id]: r.count }), {}),
        avgProbability: avgProb[0]?.avg?.toFixed(3) || 0,
      },
    });
  } catch (err) {
    next(err);
  }
};
