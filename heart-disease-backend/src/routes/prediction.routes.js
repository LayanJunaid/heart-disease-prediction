/**
 * POST  /api/v1/predict            
 * GET   /api/v1/predict/stats       
 * GET   /api/v1/predict/:id         
 */

const router = require("express").Router();
const predictionController = require("../controllers/prediction.controller");
const { protect, optionalAuth, restrictTo } = require("../middleware/auth");
const { validate, predictionRules, objectIdParam } = require("../middleware/validate");

// Prediction endpoint — works for both anonymous and authenticated users
router.post("/", optionalAuth, predictionRules, validate, predictionController.predict);

// Admin stats
router.get("/stats", protect, restrictTo("admin"), predictionController.getStats);

// Single prediction lookup
router.get("/:id", [...objectIdParam("id"), validate], optionalAuth, predictionController.getPredictionById);

module.exports = router;
