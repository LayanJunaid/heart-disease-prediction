/**
 * GET    /api/v1/history             
 * GET    /api/v1/history/summary     
 * GET    /api/v1/history/:id        
 * DELETE /api/v1/history             
 * DELETE /api/v1/history/:id         
 */

const router = require("express").Router();
const historyController = require("../controllers/history.controller");
const { protect } = require("../middleware/auth");
const { validate, objectIdParam } = require("../middleware/validate");


router.use(protect);

router.get("/summary", historyController.getMySummary);
router.get("/", historyController.getMyHistory);
router.get("/:id", [...objectIdParam("id"), validate], historyController.getHistoryItem);
router.delete("/", historyController.clearHistory);
router.delete("/:id", [...objectIdParam("id"), validate], historyController.deleteHistoryItem);

module.exports = router;
