
 // GET /api/v1/health → Server + DB health status

const router = require("express").Router();
const healthController = require("../controllers/health.controller");

router.get("/", healthController.healthCheck);

module.exports = router;
