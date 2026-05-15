
// GET /api/v1/resources 

const router = require("express").Router();
const resourceController = require("../controllers/resource.controller");

router.get("/", resourceController.getResources);

module.exports = router;
