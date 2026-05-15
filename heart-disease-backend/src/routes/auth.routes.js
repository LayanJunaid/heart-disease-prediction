
const router = require("express").Router();
const authController = require("../controllers/auth.controller");
const { protect } = require("../middleware/auth");
const {
  validate,
  registerRules,
  loginRules,
  updatePasswordRules,
} = require("../middleware/validate");

router.post("/register", registerRules, validate, authController.register);
router.post("/login", loginRules, validate, authController.login);
router.post("/refresh", authController.refreshToken);
router.post("/logout", protect, authController.logout);
router.get("/me", protect, authController.getMe);
router.put("/password", protect, updatePasswordRules, validate, authController.updatePassword);

module.exports = router;
