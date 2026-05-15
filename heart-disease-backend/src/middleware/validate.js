/**
 * Validation Middleware — middleware/validate.js
 * Uses express-validator for request body validation.
 */

const { validationResult, body, param } = require("express-validator");
const { AppError } = require("./errorHandler");

//  Run validation result 
const validate = (req, res, next) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    const messages = errors.array().map((e) => `${e.path}: ${e.msg}`);
    return next(new AppError(messages.join(". "), 422));
  }
  next();
};

//  Auth validators 
const registerRules = [
  body("name").trim().notEmpty().withMessage("Name is required").isLength({ min: 2, max: 60 }),
  body("email").isEmail().withMessage("Valid email is required").normalizeEmail(),
  body("password").isLength({ min: 6 }).withMessage("Password must be at least 6 characters"),
];

const loginRules = [
  body("email").isEmail().withMessage("Valid email is required").normalizeEmail(),
  body("password").notEmpty().withMessage("Password is required"),
];

const updatePasswordRules = [
  body("currentPassword").notEmpty().withMessage("Current password is required"),
  body("newPassword").isLength({ min: 6 }).withMessage("New password must be at least 6 characters"),
];

//  Prediction validators 
const predictionRules = [
  body("age")
    .isFloat({ min: 1, max: 120 })
    .withMessage("age must be between 1 and 120"),
  body("sex")
    .isIn([0, 1])
    .withMessage("sex must be 0 (female) or 1 (male)"),
  body("cp")
    .isIn([0, 1, 2, 3])
    .withMessage("cp (chest pain type) must be 0, 1, 2, or 3"),
  body("trestbps")
    .isFloat({ min: 50, max: 300 })
    .withMessage("trestbps (resting BP) must be between 50 and 300 mm Hg"),
  body("chol")
    .isFloat({ min: 50, max: 700 })
    .withMessage("chol (cholesterol) must be between 50 and 700 mg/dl"),
  body("fbs")
    .isIn([0, 1])
    .withMessage("fbs must be 0 or 1"),
  body("restecg")
    .isIn([0, 1, 2])
    .withMessage("restecg must be 0, 1, or 2"),
  body("thalach")
    .isFloat({ min: 50, max: 250 })
    .withMessage("thalach (max heart rate) must be between 50 and 250"),
  body("exang")
    .isIn([0, 1])
    .withMessage("exang must be 0 or 1"),
  body("oldpeak")
    .isFloat({ min: 0, max: 10 })
    .withMessage("oldpeak must be between 0 and 10"),
  body("slope")
    .isIn([0, 1, 2])
    .withMessage("slope must be 0, 1, or 2"),
  body("ca")
    .isIn([0, 1, 2, 3])
    .withMessage("ca must be 0, 1, 2, or 3"),
  body("thal")
    .isIn([1, 2, 3])
    .withMessage("thal must be 1, 2, or 3"),
];

//  ObjectId param validator 
const objectIdParam = (paramName) => [
  param(paramName).isMongoId().withMessage(`Invalid ${paramName} ID format`),
];

module.exports = {
  validate,
  registerRules,
  loginRules,
  updatePasswordRules,
  predictionRules,
  objectIdParam,
};
