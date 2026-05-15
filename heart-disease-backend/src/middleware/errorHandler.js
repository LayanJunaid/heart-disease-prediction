/**
 * Error Handler Middleware — middleware/errorHandler.js
 */

const logger = require("../utils/logger");

//  Custom App Error class 
class AppError extends Error {
  constructor(message, statusCode = 500) {
    super(message);
    this.statusCode = statusCode;
    this.isOperational = true;
    Error.captureStackTrace(this, this.constructor);
  }
}

//  404 Handler
const notFound = (req, res, next) => {
  next(new AppError(`Route '${req.originalUrl}' not found.`, 404));
};

//  Global Error Handler 
const errorHandler = (err, req, res, next) => {
  let { statusCode = 500, message } = err;

  // Mongoose validation error
  if (err.name === "ValidationError") {
    const messages = Object.values(err.errors).map((e) => e.message);
    statusCode = 422;
    message = messages.join(". ");
  }

  // Mongoose duplicate key error
  if (err.code === 11000) {
    const field = Object.keys(err.keyValue)[0];
    statusCode = 409;
    message = `${field.charAt(0).toUpperCase() + field.slice(1)} already exists.`;
  }

  // Mongoose CastError (invalid ObjectId)
  if (err.name === "CastError") {
    statusCode = 400;
    message = `Invalid value for field '${err.path}'.`;
  }

  // CORS error
  if (message && message.startsWith("CORS:")) {
    statusCode = 403;
  }

  if (statusCode >= 500) {
    logger.error(`[${req.method}] ${req.originalUrl} — ${message}`, {
      stack: err.stack,
    });
  }

  res.status(statusCode).json({
    success: false,
    message,
    ...(process.env.NODE_ENV === "development" && { stack: err.stack }),
  });
};

module.exports = { AppError, notFound, errorHandler };
