/**
 * Heart Disease Prediction Backend
 * Entry Point — server.js
 */

const app = require("./src/app");
const { connectDB } = require("./src/config/db");
const logger = require("./src/utils/logger");

const PORT = process.env.PORT || 5001;

// Connect to MongoDB then start server
connectDB()
  .then(() => {
    app.listen(PORT, () => {
      logger.info(` Server running on http://localhost:${PORT}`);
      logger.info(` Environment: ${process.env.NODE_ENV || "development"}`);
    });
  })
  .catch((err) => {
    logger.error("Failed to connect to database:", err.message);
    process.exit(1);
  });

// Handle unhandled promise rejections
process.on("unhandledRejection", (reason, promise) => {
  logger.error("Unhandled Rejection at:", promise, "reason:", reason);
  process.exit(1);
});

// Handle uncaught exceptions
process.on("uncaughtException", (err) => {
  logger.error("Uncaught Exception:", err.message);
  process.exit(1);
});
