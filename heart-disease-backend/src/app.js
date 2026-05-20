/**
 * Express Application Setup — app.js
 */

require("dotenv").config();

const express = require("express");
const cors = require("cors");
const helmet = require("helmet");
const morgan = require("morgan");
const rateLimit = require("express-rate-limit");

const corsOptions = require("./config/cors");
const { errorHandler, notFound } = require("./middleware/errorHandler");
const logger = require("./utils/logger");

// Routes

const authRoutes = require("./routes/auth.routes");
const predictionRoutes = require("./routes/prediction.routes");
const historyRoutes = require("./routes/history.routes");
const resourceRoutes = require("./routes/resource.routes");
const healthRoutes = require("./routes/health.routes");

const app = express();

// Security Middleware 
app.use(helmet());
app.use(cors(corsOptions));

// Rate Limiting
const globalLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 200,
  standardHeaders: true,
  legacyHeaders: false,
  message: { success: false, message: "Too many requests, please try again later." },
});

const predictionLimiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 10,
  message: { success: false, message: "Too many prediction requests. Please slow down." },
});

app.use(globalLimiter);

// Body Parsing 
app.use(express.json({ limit: "10kb" }));
app.use(express.urlencoded({ extended: true, limit: "10kb" }));

//  HTTP Request Logging 
if (process.env.NODE_ENV !== "test") {
  app.use(
    morgan("combined", {
      stream: { write: (message) => logger.http(message.trim()) },
    })
  );
}

// API Routes
app.use("/api/v1/auth", authRoutes); 
app.use("/api/v1/predict", predictionLimiter, predictionRoutes);
app.use("/api/v1/history", historyRoutes);
app.use("/api/v1/resources", resourceRoutes);
app.use("/api/v1/health", healthRoutes);

//  Root Endpoint 
app.get("/", (req, res) => {
  res.json({
    success: true,
    message: "Heart Disease Prediction API",
    version: "1.0.0",
    docs: "/api/v1/health",
  });
});

//  Error Handling
app.use(notFound);
app.use(errorHandler);

module.exports = app;
