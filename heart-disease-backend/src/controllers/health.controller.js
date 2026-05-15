
const mongoose = require("mongoose");
const os = require("os");

exports.healthCheck = async (req, res) => {
  const dbStatus = mongoose.connection.readyState === 1 ? "connected" : "disconnected";

  res.json({
    success: true,
    status: "ok",
    timestamp: new Date().toISOString(),
    uptime: `${Math.floor(process.uptime())}s`,
    environment: process.env.NODE_ENV || "development",
    database: dbStatus,
    system: {
      platform: os.platform(),
      nodeVersion: process.version,
      memoryUsageMB: Math.round(process.memoryUsage().heapUsed / 1024 / 1024),
    },
    api: {
      version: "1.0.0",
      endpoints: {
        auth: "/api/v1/auth",
        predict: "/api/v1/predict",
        history: "/api/v1/history",
        resources: "/api/v1/resources",
        health: "/api/v1/health",
      },
    },
  });
};
