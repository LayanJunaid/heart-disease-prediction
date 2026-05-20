/**
 * MongoDB Connection — config/db.js
 */

const mongoose = require("mongoose");
const logger = require("../utils/logger");

const connectDB = async () => {
  const uri = process.env.MONGODB_URI ;
             //    MongoDB Atlas      ||        Local
  try {
    const conn = await mongoose.connect(uri, {
      serverSelectionTimeoutMS: 5000,
      socketTimeoutMS: 45000,
    });

    logger.info(`MongoDB connected: ${conn.connection.host}`);

    mongoose.connection.on("disconnected", () => {
      logger.warn("MongoDB disconnected. Attempting reconnect...");
    });

    mongoose.connection.on("reconnected", () => {
      logger.info("MongoDB reconnected.");
    });

    mongoose.connection.on("error", (err) => {
      logger.error("MongoDB connection error:", err.message);
    });
  }catch (error) {
    console.error("FULL MongoDB ERROR:", error);
    logger.error(`MongoDB connection failed: ${error.message}`);
    throw error;
  }
};

const disconnectDB = async () => {
  await mongoose.connection.close();
  logger.info("MongoDB connection closed.");
};

module.exports = { connectDB, disconnectDB };
