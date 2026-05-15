
const { AppError } = require("./errorHandler");
const { verifyAccessToken } = require("../services/auth.service");
const User = require("../models/User");

//  Protect: requires valid JWT
exports.protect = async (req, res, next) => {
  try {
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith("Bearer ")) {
      return next(new AppError("Authentication required. Please log in.", 401));
    }

    const token = authHeader.split(" ")[1];
    let decoded;
    try {
      decoded = verifyAccessToken(token);
    } catch (err) {
      if (err.name === "TokenExpiredError") {
        return next(new AppError("Session expired. Please log in again.", 401));
      }
      return next(new AppError("Invalid token.", 401));
    }

    const user = await User.findById(decoded.id);
    if (!user) return next(new AppError("User no longer exists.", 401));
    if (!user.isActive) return next(new AppError("Account is deactivated.", 403));

    req.user = { id: user._id.toString(), role: user.role, email: user.email };
    next();
  } catch (err) {
    next(err);
  }
};

//  Optional Auth: attach user if token present 
exports.optionalAuth = async (req, res, next) => {
  try {
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith("Bearer ")) return next();

    const token = authHeader.split(" ")[1];
    try {
      const decoded = verifyAccessToken(token);
      const user = await User.findById(decoded.id);
      if (user && user.isActive) {
        req.user = { id: user._id.toString(), role: user.role, email: user.email };
      }
    } catch {
      // Silently ignore — prediction continues anonymously
    }
    next();
  } catch (err) {
    next(err);
  }
};

//  Role Guard 
exports.restrictTo = (...roles) => {
  return (req, res, next) => {
    if (!roles.includes(req.user.role)) {
      return next(new AppError("You do not have permission to perform this action.", 403));
    }
    next();
  };
};
