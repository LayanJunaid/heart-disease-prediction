
const jwt = require("jsonwebtoken");
const User = require("../models/User");
const jwtConfig = require("../config/jwt");
const { AppError } = require("../middleware/errorHandler");
const logger = require("../utils/logger");
const { OAuth2Client } = require("google-auth-library");

const googleClient = new OAuth2Client(
  process.env.GOOGLE_CLIENT_ID
);

//  Helpers 
const signAccessToken = (userId) =>
  jwt.sign({ id: userId }, jwtConfig.secret, { expiresIn: jwtConfig.accessExpiresIn });

const signRefreshToken = (userId) =>
  jwt.sign({ id: userId }, jwtConfig.secret + "_refresh", {
    expiresIn: jwtConfig.refreshExpiresIn,
  });

const sendTokens = (user, statusCode, res) => {
  const accessToken = signAccessToken(user._id);
  const refreshToken = signRefreshToken(user._id);

  // Store hashed refresh token in DB (simple: store plaintext for demo; hash in production)
  user.refreshToken = refreshToken;
  user.save({ validateBeforeSave: false });

  res.status(statusCode).json({
    success: true,
    accessToken,
    refreshToken,
    user: {
      id: user._id,
      name: user.name,
      email: user.email,
      role: user.role,
    },
  });
};

//  Register 
exports.register = async (req, res, next) => {
  try {
    const { name, email, password } = req.body;

    const existing = await User.findOne({ email });
    if (existing) return next(new AppError("Email already in use.", 409));

    const user = await User.create({ name, email, password });
    logger.info(`New user registered: ${email}`);
    sendTokens(user, 201, res);
  } catch (err) {
    next(err);
  }
};

//  Login 
exports.login = async (req, res, next) => {
  try {
    const { email, password } = req.body;

    const user = await User.findOne({ email }).select("+password");
    if (!user || !(await user.comparePassword(password))) {
      return next(new AppError("Invalid email or password.", 401));
    }

    if (!user.isActive) return next(new AppError("Account is deactivated.", 403));

    user.lastLogin = new Date();
    await user.save({ validateBeforeSave: false });

    logger.info(`User logged in: ${email}`);
    sendTokens(user, 200, res);
  } catch (err) {
    next(err);
  }
};

//  Refresh Token 
exports.refreshToken = async (req, res, next) => {
  try {
    const { refreshToken } = req.body;
    if (!refreshToken) return next(new AppError("Refresh token required.", 400));

    let decoded;
    try {
      decoded = jwt.verify(refreshToken, jwtConfig.secret + "_refresh");
    } catch {
      return next(new AppError("Invalid or expired refresh token.", 401));
    }

    const user = await User.findById(decoded.id).select("+refreshToken");
    if (!user || user.refreshToken !== refreshToken) {
      return next(new AppError("Refresh token mismatch.", 401));
    }

    const newAccessToken = signAccessToken(user._id);
    res.json({ success: true, accessToken: newAccessToken });
  } catch (err) {
    next(err);
  }
};

//  Logout 
exports.logout = async (req, res, next) => {
  try {
    await User.findByIdAndUpdate(req.user.id, { refreshToken: null });
    res.json({ success: true, message: "Logged out successfully." });
  } catch (err) {
    next(err);
  }
};

//  Get Current User 
exports.getMe = async (req, res, next) => {
  try {
    const user = await User.findById(req.user.id);
    if (!user) return next(new AppError("User not found.", 404));
    res.json({ success: true, user });
  } catch (err) {
    next(err);
  }
};

//  Update Password 
exports.updatePassword = async (req, res, next) => {
  try {
    const { currentPassword, newPassword } = req.body;
    const user = await User.findById(req.user.id).select("+password");

    if (!(await user.comparePassword(currentPassword))) {
      return next(new AppError("Current password is incorrect.", 401));
    }

    user.password = newPassword;
    await user.save();

    sendTokens(user, 200, res);
  } catch (err) {
    next(err);
  }
};
// Google Login / Signup
exports.googleLogin = async (req, res, next) => {
  try {
    const { credential } = req.body;

    if (!credential) {
      return next(new AppError("Google credential is required.", 400));
    }

    const ticket = await googleClient.verifyIdToken({
      idToken: credential,
      audience: process.env.GOOGLE_CLIENT_ID,
    });

    const payload = ticket.getPayload();

    const email = payload.email;
    const name = payload.name;
    const picture = payload.picture;
    const googleId = payload.sub;

    if (!email) {
      return next(new AppError("Google account email not found.", 400));
    }

    let user = await User.findOne({ email });

    if (!user) {
      user = await User.create({
        name,
        email,
        password: googleId + process.env.JWT_SECRET.slice(0, 10),
        authProvider: "google",
        googleId,
        profileImage: picture,
      });
    }

    if (!user.isActive) {
      return next(new AppError("Account is deactivated.", 403));
    }

    user.lastLogin = new Date();

    if (!user.googleId) {
      user.googleId = googleId;
    }

    if (!user.profileImage && picture) {
      user.profileImage = picture;
    }

    await user.save({ validateBeforeSave: false });

    logger.info(`User logged in with Google: ${email}`);

    sendTokens(user, 200, res);
  } catch (err) {
    next(err);
  }
};