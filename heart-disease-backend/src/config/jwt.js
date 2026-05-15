/**
 * JWT Configuration — config/jwt.js
 */

module.exports = {
  secret: process.env.JWT_SECRET || "change_this_super_secret_key_in_production",
  accessExpiresIn: process.env.JWT_ACCESS_EXPIRES || "1h",
  refreshExpiresIn: process.env.JWT_REFRESH_EXPIRES || "7d",
};
