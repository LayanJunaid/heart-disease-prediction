
//  Token verification helper used by middleware.


const jwt = require("jsonwebtoken");
const jwtConfig = require("../config/jwt");

exports.verifyAccessToken = (token) => {
  return jwt.verify(token, jwtConfig.secret);
};
