import { useState } from "react";
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { GoogleLogin } from "@react-oauth/google";
import "../styles/auth.css";

function SignUp() {
  const { t } = useTranslation();
  const navigate = useNavigate();

  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "",
  });

  const [successMsg, setSuccessMsg] = useState("");
  const [errorMsg, setErrorMsg] = useState("");

  const saveLoginData = (data) => {
    localStorage.setItem("accessToken", data.accessToken);
    localStorage.setItem("refreshToken", data.refreshToken);
    localStorage.setItem("user", JSON.stringify(data.user));
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleSignup = async (e) => {
    e.preventDefault();
    setErrorMsg("");
    setSuccessMsg("");

    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/v1/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      const data = await response.json();

      if (!response.ok) {
        setErrorMsg(data.message || t("signupFailed"));
        return;
      }

      setSuccessMsg(t("accountCreated"));
      setTimeout(() => navigate("/signin"), 2500);

    } catch (error) {
      setErrorMsg(t("serverError"));
      console.error(error);
    }
  };

  const handleGoogleSuccess = async (credentialResponse) => {
    setErrorMsg("");
    setSuccessMsg("");

    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/v1/auth/google`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ credential: credentialResponse.credential }),
      });

      const data = await response.json();

      if (!response.ok) {
        setErrorMsg(data.message || t("googleFailed"));
        return;
      }

      saveLoginData(data);
      navigate("/profile");
      window.location.reload();

    } catch (error) {
      setErrorMsg(t("serverError"));
      console.error(error);
    }
  };

  return (
    <>
      <Navbar />

      <div className="auth-page">
        <div className="auth-card">

          <h2>{t("signUp")}</h2>
          <p className="auth-subtitle">{t("signUpDesc")}</p>

          {successMsg && (
            <div className="auth-success">✓ {successMsg}</div>
          )}

          {errorMsg && (
            <div className="auth-error">✕ {errorMsg}</div>
          )}

          <GoogleLogin
            onSuccess={handleGoogleSuccess}
            onError={() => setErrorMsg(t("googleFailed"))}
            locale="en"
            text="signin_with"
            size="large"
            width="650"
          />

          <div className="divider">OR</div>

          <form className="auth-form" onSubmit={handleSignup}>
            <input
              type="text"
              name="name"
              placeholder={t("fullName")}
              value={formData.name}
              onChange={handleChange}
              required
            />

            <input
              type="email"
              name="email"
              placeholder={t("email")}
              value={formData.email}
              onChange={handleChange}
              required
            />

            <input
              type="password"
              name="password"
              placeholder={t("password")}
              value={formData.password}
              onChange={handleChange}
              required
            />

            <button type="submit">{t("createAccount")}</button>
          </form>

        </div>
      </div>

      <Footer />
    </>
  );
}

export default SignUp;