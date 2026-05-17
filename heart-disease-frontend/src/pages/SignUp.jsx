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

  const saveLoginData = (data) => {
    localStorage.setItem("accessToken", data.accessToken);
    localStorage.setItem("refreshToken", data.refreshToken);
    localStorage.setItem("user", JSON.stringify(data.user));
  };

  const handleChange = (e) => {
    const { name, value } = e.target;

    setFormData({
      ...formData,
      [name]: value,
    });
  };

  const handleSignup = async (e) => {

    e.preventDefault();

    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/v1/auth/register`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      const data = await response.json();

      if (!response.ok) {
        alert(data.message || "Signup failed");
        return;
      }

      saveLoginData(data);

      navigate("/profile");
      window.location.reload();

    } catch (error) {
      alert("Server connection error");
      console.error(error);
    }
  };

  const handleGoogleSuccess = async (credentialResponse) => {

    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/v1/auth/google`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          credential: credentialResponse.credential,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        alert(data.message || "Google signup failed");
        return;
      }

      saveLoginData(data);

      navigate("/profile");
      window.location.reload();

    } catch (error) {
      alert("Server connection error");
      console.error(error);
    }
  };

  return (
    <>

      <Navbar />

      <div className="auth-page">

        <div className="auth-card">

          <h2>
            {t("signUp")}
          </h2>

          <p className="auth-subtitle">
            {t("signUpDesc")}
          </p>

          
         <GoogleLogin
           onSuccess={handleGoogleSuccess}
           onError={() => {
           alert("Google login failed");
             }}
               locale="en"
               text="signin_with"
                size="large"
              width="650"
             />
       

          <div className="divider">
            OR
          </div>

          <form
            className="auth-form"
            onSubmit={handleSignup}
          >

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

            <button>
              {t("createAccount")}
            </button>

          </form>

        </div>

      </div>

      <Footer />

    </>
  );
}

export default SignUp;