import Navbar from "../components/Navbar";
import Footer from "../components/Footer";

import { useNavigate } from "react-router-dom";

import { useTranslation } from "react-i18next";

import {
  FaGoogle
} from "react-icons/fa";

import "../styles/auth.css";

function SignUp() {

  const { t } = useTranslation();

  const navigate = useNavigate();

  const handleSignup = (e) => {

    e.preventDefault();

    localStorage.setItem(
      "user",

      JSON.stringify({
        name:"John Doe",
        email:"johndoe@gmail.com",
      })
    );

    navigate("/profile");

    window.location.reload();
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

          <button
            className="google-btn"
            onClick={handleSignup}
          >

            <FaGoogle />

            {t("continueGoogle")}

          </button>

          <div className="divider">
            OR
          </div>

          <form
            className="auth-form"
            onSubmit={handleSignup}
          >

            <input
              type="text"
              placeholder={t("fullName")}
            />

            <input
              type="email"
              placeholder={t("email")}
            />

            <input
              type="password"
              placeholder={t("password")}
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