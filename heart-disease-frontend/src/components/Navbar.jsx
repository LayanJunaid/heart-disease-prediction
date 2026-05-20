import { Link } from "react-router-dom";

import {
  FaUserCircle
} from "react-icons/fa";

import { useTranslation } from "react-i18next";

import "../styles/navbar.css";

function Navbar() {

  const { i18n, t } = useTranslation();

  const user = JSON.parse(
    localStorage.getItem("user") || "null"
  );

  const languages = [

    {
      code:"en",
      label:"English",
      flag:"🇬🇧",
    },

    {
      code:"ar",
      label:"العربية",
      flag:"🇸🇾",
    },

    {
      code:"tr",
      label:"Türkçe",
      flag:"🇹🇷",
    },
  ];

  return (

    <nav className="navbar">

      <div className="logo">

        <span>Heart</span>
        <span>Disease</span>
        <span>Prediction</span>

      </div>

      <div className="nav-links">

        <Link to="/">
          {t("home")}
        </Link>

        <Link to="/test">
          {t("test")}
        </Link>

        {!user && (
          <>

            <Link to="/signin">
              {t("signIn")}
            </Link>

            <Link to="/signup">
              {t("signUp")}
            </Link>

          </>
        )}

      </div>

      <div className="navbar-right">

        <div className="language-switcher">

          {languages.map((lang) => (

            <button
              key={lang.code}

              className={
                i18n.language === lang.code
                ? "lang-btn active-lang"
                : "lang-btn"
              }

              onClick={() =>
                i18n.changeLanguage(lang.code)
              }
            >

              <span className="flag">
                {lang.flag}
              </span>

              <span>
                {lang.label}
              </span>

            </button>
          ))}

        </div>

        <Link
          to="/profile"
          className="profile-icon"
        >

          <FaUserCircle />

        </Link>

      </div>

    </nav>
  );
}

export default Navbar;