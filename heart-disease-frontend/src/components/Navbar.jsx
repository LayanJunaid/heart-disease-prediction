import { useState, useRef, useEffect } from "react";

import { Link } from "react-router-dom";

import {
  FaUserCircle,
  FaChevronDown
} from "react-icons/fa";

import { useTranslation } from "react-i18next";

import UKFlag from "../assets/uk.jpg";

import SyriaFlag from "../assets/syria.jpg";

import TurkeyFlag from "../assets/turkey.jpg";

import "../styles/navbar.css";

function Navbar() {

  const { i18n, t } = useTranslation();

  const [openDropdown, setOpenDropdown] =
    useState(false);

  const dropdownRef = useRef(null);

  const user = JSON.parse(
    localStorage.getItem("user") || "null"
  );

  const languages = [

    {
      code:"en",
      label:"English",
      flag:UKFlag,
    },

    {
      code:"ar",
      label:"العربية",
      flag:SyriaFlag,
    },

    {
      code:"tr",
      label:"Türkçe",
      flag:TurkeyFlag,
    },
  ];

  const currentLanguage =
    languages.find(
      (lang) =>
        lang.code === i18n.language
    );

  useEffect(() => {

    const handleClickOutside = (event) => {

      if(
        dropdownRef.current &&
        !dropdownRef.current.contains(
          event.target
        )
      ){

        setOpenDropdown(false);
      }
    };

    document.addEventListener(
      "mousedown",
      handleClickOutside
    );

    return () => {

      document.removeEventListener(
        "mousedown",
        handleClickOutside
      );
    };

  }, []);

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

        <div
          className="language-dropdown"
          ref={dropdownRef}
        >

          <button
            className="language-btn"

            onClick={() =>
              setOpenDropdown(
                !openDropdown
              )
            }
          >

            <img
              src={currentLanguage?.flag}
              alt={currentLanguage?.label}
              className="flag-img"
            />

            <span>
              {currentLanguage?.label}
            </span>

            <FaChevronDown
              className={
                openDropdown
                ? "arrow rotate"
                : "arrow"
              }
            />

          </button>

          {
            openDropdown && (

              <div className="dropdown-menu">

                {languages.map((lang) => (

                  <button
                    key={lang.code}

                    className={
                      i18n.language === lang.code
                      ? "dropdown-item active-item"
                      : "dropdown-item"
                    }

                    onClick={() => {

                      i18n.changeLanguage(
                        lang.code
                      );

                      setOpenDropdown(false);
                    }}
                  >

                    <img
                      src={lang.flag}
                      alt={lang.label}
                      className="flag-img"
                    />

                    <span>
                      {lang.label}
                    </span>

                  </button>
                ))}

              </div>
            )
          }

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