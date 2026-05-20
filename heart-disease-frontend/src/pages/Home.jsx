import { useState } from "react";

import Navbar from "../components/Navbar";
import Footer from "../components/Footer";

import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";

import Image1 from "../assets/Image1.png";
import Image2 from "../assets/Image2.png";

import "../styles/home.css";

function Home() {

  const navigate = useNavigate();

  const { t } = useTranslation();

  const [expanded, setExpanded] = useState(false);

  return (
    <>

      <Navbar />

      <div className="home-container">

        <div className="circle1"></div>
        <div className="circle2"></div>
        <div className="circle3"></div>

        <section className="hero-section">

          <div className="hero-left">

            <h1>
              {t("homeTitle")}
            </h1>

            <p
              className={
                expanded
                ? "project-description expanded"
                : "project-description"
              }
            >
              {t("homeDesc")}
            </p>

            <button
              className="view-more-btn"
              onClick={() =>
                setExpanded(!expanded)
              }
            >

              {
                expanded
                ? t("viewLess")
                : t("viewMore")
              }

            </button>

            <button
              className="main-btn"
              onClick={() => navigate("/test")}
            >
              {t("startTest")}
            </button>

          </div>

          <div className="hero-right">

            <div className="hero-circle"></div>

             <img
              src={Image1}
              alt=""
            /> 

          </div>

        </section>

        <section className="resources-section">

          <h2>
            {t("resources")}
          </h2>

          <div className="resource-cards">

            <a
              href="https://github.com/LayanJunaid/heart-disease-prediction/tree/main"
              target="_blank"
            >
              GitHub Repository
            </a>

            <a
              href="https://archive.ics.uci.edu/ml/datasets/Heart+Disease"
              target="_blank"
            >
              UCI Heart Disease Dataset
            </a>

          </div>

        </section>

        <img
          src={Image2}
          alt=""
          className="floating-image"
        />

      </div>

      <Footer />

    </>
  );
}

export default Home;