import { useState } from "react";

import { useNavigate } from "react-router-dom";

import { useTranslation } from "react-i18next";

import Navbar from "../components/Navbar";
import Footer from "../components/Footer";

import Image3 from "../assets/Image3.png";

import "../styles/test.css";

function Test() {

  const navigate = useNavigate();

  const { t } = useTranslation();

  const [formData, setFormData] = useState({

    age:"",
    sex:"",
    cp:"",
    trestbps:"",
    chol:"",
    fbs:"",
    restecg:"",
    thalach:"",
    exang:"",
    oldpeak:"",
    slope:"",
    ca:"",
    thal:"",
  });

  const features = [

    {
      name:"age",
      label:t("age"),
      description:t("ageDesc"),
      min:1,
      max:120
    },

    {
      name:"sex",
      label:t("sex"),
      description:t("sexDescription"),
      min:0,
      max:1
    },

    {
      name:"cp",
      label:t("cp"),
      description:t("cpDesc"),
      min:0,
      max:3
    },

    {
      name:"trestbps",
      label:t("trestbps"),
      description:t("trestbpsDesc"),
      min:50,
      max:250
    },

    {
      name:"chol",
      label:t("chol"),
      description:t("cholDesc"),
      min:50,
      max:700
    },

    {
      name:"fbs",
      label:t("fbs"),
      description:t("fbsDesc"),
      min:0,
      max:1
    },

    {
      name:"restecg",
      label:t("restecg"),
      description:t("restecgDesc"),
      min:0,
      max:2
    },

    {
      name:"thalach",
      label:t("thalach"),
      description:t("thalachDesc"),
      min:50,
      max:250
    },

    {
      name:"exang",
      label:t("exang"),
      description:t("exangDesc"),
      min:0,
      max:1
    },

    {
      name:"oldpeak",
      label:t("oldpeak"),
      description:t("oldpeakDesc"),
      min:0,
      max:10
    },

    {
      name:"slope",
      label:t("slope"),
      description:t("slopeDesc"),
      min:0,
      max:2
    },

    {
      name:"ca",
      label:t("ca"),
      description:t("caDesc"),
      min:0,
      max:4
    },

    {
      name:"thal",
      label:t("thal"),
      description:t("thalDesc"),
      min:0,
      max:3
    },
  ];

  const handleChange = (e) => {

    const { name, value } = e.target;

    setFormData({
      ...formData,
      [name]:value,
    });
  };

  const handleSubmit = (e) => {

    e.preventDefault();

    for(let feature of features){

      const value = Number(
        formData[feature.name]
      );

      if(
        isNaN(value) ||
        value < feature.min ||
        value > feature.max
      ){

        alert(
          `${t("invalid")} ${feature.label}`
        );

        return;
      }
    }

    const fakeProbability =
      Math.floor(Math.random() * 100);

    navigate("/result", {

      state:{
        probability:fakeProbability,
      },
    });
  };

  return (
    <>

      <Navbar />

      <div className="test-page">

        <div className="test-circle1"></div>
        <div className="test-circle2"></div>

        <img
          src={Image3}
          alt=""
          className="test-top-image"
        />

        <h2>
          {t("testTitle")}
        </h2>

        <form
          className="features-grid"
          onSubmit={handleSubmit}
        >

          {features.map((feature) => (

            <div
              className="input-group"
              key={feature.name}
            >

              <label>
                {feature.label}
              </label>

              <small className="feature-description">
                {feature.description}
              </small>

              <input
                type="number"

                min={feature.min}

                max={feature.max}

                step="any"

                required

                name={feature.name}

                value={formData[feature.name]}

                onChange={handleChange}
              />

            </div>
          ))}

          <button className="submit-btn">
            {t("submit")}
          </button>

        </form>

      </div>

      <Footer />

    </>
  );
}

export default Test;