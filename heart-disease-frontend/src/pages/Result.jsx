import {
  useLocation,
  Link
} from "react-router-dom";

import { useTranslation } from "react-i18next";

import Navbar from "../components/Navbar";
import Footer from "../components/Footer";

import "../styles/result.css";

function Result() {

  const location = useLocation();

  const { t } = useTranslation();

  const probability =
    location.state?.probability || 0;

  const getColor = () => {

    if(probability >= 70)
      return "#ff5f6d";

    if(probability >= 40)
      return "#ffc857";

    return "#3ddc97";
  };

  const getMessage = () => {

    if(probability >= 70)
      return t("highRisk");

    if(probability >= 40)
      return t("mediumRisk");

    return t("lowRisk");
  };

  return (
    <>

      <Navbar />

      <div className="result-page">

        <div className="result-card">

          <h2>
            {t("result")}
          </h2>

          <div
            className="percentage-box"

            style={{
              background:getColor(),
            }}
          >

            {probability}%

          </div>

          <h3>
            {getMessage()}
          </h3>

          <p>
            {t("resultNote")}
          </p>

          <div className="save-history-box">

            <h4>
              {t("saveHistory")}
            </h4>

            <p>
              {t("saveHistoryDesc")}
            </p>

            <div className="save-history-buttons">

              <Link to="/signin">
                {t("signIn")}
              </Link>

              <Link to="/signup">
                {t("signUp")}
              </Link>

            </div>

          </div>

        </div>

      </div>

      <Footer />

    </>
  );
}

export default Result;