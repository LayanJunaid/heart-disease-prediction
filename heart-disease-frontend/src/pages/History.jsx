import Navbar from "../components/Navbar";
import Footer from "../components/Footer";

import { useTranslation } from "react-i18next";

import "../styles/history.css";

function History() {

  const { t } = useTranslation();

  return (
    <>

      <Navbar />

      <div className="history-page">

        <h2>
          {t("history")}
        </h2>

        <div className="history-card">

          <h3>
            78%
          </h3>

          <p>
            {t("highRisk")}
          </p>

        </div>

        <div className="history-card">

          <h3>
            32%
          </h3>

          <p>
            {t("lowRisk")}
          </p>

        </div>

      </div>

      <Footer />

    </>
  );
}

export default History;