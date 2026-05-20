import {
  FaGithub,
  FaCopyright
} from "react-icons/fa";

import { useTranslation } from "react-i18next";

import "../styles/footer.css";

function Footer() {

  const { t } = useTranslation();

  return (
    <footer className="footer">

      <div>

        <h3>
          {t("footerTitle")}
        </h3>

        <p>
          {t("footerAbout")}
        </p>

        <p className="footer-note">
          {t("disclaimer")}
        </p>

      </div>

      <div>

        <h4>
          {t("importantLinks")}
        </h4>

        <a
          href="https://github.com/LayanJunaid/heart-disease-prediction/tree/main"
          target="_blank"
          rel="noopener noreferrer"
        >
          {t("githubRepo")}
        </a>

        <a
          href="https://archive.ics.uci.edu/ml/datasets/Heart+Disease"
          target="_blank"
          rel="noopener noreferrer"
        >
          {t("dataset")}
        </a>

      </div>

      <div>

        <h4>
          {t("teamMembers")}
        </h4>

        <div className="member">

          <span>
            Sidra Ashram
          </span>

          <a
            href="https://github.com/SidraAhram"
            target="_blank"
            rel="noopener noreferrer"
          >
            <FaGithub />
          </a>

        </div>

        <div className="member">

          <span>
            Layan Junaid
          </span>

          <a
            href="https://github.com/LayanJunaid"
            target="_blank"
            rel="noopener noreferrer"
          >
            <FaGithub />
          </a>

        </div>

        <div className="member">

          <span>
            Ruha Kabbani
          </span>

          <a
            href="https://github.com/afakruha2003"
            target="_blank"
            rel="noopener noreferrer"
          >
            <FaGithub />
          </a>

        </div>

      </div>

      <div>

        <h4>
          {t("advisor")}
        </h4>

        <p>
          Shahaboddın DANESHVAR
        </p>

      </div>

      <div className="copyright">

        <FaCopyright />

        <span>
          2026 Senior Project
        </span>

      </div>

    </footer>
  );
}

export default Footer;