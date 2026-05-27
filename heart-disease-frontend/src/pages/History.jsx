import { useState, useEffect } from "react";
import { useTranslation } from "react-i18next";
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";
import "../styles/history.css";

function History() {
  const { t } = useTranslation();
  const [history, setHistory] = useState([]);

  useEffect(() => {
    const stored = JSON.parse(localStorage.getItem("heartHistory") || "[]");
    setHistory(stored);
  }, []);

  const formatDate = (isoString) => {
    const date = new Date(isoString);
    return date.toLocaleDateString("tr-TR", {
      day: "2-digit",
      month: "2-digit",
      year: "numeric",
    });
  };

  const formatTime = (isoString) => {
    const date = new Date(isoString);
    return date.toLocaleTimeString("tr-TR", {
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  return (
    <>
      <Navbar />
      <div className="history-page">
        <h2>{t("history")}</h2>

        {history.length === 0 ? (
          <p className="no-history">{t("noHistory")}</p>
        ) : (
          history.map((entry, index) => (
            <div className="history-card" key={index}>
              <h3>{entry.probability}%</h3>
              <p>{entry.message}</p>
              <div className="history-date">
                <span>📅 {formatDate(entry.date)}</span>
                <span>🕐 {formatTime(entry.date)}</span>
              </div>
            </div>
          ))
        )}
      </div>
      <Footer />
    </>
  );
}

export default History;