import Navbar from "../components/Navbar";
import Footer from "../components/Footer";

import {
  Navigate,
  Link
} from "react-router-dom";

import { useTranslation } from "react-i18next";

import "../styles/profile.css";

function Profile() {

  const { t } = useTranslation();

  const user = JSON.parse(
    localStorage.getItem("user") || "null"
  );

  if(!user){

    return <Navigate to="/signin" />;
  }

  const handleLogout = () => {

    localStorage.removeItem("user");

    window.location.href = "/";
  };

  return (
    <>

      <Navbar />

      <div className="profile-page">

        <div className="profile-card">

          <h2>
            {t("profile")}
          </h2>

          <p>
            <strong>
              {t("fullName")}:
            </strong>

            {user.name}
          </p>

          <p>
            <strong>
              {t("email")}:
            </strong>

            {user.email}
          </p>

          <div className="profile-buttons">

            <Link to="/history">
              {t("history")}
            </Link>

            <Link to="/edit-profile">
              {t("editProfile")}
            </Link>

            <button
              className="logout-btn"
              onClick={handleLogout}
            >
              {t("logout")}
            </button>

          </div>

        </div>

      </div>

      <Footer />

    </>
  );
}

export default Profile;