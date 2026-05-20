import Navbar from "../components/Navbar";
import Footer from "../components/Footer";

import { useTranslation } from "react-i18next";

import "../styles/auth.css";

function EditProfile() {

  const { t } = useTranslation();

  return (
    <>

      <Navbar />

      <div className="auth-page">

        <div className="auth-card">

          <h2>
            {t("editProfile")}
          </h2>

          <form className="auth-form">

            <input
              type="text"
              placeholder={t("fullName")}
            />

            <input
              type="email"
              placeholder={t("email")}
            />

            <button>
              {t("saveChanges")}
            </button>

          </form>

        </div>

      </div>

      <Footer />

    </>
  );
}

export default EditProfile;