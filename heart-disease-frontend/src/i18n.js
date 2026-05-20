import i18n from "i18next";
import { initReactI18next } from "react-i18next";

import en from "./translations/en";
import ar from "./translations/ar";
import tr from "./translations/tr";

i18n.use(initReactI18next).init({
  resources: {
    en: { translation: en },
    ar: { translation: ar },
    tr: { translation: tr },
  },
  lng: "en",
  fallbackLng: "en",
  interpolation: {
    escapeValue: false,
  },
});

export default i18n;