const transformToModelFeatures = (raw) => {
  const cp    = Number(raw.cp);
  const ca    = Number(raw.ca);
  const slope = Number(raw.slope);
  const thal  = Number(raw.thal);

  return {

    age      : Number(raw.age),
    sex      : Number(raw.sex),
    trestbps : Number(raw.trestbps),
    chol     : Number(raw.chol),
    thalach  : Number(raw.thalach),
    oldpeak  : Number(raw.oldpeak),
    exang    : Number(raw.exang),

   
    ca_1 : ca === 1 ? 1 : 0,
    ca_2 : ca === 2 ? 1 : 0,
    ca_3 : ca === 3 ? 1 : 0,


    "cp_3.0" : cp === 2 ? 1 : 0,   // API cp=2 → orijinal cp=3
    "cp_4.0" : cp === 3 ? 1 : 0,   // API cp=3 → orijinal cp=4 (asymptomatic)
    "slope_2.0" : slope === 1 ? 1 : 0,   // API slope=1 (flat) → orijinal slope=2
    thal_7 : thal === 3 ? 1 : 0,
  };
};

const formatForDisplay = (modelFeatures) => {
  const LABELS = {
    age         : "Yaş",
    sex         : "Cinsiyet",
    trestbps    : "Dinlenme Kan Basıncı",
    chol        : "Kolesterol",
    thalach     : "Maks. Kalp Atış Hızı",
    oldpeak     : "ST Depresyonu",
    exang       : "Egzersiz Anginası",
    ca_1        : "Damar Sayısı = 1",
    ca_2        : "Damar Sayısı = 2",
    ca_3        : "Damar Sayısı = 3",
    "cp_3.0"    : "Göğüs Ağrısı: Non-Anginal",
    "cp_4.0"    : "Göğüs Ağrısı: Asemptomatik",
    "slope_2.0" : "ST Eğimi: Düz",
    thal_7      : "Talasemi: Geri Dönüşümlü Defekt",
  };

  return Object.entries(modelFeatures).map(([key, value]) => ({
    key,
    label : LABELS[key] || key,
    value,
  }));
};

module.exports = { transformToModelFeatures, formatForDisplay };
