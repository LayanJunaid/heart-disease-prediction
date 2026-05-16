
import sys
import json
import os
import pickle
import numpy as np
import pandas as pd  

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(SCRIPT_DIR, "model.pkl")
SCALER_PATH   = os.path.join(SCRIPT_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(SCRIPT_DIR, "features.pkl")
META_PATH     = os.path.join(SCRIPT_DIR, "meta.json")


def load_artifacts():

    for path, label in [
        (MODEL_PATH,    "model.pkl"),
        (SCALER_PATH,   "scaler.pkl"),
        (FEATURES_PATH, "features.pkl"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label} not found. Please run train.py before : "
                f"python3 ml/train.py /path/to/saved_data/"
            )

    with open(MODEL_PATH,    "rb") as f: model    = pickle.load(f)
    with open(SCALER_PATH,   "rb") as f: scaler   = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f: features = pickle.load(f)

    return model, scaler, features

# It checks for missing features and ensures that all values can be converted to floats, which is necessary for the model's prediction.
def build_feature_dataframe(data: dict, features: list) -> pd.DataFrame:

    row = []
    missing = []
    for feat in features:
        if feat not in data:
            missing.append(feat)
        else:
            try:
                row.append(float(data[feat]))
            except (TypeError, ValueError):
                raise ValueError(
                    f"Feature '{feat}' cannot be converted to float: {data[feat]!r}"
                )

    if missing:
        raise KeyError(
            f"Missing features: {missing}. "
            f"Expected feature list: {features}"
        )

   
    return pd.DataFrame([row], columns=features)


def predict(data: dict) -> dict:
    model, scaler, features = load_artifacts()

    X_raw_df = build_feature_dataframe(data, features)
    
    X_scaled_arr = scaler.transform(X_raw_df)
    X_scaled_df  = pd.DataFrame(X_scaled_arr, columns=features)

    prediction = int(model.predict(X_scaled_df)[0])

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(X_scaled_df)[0][1])
    elif hasattr(model, "decision_function"):
        # Yedek: sigmoid(decision_function)
        score = model.decision_function(X_scaled_df)[0]
        probability = float(1.0 / (1.0 + np.exp(-score)))
    else:
        probability = float(prediction)

    return {
        "prediction" : prediction,
        "probability": round(probability, 4),
        "model_used" : type(model).__name__,
        "n_features" : len(features),
        "feature_names": features,
    }


def main():
    raw = sys.stdin.read().strip()

    if not raw:
        out = {"error": "stdin boş — Node.js hiç veri göndermedi"}
        print(json.dumps(out))
        sys.exit(1)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        out = {"error": f"JSON parse hatası: {e} | Gelen: {raw[:200]}"}
        print(json.dumps(out))
        sys.exit(1)


    try:
        result = predict(data)
        print(json.dumps(result))

    except FileNotFoundError as e:
        print(json.dumps({"error": str(e)}))
        sys.stderr.write(f"[predict_bridge] FileNotFoundError: {e}\n")
        sys.exit(2)

    except KeyError as e:
        print(json.dumps({"error": f"Missing feature: {e}"}))
        sys.stderr.write(f"[predict_bridge] KeyError: {e}\n")
        sys.exit(1)

    except Exception as e:
        print(json.dumps({"error": f"Unexpected error: {type(e).__name__}: {e}"}))
        sys.stderr.write(f"[predict_bridge] Exception: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()