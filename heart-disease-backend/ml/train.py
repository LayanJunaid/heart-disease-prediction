
import sys
import os
import json
import joblib
import pickle
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler  # <--- EKLENDİ
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    accuracy_score, roc_auc_score
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    SCRIPT_DIR, "saved_data"
)


def check_file(path, label):
    if not os.path.exists(path):
        print(f"   ERROR: '{label}' not found: {path}")
        print(f"     → PLEASE run 05_embedded_feature_selection.py first.")
        sys.exit(1)


def main():
    print("=" * 60)
    print("  Heart Disease Prediction — SVM Model Training")
    print("  Feature Set: Embedded Union (Emb-Union)")
    print("=" * 60)
    print(f"\n  saved_data yolu: {SAVED_DATA}\n")

    
    required = {
        "X_train_scaled.pkl":       "Scaled training data",
        "X_test_scaled.pkl":        "Scaled test data",
        "y_train.pkl":              "Training labels",
        "y_test.pkl":               "Test labels",
        "feature_names.pkl":        "All feature names (after OHE)",
        "embedded_union_features.pkl": "Emb-Union selected feature names",
        "scaler.pkl":               "StandardScaler object",
    }
    print("  Files are being checked...")
    for fname, label in required.items():
        check_file(os.path.join(SAVED_DATA, fname), label)
        print(f"  {fname}")

    print("\n  Loading data...")
    X_train_sc = joblib.load(os.path.join(SAVED_DATA, "X_train_scaled.pkl"))
    X_test_sc  = joblib.load(os.path.join(SAVED_DATA, "X_test_scaled.pkl"))
    y_train    = joblib.load(os.path.join(SAVED_DATA, "y_train.pkl"))
    y_test     = joblib.load(os.path.join(SAVED_DATA, "y_test.pkl"))
    feat_names = joblib.load(os.path.join(SAVED_DATA, "feature_names.pkl"))
    union_feat = joblib.load(os.path.join(SAVED_DATA, "embedded_union_features.pkl"))
    scaler     = joblib.load(os.path.join(SAVED_DATA, "scaler.pkl"))


#checking if loaded objects are already DataFrames/Series, if not convert them.
    if not isinstance(X_train_sc, pd.DataFrame):
        X_train_sc = pd.DataFrame(X_train_sc, columns=feat_names)
    if not isinstance(X_test_sc, pd.DataFrame):
        X_test_sc  = pd.DataFrame(X_test_sc,  columns=feat_names)
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)
    if not isinstance(y_test, pd.Series):
        y_test  = pd.Series(y_test)

    X_train_sc = X_train_sc.reset_index(drop=True)
    X_test_sc  = X_test_sc.reset_index(drop=True)
    y_train    = y_train.reset_index(drop=True)
    y_test     = y_test.reset_index(drop=True)

    print(f"    Total features          : {len(feat_names)}")
    print(f"    Emb-Union feature count : {len(union_feat)}")
    print(f"    Emb-Union features      : {union_feat}")
    print(f"    Training samples        : {X_train_sc.shape[0]}")
    print(f"    Test samples            : {X_test_sc.shape[0]}")
    print(f"    Class distribution (train): {y_train.value_counts().to_dict()}")

    missing = [f for f in union_feat if f not in X_train_sc.columns]
    if missing:
        print(f"\n   The following features were not found: {missing}")
        print(f"     Available features: {list(X_train_sc.columns)}")
        sys.exit(1)

    #we transform the raw scaled data back to its original form using the inverse_transform method of the scaler. This gives us the raw feature values before scaling, which we can then filter to keep only the 14 features selected by the embedded union method. After filtering, we fit a new StandardScaler on this reduced set of features to ensure that our SVM model is trained on properly scaled data.
    X_train_raw = pd.DataFrame(scaler.inverse_transform(X_train_sc), columns=feat_names)
    X_test_raw  = pd.DataFrame(scaler.inverse_transform(X_test_sc), columns=feat_names)

    #now we filter 14 features selected by the embedded union method. 
    X_train_14_raw = X_train_raw[union_feat].copy()
    X_test_14_raw  = X_test_raw[union_feat].copy()

    # train only for 14 features 
    new_scaler = StandardScaler()
    X_train = pd.DataFrame(new_scaler.fit_transform(X_train_14_raw), columns=union_feat)
    X_test  = pd.DataFrame(new_scaler.transform(X_test_14_raw), columns=union_feat)

    print(f"\n  Feature seçimi OK: {X_train.shape[1]} feature")
    print(f"  YENİ Scaler {X_train.shape[1]} özellik için başarıyla eğitildi.")

    # SVM train
    # 09_hyperparameter_tuning.py'den en iyi parametreler:
    # C=1.0 | kernel=rbf | gamma=scale | class_weight=None
    print("\n  SVM train...")
    print("    C=1.0 | kernel=rbf | gamma=scale | probability=True")

    model = SVC(
        C=1.0,
        kernel="rbf",
        gamma="scale",
        class_weight=None,
        probability=True,   # predict_proba() için ZORUNLU
        random_state=42,
    )
    model.fit(X_train, y_train)
    print("Eğitim tamamlandı")

    #  5. evaluation on test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy" : round(accuracy_score(y_test, y_pred)  * 100, 2),
        "precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        "recall"   : round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        "f1"       : round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        "roc_auc"  : round(roc_auc_score(y_test, y_prob) * 100, 2),
    }

    print("\n   results on test set:")
    print(f"    Accuracy  : {metrics['accuracy']:.2f}%")
    print(f"    Precision : {metrics['precision']:.2f}%")
    print(f"    Recall    : {metrics['recall']:.2f}%  ← tıbbi öncelik")
    print(f"    F1-Score  : {metrics['f1']:.2f}%")
    print(f"    ROC-AUC   : {metrics['roc_auc']:.2f}%")

    print("\n   Expected vs Actual Metrics:")
    expected = {"f1": 86.21, "recall": 89.29, "roc_auc": 94.91}
    for k, exp in expected.items():
        got   = metrics[k]
        delta = got - exp
        sign  = f"↑ +{delta:.2f}" if delta >= 0 else f"↓ {delta:.2f}"
        print(f"    {k:<10}: expected={exp:.2f}%  got={got:.2f}%  {sign}")




    out = {
        "model.pkl"   : model,
        "features.pkl": union_feat,
        "scaler.pkl"  : new_scaler,  
    }
    for fname, obj in out.items():
        with open(os.path.join(SCRIPT_DIR, fname), "wb") as f:
            pickle.dump(obj, f)
        print(f"\n{fname} saved")

    meta = {
        "model_name"   : "SVM",
        "kernel"       : "rbf",
        "feature_set"  : "Emb-Union",
        "n_features"   : len(union_feat),
        "feature_names": union_feat,
        "metrics"      : metrics,
        "trained_on"   : str(pd.Timestamp.now()),
    }
    meta_path = os.path.join(SCRIPT_DIR, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f" meta.json saved with model metadata")

    print(f"\n  total files: {SCRIPT_DIR}/")
    print(f"\n  ok now you can run npm run dev\n")


if __name__ == "__main__":
    main()

    