import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42

CODES_DECES = [11, 19, 20, 21]

CORRESPONDANCE_CIBLE = {"NO": 0, ">30": 1, "<30": 1}

CORRESPONDANCE_AGE = {
    "[0-10)": 0,
    "[10-20)": 1,
    "[20-30)": 2,
    "[30-40)": 3,
    "[40-50)": 4,
    "[50-60)": 5,
    "[60-70)": 6,
    "[70-80)": 7,
    "[80-90)": 8,
    "[90-100)": 9,
}

CORRESPONDANCE_MED = {"No": 0, "Steady": 1, "Up": 2, "Down": 3}

COLONNES_MED = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

PARAM_GRID_DT = {"max_depth": [5, 8, 12, 15], "min_samples_leaf": [2, 5, 10]}


def convert_icd(code):
    """Regroupement ICD-9 simplifié (utilisé pour les variables diag_*_cat)."""
    if code == "?" or pd.isna(code) or str(code).strip() == "Inconnu":
        return "Inconnu"
    if str(code).startswith("V") or str(code).startswith("E"):
        return "Autre"
    try:
        val = float(code)
        if (390 <= val <= 459) or val == 785:
            return "Circulatoire"
        if (460 <= val <= 519) or val == 786:
            return "Respiratoire"
        if (520 <= val <= 579) or val == 787:
            return "Digestif"
        if int(val) == 250:
            return "Diabete"
        if 800 <= val <= 999:
            return "Traumatisme"
        if 710 <= val <= 739:
            return "Musculo_squelettique"
        if (580 <= val <= 629) or val == 788:
            return "Genito_urinaire"
        if 140 <= val <= 239:
            return "Tumeurs"
        return "Autre"
    except ValueError:
        return "Autre"


def regrouper_diagnostic(code):
    """Regroupe un code ICD-9 en famille médicale (utilisé pour diag_1/2/3)."""
    code = str(code).strip()
    if code == "Inconnu":
        return "Inconnu"
    if code.startswith("V"):
        return "Facteur_sante"
    if code.startswith("E"):
        return "Cause_externe"
    try:
        num = float(code)
    except ValueError:
        return "Autre"
    familles = [
        (1, 139, "Infectieux"),
        (140, 239, "Tumeurs"),
        (240, 279, "Endocrinien"),
        (280, 289, "Sanguin"),
        (290, 319, "Troubles_mentaux"),
        (320, 389, "Systeme_nerveux"),
        (390, 459, "Circulatoire"),
        (460, 519, "Respiratoire"),
        (520, 579, "Digestif"),
        (580, 629, "Genito_urinaire"),
        (630, 679, "Grossesse"),
        (680, 709, "Peau"),
        (710, 739, "Musculo_squelettique"),
        (740, 759, "Congenital"),
        (760, 799, "Perinatal_Symptomes"),
        (800, 999, "Traumatisme"),
    ]
    for borne_inf, borne_sup, famille in familles:
        if borne_inf <= num <= borne_sup:
            return famille
    return "Autre"


def load_data(data_path):
    """Charge le fichier CSV brut."""
    df = pd.read_csv(data_path)
    print(f"[load_data] {df.shape[0]} séjours, {df.shape[1]} variables")
    return df


def clean_data(df):
    """Nettoyage : colonnes inutiles, NaN, valeurs '?', séjours avec décès."""
    df = df.copy()

    # weight (97 % manquant) et payer_code (variable administrative)
    df = df.drop(columns=["weight", "payer_code"], errors="ignore")

    # NaN des tests biologiques = test non réalisé
    df["max_glu_serum"] = df["max_glu_serum"].fillna("Non_teste")
    df["A1Cresult"] = df["A1Cresult"].fillna("Non_teste")

    # Valeurs inconnues
    df["gender"] = df["gender"].replace("Unknown/Invalid", "Female")
    for col in ["race", "medical_specialty", "diag_1", "diag_2", "diag_3"]:
        df[col] = df[col].replace("?", "Inconnu")

    # Exclusion des séjours terminés par un décès
    df = df[~df["discharge_disposition_id"].isin(CODES_DECES)].copy()

    print(f"[clean_data] {df.shape[0]} séjours après nettoyage")
    return df


def feature_engineering(df):
    """Création des variables médicales dérivées."""
    df = df.copy()

    df["patient_stay_count"] = df["patient_nbr"].map(df["patient_nbr"].value_counts())
    df["total_visits"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )
    df["has_inpatient"] = (df["number_inpatient"] > 0).astype(int)
    df["has_emergency"] = (df["number_emergency"] > 0).astype(int)
    df["lab_per_day"] = df["num_lab_procedures"] / (df["time_in_hospital"] + 1e-5)
    df["meds_per_day"] = df["num_medications"] / (df["time_in_hospital"] + 1e-5)
    df["procedures_per_day"] = df["num_procedures"] / (df["time_in_hospital"] + 1e-5)
    df["num_changes"] = (df["change"] == "Ch").astype(int)
    df["high_inpatient"] = (df["number_inpatient"] >= 2).astype(int)

    for col in ["diag_1", "diag_2", "diag_3"]:
        df[col + "_cat"] = df[col].apply(convert_icd)

    df = df.drop(columns=["encounter_id", "patient_nbr"], errors="ignore")
    return df


def encode_features(df):
    """Sépare X / y et encode toutes les variables catégorielles."""
    X = df.drop(columns=["readmitted"]).copy()
    y = df["readmitted"].map(CORRESPONDANCE_CIBLE)
    if y.isna().any():
        raise ValueError("Valeurs non reconnues dans la colonne readmitted")

    X["age"] = X["age"].map(CORRESPONDANCE_AGE)
    X["gender"] = X["gender"].map({"Female": 0, "Male": 1})
    X["change"] = X["change"].map({"No": 0, "Ch": 1})
    X["diabetesMed"] = X["diabetesMed"].map({"No": 0, "Yes": 1})
    for col in COLONNES_MED:
        if col in X.columns:
            X[col] = X[col].map(CORRESPONDANCE_MED)

    for col in ["diag_1", "diag_2", "diag_3"]:
        X[col] = X[col].apply(regrouper_diagnostic)

    seuil = int(len(X) * 0.005)
    freq = X["medical_specialty"].value_counts()
    rares = freq[freq < seuil].index
    X["medical_specialty"] = X["medical_specialty"].replace(
        list(rares), "Autre_specialite"
    )

    colonnes_nominales = X.select_dtypes(include="object").columns.tolist()
    X = pd.get_dummies(X, columns=colonnes_nominales, dtype=float)

    if X.isna().sum().sum() > 0:
        raise ValueError("Des valeurs manquantes subsistent dans X après encodage")

    print(f"[encode_features] X : {X.shape}, y : {y.shape}")
    return X, y


def prepare_data(data_path="diabetic_data.csv", test_size=0.20):
    """
    Charge et prétraite les données, puis fait un split stratifié.

    Retourne : X_train, X_test, y_train, y_test
    """
    df = load_data(data_path)
    df = clean_data(df)
    df = feature_engineering(df)
    X, y = encode_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[prepare_data] Train : {X_train.shape[0]} | Test : {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, param_grid=None, cv=3):
    """
    Entraîne un arbre de décision avec recherche d'hyperparamètres (GridSearchCV).

    Retourne : le meilleur modèle entraîné
    """
    if param_grid is None:
        param_grid = PARAM_GRID_DT

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid=param_grid,
        cv=skf,
        scoring="f1_macro",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    print(f"[train_model] Meilleurs paramètres : {grid.best_params_}")
    print(f"[train_model] F1 macro (CV)        : {grid.best_score_:.4f}")
    return grid.best_estimator_


def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle sur le jeu de test.

    Retourne : dictionnaire des métriques
    """
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    print("[evaluate_model] Métriques sur le test :")
    for nom, valeur in metrics.items():
        print(f"  {nom:10s}: {valeur:.4f}")
    print("\nMatrice de confusion :")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
    print("\nRapport de classification :")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Non réadmis (0)", "Réadmis (1)"],
            zero_division=0,
        )
    )
    return metrics


def save_model(model, model_path="model.joblib"):
    """Sauvegarde le modèle entraîné avec joblib."""
    joblib.dump(model, model_path)
    print(f"[save_model] Modèle sauvegardé : {model_path}")


def load_model(model_path="model.joblib"):
    """Charge un modèle sauvegardé avec joblib."""
    model = joblib.load(model_path)
    print(f"[load_model] Modèle chargé : {model_path}")
    return model
