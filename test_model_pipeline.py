import numpy as np
from sklearn.tree import DecisionTreeClassifier

from model_pipeline import (
    convert_icd,
    load_model,
    regrouper_diagnostic,
    save_model,
)


def test_regrouper_diagnostic_familles():
    assert regrouper_diagnostic("250.83") == "Endocrinien"
    assert regrouper_diagnostic("428") == "Circulatoire"
    assert regrouper_diagnostic("V57") == "Facteur_sante"
    assert regrouper_diagnostic("E909") == "Cause_externe"
    assert regrouper_diagnostic("Inconnu") == "Inconnu"


def test_convert_icd_familles():
    assert convert_icd("250.83") == "Diabete"
    assert convert_icd("428") == "Circulatoire"
    assert convert_icd("?") == "Inconnu"
    assert convert_icd("V57") == "Autre"


def test_save_et_load_model(tmp_path):
    X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    y = np.array([0, 1, 1, 0])
    model = DecisionTreeClassifier(random_state=42).fit(X, y)

    chemin = str(tmp_path / "model.joblib")
    save_model(model, chemin)
    model_charge = load_model(chemin)

    assert (model_charge.predict(X) == model.predict(X)).all()
