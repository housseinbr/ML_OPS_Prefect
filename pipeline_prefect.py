import argparse
import subprocess  # nosec B404
import sys

from prefect import flow, task

from model_pipeline import (
    evaluate_model,
    load_data,
    load_model,
    prepare_data,
    save_model,
    train_model,
)

DATA_PATH = "diabetic_data.csv"
FICHIERS = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


def lancer(commande, check=True):
    subprocess.run([sys.executable, "-m"] + commande, check=check)  # nosec B603


@task
def install_dependencies():
    lancer(["pip", "install", "-r", "requirements.txt"])


@task
def format_code():
    lancer(["black"] + FICHIERS)


@task
def check_quality():
    lancer(["pylint", "--disable=C0114,C0116"] + FICHIERS, check=False)


@task
def check_security():
    lancer(["bandit", "-q"] + FICHIERS, check=False)


@task
def run_tests():
    lancer(["pytest", "tests", "-v"])


@task
def import_data_task():
    df = load_data(DATA_PATH)
    print(f"Donnees importees : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


@task
def prepare_data_task():
    return prepare_data(DATA_PATH)


@task
def train_model_task(data):
    X_train, X_test, y_train, y_test = data
    return train_model(X_train, y_train)


@task
def save_model_task(model):
    save_model(model)


@task
def load_model_task():
    return load_model()


@task
def evaluate_model_task(model, data):
    X_train, X_test, y_train, y_test = data
    return evaluate_model(model, X_test, y_test)


@flow(name="code")
def code_flow():
    install_dependencies()
    format_code()
    check_quality()
    check_security()
    run_tests()


@flow(name="import")
def import_flow():
    import_data_task()


@flow(name="prepare")
def prepare_flow():
    prepare_data_task()


@flow(name="train")
def train_flow():
    data = prepare_data_task()
    model = train_model_task(data)
    save_model_task(model)


@flow(name="save")
def save_flow():
    data = prepare_data_task()
    model = train_model_task(data)
    save_model_task(model)


@flow(name="evaluate")
def evaluate_flow():
    model = load_model_task()
    data = prepare_data_task()
    evaluate_model_task(model, data)


@flow(name="all")
def all_flow():
    code_flow()
    import_data_task()
    data = prepare_data_task()
    model = train_model_task(data)
    save_model_task(model)
    evaluate_model_task(model, data)


FLOWS = {
    "all": all_flow,
    "import": import_flow,
    "prepare": prepare_flow,
    "save": save_flow,
    "code": code_flow,
    "train": train_flow,
    "entrainement": train_flow,
    "evaluate": evaluate_flow,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow", choices=FLOWS, default="all")
    args = parser.parse_args()
    FLOWS[args.flow]()
