
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from mlflow.tracking import MlflowClient


# --------------------------------------------------
# 1. PROJECT PATHS AND MLFLOW SETTINGS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_DIR = PROJECT_DIR / "models" / "machine_learning"
PREPROCESSOR_PATH = PROJECT_DIR / "models" / "preprocessor.joblib"

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Project6_Rossmann_RandomForest"
REGISTERED_MODEL_NAME = "Rossmann_RandomForest_Sales"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


# --------------------------------------------------
# 2. EXISTING MODEL FILES
# --------------------------------------------------

models_to_register = [
    {
        "version_label": "Pilot",
        "file": MODEL_DIR / "RandomForest_05-11-2026-17-11-55-00.pkl",
        "training_rows": 60000,
        "metrics": {
            "validation_MAE": 1898.0760,
            "validation_RMSE": 2570.8308,
            "validation_RMSPE_Percent": 42.39133,
        },
    },
    {
        "version_label": "FullTraining",
        "file": MODEL_DIR / "RandomForest_FullTraining_05-11-2026-21-49-07-00.pkl",
        "training_rows": 970379,
        "metrics": {
            "validation_MAE": 1743.0160,
            "validation_RMSE": 2367.1110,
            "validation_RMSPE_Percent": 41.54042,
        },
    },
]


# --------------------------------------------------
# 3. LOAD AND PREPARE MODEL PIPELINE
# --------------------------------------------------

def load_combined_pipeline(model_path):

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    model_data = joblib.load(model_path)

    if not isinstance(model_data, dict):
        raise TypeError(
            "Expected the saved model artifact to be a dictionary."
        )

    # Get the trained Random Forest pipeline
    saved_pipeline = model_data["model_pipeline"]

    # Extract the trained Random Forest estimator
    if hasattr(saved_pipeline, "named_steps"):
        if "regressor" in saved_pipeline.named_steps:
            estimator = saved_pipeline.named_steps["regressor"]
        else:
            estimator = saved_pipeline.steps[-1][1]
    else:
        estimator = saved_pipeline

    # Get the fitted preprocessor
    preprocessor = model_data.get("preprocessor")

    if preprocessor is None:
        preprocessor = joblib.load(PREPROCESSOR_PATH)

    # Combine preprocessing and model into one pipeline
    combined_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", estimator),
        ]
    )

    return combined_pipeline, model_data


# --------------------------------------------------
# 4. REGISTER BOTH MODEL VERSIONS
# --------------------------------------------------

client = MlflowClient()

for model_info in models_to_register:

    version_label = model_info["version_label"]
    model_path = model_info["file"]

    print("\n" + "=" * 60)
    print(f"Registering {version_label} Random Forest Model")
    print("=" * 60)

    combined_pipeline, model_data = load_combined_pipeline(
        model_path
    )

    metadata = model_data.get("metadata", {})

    # Log model parameters
    estimator = combined_pipeline.named_steps["regressor"]
    model_params = estimator.get_params()

    with mlflow.start_run(
        run_name=f"RandomForest_{version_label}"
    ) as run:

        mlflow.set_tags({
            "model_type": "RandomForestRegressor",
            "model_stage": version_label,
            "target": "Sales",
            "project": "Rossmann Store Sales Forecasting",
        })

        mlflow.log_params({
            "model_stage": version_label,
            "training_rows": model_info["training_rows"],
            "n_estimators": model_params.get("n_estimators"),
            "max_depth": model_params.get("max_depth"),
            "min_samples_split": model_params.get("min_samples_split"),
            "min_samples_leaf": model_params.get("min_samples_leaf"),
            "max_features": str(model_params.get("max_features")),
            "random_state": model_params.get("random_state"),
            "processed_feature_count": len(
                model_data.get("feature_names", [])
            ),
        })

        # Log validation metrics
        mlflow.log_metrics(model_info["metrics"])

        # Log the complete preprocessing + prediction pipeline
        model_result = mlflow.sklearn.log_model(
        sk_model=combined_pipeline,
        name="random_forest_pipeline",
        registered_model_name=REGISTERED_MODEL_NAME,
        skops_trusted_types=[
            "numpy.dtype",
            "sklearn.tree._tree.Tree"
        ]
)

        print("Run ID:", run.info.run_id)
        print("Model URI:", model_result.model_uri)
        print(f"{version_label} model logged and registered.")

    print("Registration completed.")


# --------------------------------------------------
# 5. VERIFY REGISTERED MODEL VERSIONS
# --------------------------------------------------

print("\n" + "=" * 60)
print("REGISTERED MODEL VERSION HISTORY")
print("=" * 60)

registered_versions = client.search_model_versions(
    f"name='{REGISTERED_MODEL_NAME}'"
)

registered_versions = sorted(
    registered_versions,
    key=lambda item: int(item.version)
)

for version in registered_versions:
    print(
        f"Version: {version.version} | "
        f"Status: {version.status} | "
        f"Run ID: {version.run_id}"
    )

print("\nTotal registered versions:", len(registered_versions))
print("\nMLflow registration process finished.")
