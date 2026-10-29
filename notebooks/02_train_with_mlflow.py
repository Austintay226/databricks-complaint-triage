# Databricks notebook source
# MAGIC %md
# MAGIC # Stage 5: Training with MLflow and registering the model in Unity Catalog
# MAGIC
# MAGIC **Goal:** retrain the complaint router so that every experiment is recorded, pick the winner by a fixed rule,
# MAGIC and register it as a versioned model in Unity Catalog.
# MAGIC
# MAGIC Run one cell at a time with **Shift + Enter** on **Serverless** compute. Training takes a few minutes in total.
# MAGIC
# MAGIC Lessons:
# MAGIC 1. Experiments and runs
# MAGIC 2. Loading the gold table (and why pandas is fine here)
# MAGIC 3. Logging training runs: parameters and metrics
# MAGIC 4. Comparing runs and choosing by a rule
# MAGIC 5. Testing the winner once, and choosing the routing threshold
# MAGIC 6. Logging and registering the model in Unity Catalog
# MAGIC 7. Loading the model by its alias and scoring new data

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup: make sure MLflow is installed
# MAGIC `%pip` installs Python packages into this notebook's environment. Restarting Python afterward makes the new version take effect.
# MAGIC Run this cell once; the notebook clears its variables when Python restarts, which is why it comes first.

# COMMAND ----------

# MAGIC %pip install -q --upgrade mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lesson 1: Experiments and runs
# MAGIC
# MAGIC MLflow organizes work in two levels:
# MAGIC - An **experiment** is a project folder (here: `complaint_triage` in your user folder).
# MAGIC - A **run** is one training attempt inside it, with its **parameters** (settings you chose), **metrics** (results), **artifacts** (files like charts and the model itself) and **tags** (labels for searching).
# MAGIC
# MAGIC In your Python project, results lived in printed tables and a `results.md` file. MLflow keeps every attempt, searchable, so you can always answer "which settings produced the model we shipped?"
# MAGIC
# MAGIC `set_registry_uri("databricks-uc")` tells MLflow to register models in **Unity Catalog**, next to your tables, with the same `catalog.schema.name` naming and permissions.

# COMMAND ----------

import mlflow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

SCHEMA = "workspace.complaint_triage"
MODEL_NAME = f"{SCHEMA}.complaint_router"
TARGET_ACCURACY = 0.95

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/complaint_triage")
mlflow.set_registry_uri("databricks-uc")
print("Experiment:", f"/Users/{user}/complaint_triage")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lesson 2: Loading the gold table
# MAGIC
# MAGIC The gold table from stage 4 already has the `split` column, so training uses exactly the same rows as every other step.
# MAGIC
# MAGIC Why `toPandas()` is fine here: scikit-learn trains on one machine, and 40,000 complaints fit easily in memory.
# MAGIC Spark did the heavy lifting upstream (reading, cleaning, deduplicating); now the small, clean result moves to pandas.
# MAGIC For data too big for one machine, you'd use Spark's own ML library or distributed training instead.

# COMMAND ----------

gold = (spark.table(f"{SCHEMA}.gold_training_complaints")
             .select("date", "product", "narrative", "split")
             .toPandas())
parts = {name: gold[gold["split"] == name] for name in ["train", "validation", "test", "holdout_2026"]}
for name, part in parts.items():
    print(f"{name:13s} {len(part):6,} complaints")

train, val, test, holdout = (parts[n] for n in ["train", "validation", "test", "holdout_2026"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lesson 3: Logging training runs
# MAGIC
# MAGIC Each candidate model becomes one MLflow run. Inside `with mlflow.start_run(...)`:
# MAGIC - `log_params`: the settings (model type, regularization `C`, class weighting, feature settings)
# MAGIC - `log_metrics`: validation accuracy and macro-F1
# MAGIC - `set_tags`: a `session` tag so we can find just this batch of runs later
# MAGIC
# MAGIC The model is a scikit-learn **Pipeline**: TF-IDF text features feeding a classifier, packaged as one object, so the exact
# MAGIC text processing used in training travels with the model. `ColumnTransformer` lets it take a table with a `narrative` column, which is the shape a serving endpoint sends.
# MAGIC
# MAGIC Only the **validation** set is used here. The test set stays untouched until one model has been chosen.

# COMMAND ----------

STOP_WORDS = list(ENGLISH_STOP_WORDS | {"xx", "xxx", "xxxx"})  # CFPB masks names and dates as XXXX


def make_pipeline(classifier):
    features = ColumnTransformer([(
        "tfidf",
        TfidfVectorizer(stop_words=STOP_WORDS, ngram_range=(1, 2), min_df=3, max_df=0.9,
                        sublinear_tf=True, max_features=200_000),
        "narrative",
    )])
    return Pipeline([("features", features), ("classifier", classifier)])


candidates = {
    "naive_bayes": (ComplementNB(alpha=0.3), {"alpha": 0.3}),
    "logreg_c1": (LogisticRegression(C=1, max_iter=2000), {"C": 1, "class_weight": "none"}),
    "logreg_c5": (LogisticRegression(C=5, max_iter=2000), {"C": 5, "class_weight": "none"}),
    "logreg_c5_balanced": (LogisticRegression(C=5, max_iter=2000, class_weight="balanced"),
                           {"C": 5, "class_weight": "balanced"}),
}

session = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
fitted, run_ids = {}, {}
for name, (classifier, params) in candidates.items():
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({"stage": "candidate", "session": session})
        mlflow.log_params({"model_type": type(classifier).__name__, "ngram_range": "1-2",
                           "max_features": 200_000, "train_rows": len(train), **params})
        pipe = make_pipeline(classifier).fit(train[["narrative"]], train["product"])
        pred = pipe.predict(val[["narrative"]])
        metrics = {"val_accuracy": accuracy_score(val["product"], pred),
                   "val_macro_f1": f1_score(val["product"], pred, average="macro")}
        mlflow.log_metrics(metrics)
        fitted[name], run_ids[name] = pipe, run.info.run_id
        print(f"{name:20s} val accuracy {metrics['val_accuracy']:.3f}   val macro-F1 {metrics['val_macro_f1']:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lesson 4: Comparing runs and choosing by a rule
# MAGIC
# MAGIC **Look in the UI:** in the left sidebar, open **Experiments**, then `complaint_triage`. You'll see a row per run.
# MAGIC Select several runs and click **Compare** to see their parameters and metrics side by side.
# MAGIC
# MAGIC **Then in code:** `mlflow.search_runs` queries runs like a table. We pick the winner with the same rule as the Python project:
# MAGIC **highest validation macro-F1**. Deciding the rule before looking at test results is what keeps the reported test numbers honest.

# COMMAND ----------

runs = mlflow.search_runs(filter_string=f"tags.session = '{session}'",
                          order_by=["metrics.val_macro_f1 DESC"])
display(runs[["tags.mlflow.runName", "params.model_type", "params.C", "params.class_weight",
              "metrics.val_accuracy", "metrics.val_macro_f1"]])

best_name = runs.iloc[0]["tags.mlflow.runName"]
best_pipe, best_run_id = fitted[best_name], run_ids[best_name]
print("Selected by validation macro-F1:", best_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lesson 5: Testing the winner once, and choosing the routing threshold
# MAGIC
# MAGIC Two steps, in the right order:
# MAGIC 1. **Threshold from validation:** the lowest confidence cutoff where auto-routed complaints still reach 95% accuracy on validation data.
# MAGIC 2. **Test once:** apply the model and that threshold to the test set and record the results.
# MAGIC
# MAGIC Everything is logged back into the winning run, so the run holds the full story: settings, validation, threshold, test results, and a confusion matrix chart (`log_figure`).

# COMMAND ----------

def threshold_for(conf, correct, target, min_routed=50):
    """Lowest confidence cutoff whose auto-routed set still meets the target accuracy."""
    order = np.argsort(-conf)
    cum_acc = np.cumsum(correct[order]) / np.arange(1, len(conf) + 1)
    ok = np.where(cum_acc[min_routed - 1:] >= target)[0]
    return 1.01 if len(ok) == 0 else float(conf[order][ok.max() + min_routed - 1])


def routing_metrics(pipe, part, threshold, prefix):
    proba = pipe.predict_proba(part[["narrative"]])
    pred = pipe.classes_[proba.argmax(axis=1)]
    correct = pred == part["product"].to_numpy()
    auto = proba.max(axis=1) >= threshold
    return pred, {
        f"{prefix}_accuracy": float(correct.mean()),
        f"{prefix}_macro_f1": float(f1_score(part["product"], pred, average="macro")),
        f"{prefix}_auto_routed_share": float(auto.mean()),
        f"{prefix}_auto_routed_accuracy": float(correct[auto].mean()) if auto.any() else float("nan"),
    }


val_proba = best_pipe.predict_proba(val[["narrative"]])
val_correct = best_pipe.classes_[val_proba.argmax(axis=1)] == val["product"].to_numpy()
threshold = threshold_for(val_proba.max(axis=1), val_correct, TARGET_ACCURACY)

test_pred, test_metrics = routing_metrics(best_pipe, test, threshold, "test")

with mlflow.start_run(run_id=best_run_id):
    mlflow.set_tag("stage", "champion_candidate")
    mlflow.log_params({"target_accuracy": TARGET_ACCURACY, "auto_route_threshold": round(threshold, 4)})
    mlflow.log_metrics(test_metrics)
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(test["product"], test_pred, normalize="true",
                                            xticks_rotation=90, values_format=".0%", ax=ax, colorbar=False)
    ax.set_title("Test set, row %")
    fig.tight_layout()
    mlflow.log_figure(fig, "confusion_matrix_test.png")
    plt.close(fig)

print(f"Threshold for {TARGET_ACCURACY:.0%} target: {threshold:.2f}")
for k, v in test_metrics.items():
    print(f"{k:28s} {v:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lesson 6: Logging and registering the model
# MAGIC
# MAGIC - **`log_model`** saves the whole pipeline as an artifact of the run.
# MAGIC - A **signature** records the expected input (a table with a string `narrative` column) and output. Unity Catalog requires one, and serving endpoints use it to reject malformed requests.
# MAGIC - **`registered_model_name`** creates the model in Unity Catalog, or adds a **new version** if it already exists. Rerun this notebook and you get version 2, 3, and so on, each linked to the run that produced it.
# MAGIC - An **alias** such as `champion` is a movable pointer to one version. Downstream jobs load `@champion`, so promoting a new model, or rolling back, means moving the alias rather than changing code.
# MAGIC - A **tag** on the version stores the routing threshold, so the threshold travels with the model.
# MAGIC
# MAGIC After running, open **Catalog**, then `workspace`, then `complaint_triage`, and find `complaint_router` under **Models** to see its versions.

# COMMAND ----------

example = val[["narrative"]].head(3)
signature = infer_signature(example, best_pipe.predict(example))

with mlflow.start_run(run_id=best_run_id):
    mlflow.sklearn.log_model(best_pipe, name="model", signature=signature,
                             input_example=example, registered_model_name=MODEL_NAME)

client = MlflowClient()
version = max(int(v.version) for v in client.search_model_versions(f"name = '{MODEL_NAME}'"))
client.set_registered_model_alias(MODEL_NAME, "champion", version)
client.set_model_version_tag(MODEL_NAME, version, "auto_route_threshold", f"{threshold:.4f}")
client.set_model_version_tag(MODEL_NAME, version, "selected_by", "validation macro-F1")
print(f"Registered {MODEL_NAME} version {version} with alias 'champion'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lesson 7: Loading the model by alias and scoring new data
# MAGIC
# MAGIC Anyone with access can now load the model by name and alias, without knowing which run or file it came from.
# MAGIC - `mlflow.pyfunc.load_model` gives a **generic** model with a `predict` method. This is the interface serving endpoints and batch jobs use, whatever library built the model.
# MAGIC - `mlflow.sklearn.load_model` gives back the original scikit-learn pipeline, including `predict_proba`, which the routing threshold needs.
# MAGIC
# MAGIC Scoring the 2026 holdout should reproduce your label-shift story: lower overall accuracy, driven by the change in product mix.

# COMMAND ----------

champion = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
print("pyfunc predictions:", [str(p) for p in champion.predict(holdout[["narrative"]].head(3))])

champion_sk = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
saved_threshold = float(client.get_model_version(MODEL_NAME, version).tags["auto_route_threshold"])
_, holdout_metrics = routing_metrics(champion_sk, holdout, saved_threshold, "holdout_2026")
for k, v in holdout_metrics.items():
    print(f"{k:36s} {v:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wrap-up
# MAGIC
# MAGIC You've used: MLflow experiments, runs, parameters, metrics, tags, `search_runs`, `log_figure`, model signatures,
# MAGIC Unity Catalog model registration, versions, aliases, and loading by alias.
# MAGIC
# MAGIC **Write down your answers** (interview material):
# MAGIC 1. Which model won, and how close were the test numbers to your Python project (84.2% accuracy, 0.701 macro-F1)?
# MAGIC 2. Why do downstream jobs load `@champion` instead of a version number?
# MAGIC 3. If the new model misbehaves in production, how would you roll back?
# MAGIC
# MAGIC **Next (stage 6):** a scheduled job that loads `@champion` and writes routing decisions to a table.
