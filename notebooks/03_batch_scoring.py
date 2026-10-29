# Databricks notebook source
# MAGIC %md
# MAGIC # Stage 6: Batch scoring job
# MAGIC
# MAGIC **Goal:** a notebook that a scheduled **job** runs with no one watching. Each run loads the `champion` model,
# MAGIC finds complaints that haven't been scored yet, routes them, and records the decisions in a table.
# MAGIC
# MAGIC First, run it by hand (one cell at a time, Shift + Enter, on Serverless) to learn what it does.
# MAGIC Then you'll turn it into a scheduled job (instructions in chat).
# MAGIC
# MAGIC Lessons:
# MAGIC 1. Job parameters (widgets)
# MAGIC 2. Loading the champion, pinned to one version
# MAGIC 3. Creating the output tables
# MAGIC 4. Finding what's new: incremental and idempotent processing
# MAGIC 5. Scoring and routing decisions
# MAGIC 6. Writing with MERGE
# MAGIC 7. Logging each run

# COMMAND ----------

# MAGIC %pip install -q --upgrade mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lesson 1: Job parameters (widgets)
# MAGIC
# MAGIC **Widgets** are the inputs a notebook accepts. When you run it by hand, they appear as boxes at the top of the notebook.
# MAGIC When a **job** runs it, the job passes values in as **parameters** with the same names.
# MAGIC That lets one notebook serve several purposes without editing code: for example a small `batch_size` for testing,
# MAGIC or `model_alias = challenger` to score with a candidate model.
# MAGIC
# MAGIC Here `batch_size` limits how many complaints each run scores, so a few runs simulate several days of new complaints arriving.

# COMMAND ----------

dbutils.widgets.text("batch_size", "5000", "Complaints to score per run")
dbutils.widgets.text("model_alias", "champion", "Model alias to use")

BATCH_SIZE = int(dbutils.widgets.get("batch_size"))
MODEL_ALIAS = dbutils.widgets.get("model_alias")

SCHEMA = "workspace.complaint_triage"
MODEL_NAME = f"{SCHEMA}.complaint_router"
GOLD = f"{SCHEMA}.gold_training_complaints"
DECISIONS = f"{SCHEMA}.routing_decisions"
RUNS = f"{SCHEMA}.scoring_runs"
print(f"batch_size={BATCH_SIZE}, model_alias={MODEL_ALIAS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lesson 2: Loading the champion, pinned to one version
# MAGIC
# MAGIC The job asks Unity Catalog which version the alias points to **right now**, then loads **that exact version**
# MAGIC and reads its threshold tag. Loading by version number after resolving the alias means that even if someone
# MAGIC moves the alias mid-run, this run stays consistent, and every decision it writes records the version that made it.

# COMMAND ----------

import uuid

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from pyspark.sql import functions as F

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
model_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
version = str(model_version.version)
threshold = float(model_version.tags["auto_route_threshold"])
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")
print(f"Loaded {MODEL_NAME} version {version} ({MODEL_ALIAS}), auto-route threshold {threshold:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lesson 3: Creating the output tables
# MAGIC
# MAGIC Two tables, created only if they don't exist yet, so the job can run again and again:
# MAGIC - **`routing_decisions`**: one row per complaint, with the queue, confidence, auto-route or human-review decision, and the model version that decided.
# MAGIC - **`scoring_runs`**: one row per job run, an operational log (how many were scored, what share was automated).
# MAGIC
# MAGIC Declaring the columns and types up front is a **contract**: downstream users, like the stage 7 dashboard, can rely on the shape not drifting.
# MAGIC
# MAGIC A note on `correct`: here we already know the true product, so we can measure accuracy immediately.
# MAGIC In a real deployment the true answer arrives later, from analysts' corrections, and would be joined in afterward.

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {DECISIONS} (
  complaint_id STRING, date DATE, month STRING,
  actual_product STRING, predicted_queue STRING, second_choice STRING,
  confidence DOUBLE, decision STRING, correct BOOLEAN,
  model_version STRING, threshold DOUBLE, scoring_run_id STRING, scored_at TIMESTAMP
)""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {RUNS} (
  scoring_run_id STRING, scored_at TIMESTAMP, model_version STRING, threshold DOUBLE,
  complaints_scored BIGINT, auto_routed_share DOUBLE, auto_routed_accuracy DOUBLE,
  remaining_after_run BIGINT
)""")
print("Tables ready:", DECISIONS, "and", RUNS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lesson 4: Finding what's new
# MAGIC
# MAGIC A **left anti join** keeps the rows from the left table that have **no match** in the right table:
# MAGIC here, complaints that don't have a routing decision yet.
# MAGIC
# MAGIC This makes the job **incremental** (each run only processes new work) and **idempotent** (rerunning it, or a
# MAGIC job retry after a failure, never scores the same complaint twice). Idempotency is one of the most important
# MAGIC properties of a production job, and a good word to use in interviews.
# MAGIC
# MAGIC We score everything except the training rows, oldest first, to mimic complaints arriving over time.

# COMMAND ----------

already_scored = spark.table(DECISIONS).select("complaint_id")
to_score = (spark.table(GOLD)
                 .filter(F.col("split") != "train")
                 .withColumn("complaint_id", F.col("complaint_id").cast("string"))
                 .join(already_scored, "complaint_id", "left_anti"))

waiting = to_score.count()
batch = (to_score.orderBy("date", "complaint_id")
                 .limit(BATCH_SIZE)
                 .select("complaint_id", "date", "month", "product", "narrative")
                 .toPandas())
print(f"Waiting to be scored: {waiting:,}. Scoring {len(batch):,} this run.")

if batch.empty:
    dbutils.notebook.exit("Nothing new to score")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lesson 5: Scoring and routing decisions
# MAGIC
# MAGIC For each complaint: the predicted queue, the model's confidence, its second choice (useful to a human reviewer),
# MAGIC and the decision: **auto_route** if confidence clears the threshold, otherwise **human_review**.
# MAGIC This is the routing policy from your Python project, now running as a service would.

# COMMAND ----------

proba = model.predict_proba(batch[["narrative"]])
ranked = np.argsort(-proba, axis=1)
classes = model.classes_
run_id = str(uuid.uuid4())
scored_at = pd.Timestamp.now(tz="UTC").tz_localize(None)

decisions = pd.DataFrame({
    "complaint_id": batch["complaint_id"],
    "date": batch["date"],
    "month": batch["month"],
    "actual_product": batch["product"],
    "predicted_queue": classes[ranked[:, 0]],
    "second_choice": classes[ranked[:, 1]],
    "confidence": proba.max(axis=1),
})
decisions["decision"] = np.where(decisions["confidence"] >= threshold, "auto_route", "human_review")
decisions["correct"] = decisions["predicted_queue"] == decisions["actual_product"]
decisions["model_version"] = version
decisions["threshold"] = threshold
decisions["scoring_run_id"] = run_id
decisions["scored_at"] = scored_at

auto = decisions["decision"] == "auto_route"
print(f"Auto-routed {auto.mean():.1%} at {decisions.loc[auto, 'correct'].mean():.1%} accuracy; "
      f"{(~auto).sum():,} sent to human review")
display(decisions.drop(columns=["scoring_run_id"]).head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lesson 6: Writing with MERGE
# MAGIC
# MAGIC `MERGE` compares new rows with the existing table on a key (`complaint_id`) and decides per row what to do.
# MAGIC Here: insert rows that aren't already there, and leave existing ones alone.
# MAGIC Combined with the anti join, this is belt and braces: even if two runs overlapped, a complaint could never be written twice.
# MAGIC `MERGE` can also **update** matching rows (`WHEN MATCHED THEN UPDATE`), which is how tables get upserted.
# MAGIC
# MAGIC The new rows are first cast to the table's exact column types, so the write matches the contract from Lesson 3.

# COMMAND ----------

target_schema = spark.table(DECISIONS).schema
new_rows = spark.createDataFrame(decisions).select(
    [F.col(field.name).cast(field.dataType).alias(field.name) for field in target_schema])
new_rows.createOrReplaceTempView("new_decisions")

spark.sql(f"""
MERGE INTO {DECISIONS} AS target
USING new_decisions AS source
ON target.complaint_id = source.complaint_id
WHEN NOT MATCHED THEN INSERT *
""")
print("Decisions table now holds", spark.table(DECISIONS).count(), "complaints")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lesson 7: Logging each run
# MAGIC
# MAGIC One row per run in `scoring_runs`: when it ran, which model version, how many complaints, and how much was automated.
# MAGIC This is **operational monitoring**: if a run scores zero complaints, or the automated share suddenly drops, this table shows it.
# MAGIC Writing with `mode("append")` simply adds rows; that's safe here because each run has a unique ID.
# MAGIC
# MAGIC Run this whole notebook again a couple of times: each run picks up the next batch, until there's nothing left
# MAGIC and the run exits early with "Nothing new to score".

# COMMAND ----------

run_summary = pd.DataFrame([{
    "scoring_run_id": run_id,
    "scored_at": scored_at,
    "model_version": version,
    "threshold": threshold,
    "complaints_scored": len(decisions),
    "auto_routed_share": float(auto.mean()),
    "auto_routed_accuracy": float(decisions.loc[auto, "correct"].mean()) if auto.any() else None,
    "remaining_after_run": waiting - len(decisions),
}])
target_runs = spark.table(RUNS).schema
(spark.createDataFrame(run_summary)
      .select([F.col(f.name).cast(f.dataType).alias(f.name) for f in target_runs])
      .write.mode("append").saveAsTable(RUNS))

display(spark.table(RUNS).orderBy("scored_at"))
