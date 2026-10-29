# Stage 4: medallion pipeline for the complaint triage project (Lakeflow Spark Declarative Pipelines)
#
# This file is pipeline source code, not a notebook. You don't run it cell by cell:
# Lakeflow reads the decorators, works out which table depends on which, and builds them in order.
#
#   volume CSVs --Auto Loader--> bronze_complaints_stream   (raw, incremental)
#                                      |
#                                silver_complaints          (cleaned, deduplicated, quality rules)
#                                 /            \
#                gold_training_complaints    gold_monthly_mix
#                (train/val/test labels)     (product mix by month, for monitoring)

from pyspark import pipelines as dp
from pyspark.sql import Window
from pyspark.sql import functions as F

RAW_PATH = "/Volumes/workspace/complaint_triage/raw_files/"
QUEUES = [
    "Credit reporting", "Debt collection", "Checking or savings account", "Credit card",
    "Money transfer or virtual currency", "Mortgage", "Other", "Student loan", "Vehicle loan or lease",
]
QUEUE_LIST_SQL = ", ".join(f"'{q}'" for q in QUEUES)


# ---------------------------------------------------------------------------
# BRONZE: raw files in, untouched, with lineage columns.
# @dp.table on a streaming read creates a STREAMING TABLE. Auto Loader ("cloudFiles")
# remembers which files it has already loaded, so each run only processes NEW files.
# Drop a third CSV into the volume and rerun: only that file is read.
# ---------------------------------------------------------------------------
@dp.table(comment="Raw complaint CSVs loaded incrementally from the volume by Auto Loader.")
def bronze_complaints_stream():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("cloudFiles.inferColumnTypes", "true")
        .option("header", "true")
        .option("multiLine", "true")      # narratives contain line breaks
        .option("escape", '"')            # pandas escapes quotes by doubling them
        .load(RAW_PATH)
        .select(
            "*",
            F.col("_metadata.file_name").alias("source_file"),   # which file each row came from
            F.current_timestamp().alias("ingested_at"),
        )
    )


# ---------------------------------------------------------------------------
# SILVER: cleaned, deduplicated, and checked by EXPECTATIONS.
# @dp.materialized_view on a batch read creates a MATERIALIZED VIEW: a table the pipeline
# keeps up to date with its source.
#
# Expectations are data quality rules declared on the table:
#   @dp.expect          -> keep failing rows, but count them (a warning)
#   @dp.expect_or_drop  -> remove failing rows, and count them
# The pipeline records how many rows passed and failed each rule on every run.
# ---------------------------------------------------------------------------
@dp.materialized_view(comment="Cleaned, deduplicated complaints with data quality rules.")
@dp.expect_or_drop("narrative_present", "narrative IS NOT NULL AND length(trim(narrative)) > 0")
@dp.expect_or_drop("valid_date", "date IS NOT NULL")
@dp.expect("known_queue", f"product IN ({QUEUE_LIST_SQL})")
@dp.expect("parsed_cleanly", "_rescued_data IS NULL")
def silver_complaints():
    df = spark.read.table("bronze_complaints_stream")

    # Same normalization as the Python project: lowercase, tag dollar amounts,
    # remove XXXX redaction masks, collapse whitespace.
    text_key = F.lower(F.substring(F.col("narrative"), 1, 4000))
    text_key = F.regexp_replace(text_key, r"\{\$?[\d,]*\.?\d*\}", " dollaramount ")
    text_key = F.regexp_replace(text_key, r"\bx{2,}\b", " ")
    text_key = F.trim(F.regexp_replace(text_key, r"\s+", " "))

    # Deduplicate template letters: keep the earliest copy of each normalized narrative.
    # Rows with no narrative get their own group so they reach the expectation and are counted.
    group = F.coalesce(F.col("text_key"), F.concat(F.lit("missing-"), F.col("complaint_id").cast("string")))
    earliest_first = Window.partitionBy(group).orderBy("date", "complaint_id")

    return (
        df.withColumn("date", F.to_date("date"))
        .withColumn("text_key", text_key)
        .withColumn("copy_number", F.row_number().over(earliest_first))
        .filter("copy_number = 1")
        .withColumn("source_period",
                    F.when(F.col("source_file").contains("2026"), "2026").otherwise("2024-2025"))
        .withColumn("month", F.date_format("date", "yyyy-MM"))
        .withColumn("words", F.size(F.split(F.trim(F.col("narrative")), r"\s+")))
        .drop("copy_number", "text_key")
    )


# ---------------------------------------------------------------------------
# GOLD 1: training-ready data with the time-based split already assigned,
# so every later stage (training, scoring) uses exactly the same split.
# Oldest 65% train, next 15% validation, newest 20% test; 2026 is the label-shift holdout.
# ---------------------------------------------------------------------------
@dp.materialized_view(comment="Model-ready complaints with a time-based train/validation/test split.")
def gold_training_complaints():
    silver = spark.read.table("silver_complaints")
    by_time = Window.partitionBy("source_period").orderBy("date", "complaint_id")
    position = F.percent_rank().over(by_time)
    split = (F.when(F.col("source_period") == "2026", "holdout_2026")
              .when(position < 0.65, "train")
              .when(position < 0.80, "validation")
              .otherwise("test"))
    return silver.select("complaint_id", "date", "month", "product", "narrative", "words",
                         split.alias("split"))


# ---------------------------------------------------------------------------
# GOLD 2: product mix by month, the table behind the stage 7 monitoring dashboard.
# Your label-shift finding becomes something a dashboard can watch every month.
# ---------------------------------------------------------------------------
@dp.materialized_view(comment="Complaints and product share by month, for drift monitoring.")
def gold_monthly_mix():
    counts = spark.read.table("silver_complaints").groupBy("month", "product").agg(
        F.count("*").alias("complaints"))
    month_total = F.sum("complaints").over(Window.partitionBy("month"))
    return counts.withColumn("share_pct", F.round(100 * F.col("complaints") / month_total, 1))
